"""执行子 Agent。

边界（必须强制）：
- 不做任务规划、不做多步推理、不与用户对话、不读取全局上下文与长期记忆。
- 仅接收单个 subtask 字典描述（name + goal + required_skills + required_mcp_tools）。
- 完成单步动作后回传 ExecutorResult 形式的文本给主管 Agent。

实现：
1. 如 subtask.required_skills 中包含懒加载 skill，按需载入 SkillRegistry 取得提示，作为 system 提示注入。
2. 如 subtask.required_mcp_tools 中包含工具，经 MCPClient 池调用对应 MCP 服务器。
3. 调用 LLM（profile=executor）生成单步结果，再 wrap 为 ExecutorResult。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from nekoagent.agents.shared.schemas import ExecutorResult, SubTask
from nekoagent.agents.shared.output_constraints import invoke_with_retry
from nekoagent.config.loader import Config, get_config
from nekoagent.llm.factory import get_llm
from nekoagent.observability.exceptions import (
    LLMConfigError,
    MCPError,
    NekoAgentError,
    SkillError,
)
from nekoagent.observability.logging import get_logger

_log = get_logger("agent.executor")


class ExecutorAgent:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or get_config()

    # ---------------- 工具/skill 引用解构 ----------------
    def _resolve_skill(self, skill_name: str) -> str:
        """按需懒加载 skill 描述文本；找不到返回空字符串。"""

        try:
            from nekoagent.skills import get_skill_registry
            registry = get_skill_registry(self.cfg)
            if not registry.knows(skill_name):
                _log.debug("skill 不存在，跳过：%s", skill_name)
                return ""
            skill = registry.load(skill_name)
            if skill is None:
                return ""
            return (skill.get("system_prompt") or "").strip()
        except Exception:
            _log.debug("skill 加载失败，跳过：%s", skill_name)
            return ""

    def _call_mcp_tool(self, tool_ref: str, args_text: str) -> str:
        tool_ref = tool_ref.strip()
        if "." not in tool_ref:
            raise MCPError(f"MCP 工具引用格式错误 '{tool_ref}'，应为 'server.tool' 形式")
        server_name, tool_name = tool_ref.split(".", 1)
        try:
            from nekoagent.mcp import get_mcp_client_for_agent
            client = get_mcp_client_for_agent("executor_agent", self.cfg)
        except Exception as exc:
            raise MCPError(f"MCP 客户端初始化失败：{exc}", cause=exc) from exc
        if server_name not in client.allowed_servers():
            raise MCPError(f"MCP 服务器 '{server_name}' 未在 YAML 授权给 executor_agent")
        try:
            import json
            args = json.loads(args_text) if args_text.strip() else {}
        except json.JSONDecodeError as exc:
            raise MCPError(f"MCP 工具 {tool_ref} 参数非 JSON：{exc}", cause=exc) from exc
        try:
            return client.call_tool(server_name, tool_name, args)
        except Exception as exc:
            raise MCPError(f"MCP 工具调用 {tool_ref} 失败：{exc}", cause=exc) from exc

    # ---------------- 子任务执行 ----------------
    def run_subtask(self, thread_id: str, session_id: str, subtask: SubTask) -> str:
        """单步执行：拼上下文 + 调 LLM；返回 ExecutorResult.result 文本。"""

        # 拼装 system
        parts = [self.cfg.get_persona_cached("executor_agent").system_prompt]
        for skill_name in (subtask.required_skills or []):
            try:
                skill_text = self._resolve_skill(skill_name)
                if skill_text:
                    parts.append(f"[skill: {skill_name}]\n{skill_text}")
            except SkillError as exc:
                _log.warning("skill 加载失败（不阻塞，仅记日志）：thread_id=%s skill=%s err=%s",
                             thread_id, skill_name, exc.friendly_message)
        system_prompt = "\n\n".join(parts)

        # 调 MCP（按需要）—— MVP 简化：如果 subtask.required_mcp_tools 非空，先调用第一个，结果回写为事实
        tool_results: list[str] = []
        for tool_ref in (subtask.required_mcp_tools or []):
            try:
                # MVP 无从参数结构 → 调用时传空对象，要求 server docs 提供示例
                text = self._call_mcp_tool(tool_ref, "")
                tool_results.append(f"{tool_ref}: {text[:500]}")
            except MCPError as exc:
                _log.warning("MCP 调用失败：thread_id=%s tool=%s err=%s",
                             thread_id, tool_ref, exc.friendly_message)
                tool_results.append(f"{tool_ref}: [FAILURE] {exc.friendly_message}")

        user_content = (
            f"子任务名：{subtask.name}\n"
            f"目标：{subtask.goal}\n"
        )
        if subtask.context:
            user_content += f"上下文：\n{subtask.context[:2000]}\n"
        if tool_results:
            user_content += "工具返回事实：\n" + "\n".join(tool_results) + "\n"
        user_content += "请基于以上信息，输出完成本子任务的结果摘要（中文）。"

        try:
            llm = get_llm("executor_agent", self.cfg)
        except (LLMConfigError, NekoAgentError):
            raise

        # 若 MCP 有错，仍允许执行 LLM 但带 [FAILURE] 事实给后续异常处理判断
        result = invoke_with_retry(
            llm,
            ExecutorResult,
            [
                SystemMessage(content=system_prompt + "\n请输出符合 ExecutorResult 的 JSON 对象，仅含 result 与 tokens_used 与 notes 字段。"),
                HumanMessage(content=user_content),
            ],
            cfg=self.cfg,
            on_giveup_message="子 Agent 无法完成结构化输出，已触发友好回退。",
        )
        try:
            from nekoagent.observability.token_counter import estimate_tokens, record_task_tokens
            tok_count = estimate_tokens(subtask.goal) + estimate_tokens(result.result)
            record_task_tokens(session_id, tok_count, agent_role="executor_agent")
        except Exception:
            pass

        # 记录执行日志
        try:
            from datetime import datetime
            from nekoagent.memory.mysql_checkpointer import get_engine
            from nekoagent.memory.schema import get_metadata
            eng = get_engine(self.cfg)
            tbl = get_metadata().tables["executor_logs"]
            with eng.begin() as c:
                c.execute(tbl.insert().values(
                    thread_id=thread_id,
                    subtask_name=subtask.name,
                    subtask_goal=subtask.goal,
                    skills_used=", ".join(subtask.required_skills or []),
                    tools_used=", ".join(subtask.required_mcp_tools or []),
                    result=f"context:{subtask.context[:500]} | result:{result.result[:500]}",
                    status="done",
                    created_at=datetime.utcnow(),
                ))
        except Exception:
            pass
        return result.result


_executor_singleton: ExecutorAgent | None = None


def get_executor_agent(cfg: Config | None = None) -> ExecutorAgent:
    global _executor_singleton
    if _executor_singleton is None:
        _executor_singleton = ExecutorAgent(cfg or get_config())
    return _executor_singleton


def dispatch_subtask(thread_id: str, session_id: str, subtask: SubTask) -> str:
    """供 supervisor 注入为 executor_dispatch 的便利函数入口。"""

    return get_executor_agent().run_subtask(thread_id, session_id, subtask)