"""执行子 Agent。

边界（必须强制）：
- 不做任务规划、不做多步推理、不与用户对话、不读取全局上下文与长期记忆。
- 仅接收单个 subtask 字典描述（name + goal + required_skills + required_mcp_tools）。
- 完成单步动作后回传 ExecutorResult 形式的文本给主管 Agent。

实现：
1. 如 subtask.required_skills 中包含懒加载 skill，按需载入 SkillRegistry 取得提示，作为 system 提示注入。
2. MCP 工具通过工具绑定机制自动注入 LLM 可调用列表，不再手动预调。
3. ReAct 循环处理 LLM 主动发起的工具调用，完成后再进行结构化输出。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from nekoagent.agents.shared.schemas import ExecutorResult, SubTask
from nekoagent.agents.shared.output_constraints import invoke_with_retry
from nekoagent.config.loader import Config, get_config
from nekoagent.llm.factory import get_llm
from nekoagent.observability.exceptions import (
    LLMConfigError,
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

    def _build_llm_with_tools(self):
        """构建 LLM 实例，绑定 RAG + MCP 工具。"""
        try:
            from nekoagent.mcp.binding import bind_agent_tools_to_llm
            return bind_agent_tools_to_llm(get_llm("executor_agent", self.cfg), "executor_agent", self.cfg)
        except (LLMConfigError, NekoAgentError):
            raise

    def _react_until_done(self, llm, messages: list, max_iterations: int = 2) -> list:
        """ReAct 循环：处理工具调用，直到 LLM 返回非工具调用的最终响应。"""
        from nekoagent.mcp.binding import AgentMCPClient

        mcp_client = None
        iteration = 0
        while iteration < max_iterations:
            response = llm.invoke(messages)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                break

            for tc in tool_calls:
                tc_name = tc.get("name", "")
                tc_args = tc.get("args", {})
                raw_result = self._execute_any_tool(tc_name, tc_args, mcp_client)
                messages.append(SystemMessage(
                    content=f"工具执行结果：{raw_result}\n请基于此结果继续完成子任务。"
                ))
                _log.info("ReAct: 执行工具 %s 完成", tc_name)

            iteration += 1

        if iteration >= max_iterations:
            _log.warning("Executor ReAct 达到最大迭代次数 %d", max_iterations)

        return messages

    def _execute_any_tool(self, tool_name: str, args: dict, mcp_client) -> str:
        """执行 RAG 或 MCP 工具，返回结果文本。"""
        from nekoagent.rag_tools import execute_rag_tool as _exec_rag
        from nekoagent.mcp.binding import get_mcp_tool_registry

        registry = get_mcp_tool_registry()
        for server_name, tools in registry.items():
            if tool_name in tools:
                if mcp_client is None:
                    from nekoagent.mcp.binding import AgentMCPClient
                    mcp_client = AgentMCPClient("executor_agent", self.cfg)
                try:
                    return mcp_client.call_tool(server_name, tool_name, args)
                except Exception as exc:
                    return f"[MCP 错误] {exc}"

        result = _exec_rag(tool_name, args)
        if result and len(result) >= 10:
            return result
        return f"未找到工具 '{tool_name}'"

    # ---------------- 子任务执行 ----------------
    def run_subtask(self, thread_id: str, session_id: str, subtask: SubTask) -> str:
        """单步执行：拼上下文 + ReAct 工具循环 + 调 LLM 结构化输出；返回 ExecutorResult.result 文本。"""

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

        # 告知 LLM 可用的 MCP 工具
        tool_hints = self._build_tool_hints(subtask)
        if tool_hints:
            parts.append(tool_hints)

        system_prompt = "\n\n".join(parts)

        user_content = (
            f"子任务名：{subtask.name}\n"
            f"目标：{subtask.goal}\n"
        )
        if subtask.context:
            user_content += f"上下文：\n{subtask.context[:2000]}\n"
        user_content += "请根据需要调用可用工具，然后输出完成本子任务的结果摘要（中文）。"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        try:
            llm = self._build_llm_with_tools()
        except (LLMConfigError, NekoAgentError):
            raise

        # ReAct 循环：让 LLM 自主调用工具
        messages = self._react_until_done(llm, messages)
        _log.info("ReAct 循环完成 thread_id=%s", thread_id)

        # 结构化输出最终结果
        structured_prompt = (
            "请输出符合 ExecutorResult 的 json 对象，仅含 result 与 tokens_used 与 notes 字段。\n"
            "参考 json 格式：\n"
            '{"result": "执行结果摘要", "tokens_used": 0, "notes": ""}'
        )
        messages.append(SystemMessage(content=structured_prompt))

        result = invoke_with_retry(
            llm,
            ExecutorResult,
            messages,
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

    def _build_tool_hints(self, subtask: SubTask) -> str:
        """提取当前 Agent 可见的 MCP + RAG 工具列表，注入 system prompt 供 LLM 参考。"""
        from nekoagent.mcp.binding import get_mcp_tool_registry
        from nekoagent.rag_tools import get_all_rag_lib_metas

        hints = []

        # MCP 工具
        registry = get_mcp_tool_registry()
        agent_cfg = self.cfg.agents.get("executor_agent")
        if agent_cfg is not None:
            allowed = set(agent_cfg.mcp_servers)
            for server_name, tools in registry.items():
                if server_name not in allowed:
                    continue
                for tool_name, tool_def in tools.items():
                    desc = tool_def.get("description", "") or tool_name
                    hints.append(f"  - {tool_name}: {desc}")

        # RAG 工具
        for lib_name, lib_meta in get_all_rag_lib_metas().items():
            desc = lib_meta.get("description", "") or lib_name
            hints.append(f"  - search_{lib_name}: {desc}")

        if not hints:
            return ""
        return (
            "可用工具（仅在子任务目标明确匹配时才调用，不相关时不要使用）：\n"
            + "\n".join(hints)
        )


_executor_singleton: ExecutorAgent | None = None


def get_executor_agent(cfg: Config | None = None) -> ExecutorAgent:
    global _executor_singleton
    if _executor_singleton is None:
        _executor_singleton = ExecutorAgent(cfg or get_config())
    return _executor_singleton


def dispatch_subtask(thread_id: str, session_id: str, subtask: SubTask) -> str:
    """供 supervisor 注入为 executor_dispatch 的便利函数入口。"""

    return get_executor_agent().run_subtask(thread_id, session_id, subtask)
