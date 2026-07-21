"""主 Agent：固定人设对话 + /task 类型判定与核心任务提炼 + RAG 注入 + 主管回传润色。

设计：
- 由 `nekoagent.llm.factory.get_llm('main_agent')` 实例化对话型模型。
- 人设 system_prompt 从 `personas/main.yaml` 加载，支持热更新。
- `/task` 前缀触发 distillation：使用 `config/system_prompts/task_distill.md` 作为系统提示调用 LLM。
- 提炼后将任务描述交给 `TaskSubmitter`（由 task-scheduling 模块注入）。
- 主管回传（计划/异常/结果）通过受控函数 `present_plan_to_user`/`present_exception_to_user`/`present_result_to_user` 集中润色。

界面层订阅这些回调以渲染 UI；UI 在 Group 16 接入。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterable

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from nekoagent.config.loader import Config, get_config
from nekoagent.llm.factory import get_llm
from nekoagent.memory.rag.pipeline import inject_or_skip
from nekoagent.memory.stores import get_user_memory_store
from nekoagent.memory.summarize import summarize_if_needed
from nekoagent.observability.exceptions import (
    LLMConfigError,
    NekoAgentError,
    format_friendly_error,
)
from nekoagent.observability.logging import get_logger

_log = get_logger("agent.main")

_TASK_PREFIX = "/task"

# 全局主管任务提交器
_task_submitter: Callable[[str, str], dict[str, Any]] | None = None
_async_task_submitter: Callable[[str, str], Any] | None = None


def register_task_submitter(submitter: Callable[[str, str], dict[str, Any]]) -> None:
    """由 task-scheduling 模块注册：`submitter(session_id, distilled_task_summary) -> {'thread_id': ...}`"""

    global _task_submitter
    _task_submitter = submitter
    _log.info("主 Agent 已绑定 task_submitter")


def register_async_task_submitter(submitter: Callable[[str, str], Any]) -> None:
    global _async_task_submitter
    _async_task_submitter = submitter
    _log.info("主 Agent 已绑定 async task_submitter")


_main_agent_singleton: "MainAgent | None" = None


def get_main_agent(cfg: Config | None = None) -> "MainAgent":
    global _main_agent_singleton
    if _main_agent_singleton is None:
        _main_agent_singleton = MainAgent(cfg or get_config())
    return _main_agent_singleton


def _row_to_message(row: dict[str, Any]) -> BaseMessage:
    role = row["role"]
    content = row["content"]
    extra = {"nekoagent_msg_id": row["id"]} if row.get("id") else {}
    if role == "user":
        return HumanMessage(content=content, additional_kwargs=extra)
    if role == "assistant":
        return AIMessage(content=content, additional_kwargs=extra)
    if role == "system":
        return SystemMessage(content=content, additional_kwargs=extra)
    # tool / 等
    return AIMessage(content=content, additional_kwargs=extra)


class MainAgent:
    """主 Agent 实例。轻封装；持久化跨会话记忆由 MainUserMemoryStore 处理。"""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._persona_cache: dict[str, str] = {}

    # ---------- 人设 ----------
    def _persona_system_prompt(self) -> str:
        persona = self.cfg.get_persona_cached("main_agent")
        return persona.system_prompt

    def reload_persona(self) -> None:
        self.cfg.load_persona("main_agent")
        _log.info("主 Agent 人设热更新完成")

    # ---------- /task 判定 + 提炼 ----------
    @staticmethod
    def is_task_request(text: str) -> bool:
        if not text:
            return False
        stripped = text.lstrip()
        if not stripped.lower().startswith(_TASK_PREFIX.lower()):
            return False
        rest = stripped[len(_TASK_PREFIX):].lstrip()
        # 必须有实质内容才认定是任务请求
        return bool(rest)

    def _distill_prompt(self) -> str:
        """加载 config/system_prompts/task_distill.md 内容作为系统提示。"""

        path = Path("config/system_prompts/task_distill.md")
        if not path.exists():
            path = Path("config/system_prompts") / "task_distill.md"
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            _log.warning("task_distill.md 未找到，回退默认提炼提示")
            return "请将用户消息提炼为对外可执行的核心任务描述，仅输出提炼后的内容。"

    def distill_task(self, raw_user_text: str) -> str:
        from nekoagent.observability.exceptions import LLMConnectionError
        prompt_text = self._distill_prompt()
        # 去掉 /task 前缀
        body = raw_user_text.lstrip()[len(_TASK_PREFIX):].lstrip()
        try:
            llm = get_llm("main_agent", self.cfg)
        except (LLMConfigError, LLMConnectionError):
            raise
        messages = [
            SystemMessage(content=prompt_text),
            HumanMessage(content=body),
        ]
        try:
            response = llm.invoke(messages)
        except Exception as exc:
            raise NekoAgentError(
                "任务提炼调用 LLM 失败，请稍后重试。",
                cause=exc,
            ) from exc
        content = response.content if isinstance(response, BaseMessage) else str(response)
        if isinstance(content, list):
            # 部分模型返回 list of part；尽力取文本
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        distilled = content.strip()
        _log.info("提炼任务：%r -> %r", body[:80], distilled[:80])
        # `/task` 场景下 raw_user_text 可能不含 session_id，这里不记录 token（在 handle_user_message 中处理）
        return distilled

    # ---------- 主管转交 ----------
    def submit_task(self, session_id: str, distilled_task: str) -> dict[str, Any]:
        """提交任务至主管 Agent。返回 {thread_id, status}。"""

        if _task_submitter is None and _async_task_submitter is None:
            raise NekoAgentError("任务尚未就绪：主管 Agent 未配置（task_submitter 未注册）。")
        if _task_submitter is not None:
            return _task_submitter(session_id, distilled_task)
        # async 路径
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            raise NekoAgentError("无事件循环，且仅注册了 async submitter。")
        return loop.run_until_complete(_async_task_submitter(session_id, distilled_task))

    # ---------- 对话主流程 ----------
    def chat(self, session_id: str, user_text: str) -> str:
        """同步对话：返回主 Agent 完整文本回复。"""

        full = ""
        for chunk in self.chat_stream(session_id, user_text):
            full += chunk
        return full or "（空回复，请稍后重试喵~）"

    def chat_stream(self, session_id: str, user_text: str) -> Iterable[str]:
        """流式对话生成器，逐字 yield 文本块；末尾的 yield 等于完整回复。"""

        try:
            llm = get_llm("main_agent", self.cfg)
        except (LLMConfigError, NekoAgentError):
            raise

        messages = self._build_messages(session_id, user_text)
        # 持久化用户消息
        try:
            store = get_user_memory_store()
            user_row_id = store.add_message(
                session_id=session_id, role="user", content=user_text, tokens=self._rough_tokens(user_text)
            )
        except Exception as exc:
            _log.warning("用户消息持久化失败（不影响回复）：%s", exc)
            user_row_id = None

        full_text = ""
        try:
            for chunk in llm.stream(messages):
                text = chunk.content if isinstance(chunk, BaseMessage) else str(chunk)
                if isinstance(text, list):
                    text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
                if not text:
                    continue
                full_text += text
                yield text
        except Exception as exc:
            raise NekoAgentError("主 Agent 调用 LLM 失败，请稍后重试。", cause=exc) from exc

        if not full_text.strip():
            full_text = "（空回复，请稍后重试喵~）"

        try:
            store.add_message(
                session_id=session_id,
                role="assistant",
                content=full_text,
                tokens=self._rough_tokens(full_text),
            )
        except Exception as exc:
            _log.warning("回复持久化失败（已返回给用户）：%s", exc)

        try:
            from nekoagent.observability.token_counter import record_dialogue_tokens
            record_dialogue_tokens(session_id, self._rough_tokens(full_text), agent_role="main_agent")
        except Exception:
            pass

        # RAG 与总结中间件不在流式 invoke 前生效；在外层 chat 调用 hook 已处理。未来可下沉。

    def _build_messages(self, session_id: str, user_text: str) -> list[BaseMessage]:
        """构造一次对话的完整 messages：人设 + 历史 + RAG记忆 + 新 user message + 总结裁剪。"""

        try:
            store = get_user_memory_store()
            history_rows = store.fetch_active_messages(session_id, limit=100)
        except Exception as exc:
            _log.warning("主 Agent 历史消息加载失败（短路为空历史）：%s", exc)
            history_rows = []

        history_messages = [SystemMessage(content=self._persona_system_prompt())]
        history_messages.extend(_row_to_message(r) for r in history_rows)

        # Token 门槛触发总结：传入 messages 估算与裁剪
        try:
            history_messages = summarize_if_needed(history_messages, session_id=session_id, cfg=self.cfg)
        except NekoAgentError as exc:
            _log.warning("总结中间件失败（不阻塞对话）：%s", exc.friendly_message)

        # RAG 注入（仅当达到相似度门槛）
        try:
            history_messages = inject_or_skip(history_messages, user_text, self.cfg)
        except NekoAgentError as exc:
            _log.warning("RAG 注入失败（不阻塞对话）：%s", exc.friendly_message)

        history_messages.append(HumanMessage(content=user_text))
        return history_messages

    @staticmethod
    def _rough_tokens(text: str) -> int:
        """快速估算 token 数（粗略：中文按 2 倍字符数）。"""

        if not text:
            return 0
        return int(len(text) * 2.0)

    # ---------- Token 记录 + 上下文辅助 ----------
    def _record_llm_call(self, session_id: str, messages: list, response_text: str, agent_role: str = "main_agent") -> None:
        try:
            from nekoagent.observability.token_counter import estimate_tokens, record_dialogue_tokens
            inp = sum(estimate_tokens(getattr(m, 'content', str(m)) or "") for m in messages)
            out = estimate_tokens(response_text)
            record_dialogue_tokens(session_id, inp + out, agent_role=agent_role)
        except Exception:
            pass

    def _build_callback_context(self, session_id: str) -> str:
        """获取最近对话历史供回调使用。"""
        try:
            from nekoagent.memory.stores import get_user_memory_store
            store = get_user_memory_store()
            rows = store.fetch_active_messages(session_id, limit=20)
            if not rows:
                return ""
            parts = []
            for r in rows:
                role = r.get("role", "")
                content = r.get("content", "")
                if role == "user":
                    parts.append(f"用户：{content[:200]}")
                elif role == "assistant":
                    parts.append(f"助手：{content[:200]}")
            return "\n".join(parts[-10:])  # 最近10条
        except Exception:
            return ""

    # ---------- 用户决策回应：主 Agent 根据人设告知用户已收到决定 ----------
    def generate_decision_ack(self, action: str, modification: str = "", session_id: str = "") -> str:
        context_map = {
            "execute": "用户已批准执行任务计划",
            "modify": f"用户提出了修改意见：{modification}",
            "cancel": "用户取消了任务",
            "provide_info": f"用户提供了补充信息：{modification[:100]}",
        }
        ctx = context_map.get(action, f"用户操作：{action}")
        history = self._build_callback_context(session_id)
        try:
            llm = get_llm("main_agent", self.cfg)
            msgs = [SystemMessage(content=self._persona_system_prompt())]
            if history:
                msgs.append(SystemMessage(content=f"对话历史：\n{history}"))
            msgs.append(SystemMessage(content=f"{ctx}。请用你的人设风格简要告知用户已收到其决定，已经通知任务主管Agent，无需详述任务细节。"))
            response = llm.invoke(msgs)
            text = response.content if isinstance(response, BaseMessage) else str(response)
            self._record_llm_call(session_id, msgs, text)
            return text
        except Exception:
            fb = {"execute": "好的，收到！马上安排执行喵~", "modify": "好的，收到修改意见啦，让主管重新规划一下喵~", "cancel": "好哒，已取消任务喵~", "provide_info": "好的，收到你补充的信息啦~"}
            return fb.get(action, "好的喵~")

    def generate_info_request_answer(self, question: str, session_id: str = "") -> str:
        try:
            llm = get_llm("main_agent", self.cfg)
            msgs = [SystemMessage(content=self._persona_system_prompt())]
            history = self._build_callback_context(session_id)
            if history:
                msgs.append(SystemMessage(content=f"对话历史：\n{history}"))
            msgs.append(SystemMessage(content=f"任务主管 Agent 在执行子任务时需要补充以下信息：\n{question}\n\n请根据你的人设和知识直接给出该信息的答案。仅输出答案内容本身，不要询问用户。"))
            resp = llm.invoke(msgs)
            text = resp.content if isinstance(resp, BaseMessage) else str(resp)
            self._record_llm_call(session_id, msgs, text)
            return text
        except Exception:
            return ""

    def generate_info_request_feedback(self, action: str, detail: str = "", session_id: str = "") -> str:
        context_map = {
            "provide_info": f"用户提供了补充信息：{detail[:100]}",
            "auto_decide": f"用户让主Agent代为决定，主Agent的答案是：{detail[:100]}",
            "cancel": "用户取消了信息请求",
        }
        ctx = context_map.get(action, f"用户操作：{action}")
        history = self._build_callback_context(session_id)
        try:
            llm = get_llm("main_agent", self.cfg)
            msgs = [SystemMessage(content=self._persona_system_prompt())]
            if history:
                msgs.append(SystemMessage(content=f"对话历史：\n{history}"))
            msgs.append(SystemMessage(content=f"{ctx}。请用你的人设风格简要告知用户已收到该操作，相关信息已传达给任务主管Agent。"))
            resp = llm.invoke(msgs)
            text = resp.content if isinstance(resp, BaseMessage) else str(resp)
            self._record_llm_call(session_id, msgs, text)
            return text
        except Exception:
            fb = {"provide_info": "好的，收到你补充的信息啦，我马上告诉主管~", "auto_decide": "人家已经帮你决定啦，这就告诉主管~", "cancel": "好的，已取消~"}
            return fb.get(action, "好的喵~")

    def generate_info_request_notification(self, question: str, session_id: str = "") -> str:
        try:
            llm = get_llm("main_agent", self.cfg)
            msgs = [SystemMessage(content=self._persona_system_prompt())]
            history = self._build_callback_context(session_id)
            if history:
                msgs.append(SystemMessage(content=f"对话历史：\n{history}"))
            msgs.append(SystemMessage(content=f"执行子任务时需要用户补充以下信息：{question}。请用你的人设风格告知用户并请求提供该信息。"))
            resp = llm.invoke(msgs)
            text = resp.content if isinstance(resp, BaseMessage) else str(resp)
            self._record_llm_call(session_id, msgs, text)
            return text
        except Exception:
            return f"人家需要你补充一下信息呢：{question}"

    # ---------- 主管回传通道：主 Agent 润色后交给 UI / 用户 ----------
    def present_plan_to_user(self, plan: Any, session_id: str) -> str:
        try:
            llm = get_llm("main_agent", self.cfg)
            subtasks = getattr(plan, "subtasks", []) or []
            plan_summary = getattr(plan, "task_summary", "")
            subtask_lines = []
            for i, sub in enumerate(subtasks, 1):
                subtask_lines.append(f"  {i}. {getattr(sub, 'name', '')} — {getattr(sub, 'goal', '')}")
            subtask_text = "\n".join(subtask_lines)
            prompt = (
                f"任务主管Agent为用户任务「{plan_summary}」生成了以下规划方案：\n"
                f"{subtask_text}\n\n"
                f"请用你的人设风格向用户呈现这个规划方案，并询问用户是否要执行、修改或取消。"
            )
            msgs = [SystemMessage(content=self._persona_system_prompt())]
            history = self._build_callback_context(session_id)
            if history:
                msgs.append(SystemMessage(content=f"对话历史：\n{history}"))
            msgs.append(SystemMessage(content=prompt))
            response = llm.invoke(msgs)
            text = response.content if isinstance(response, BaseMessage) else str(response)
            self._record_llm_call(session_id, msgs, text)
            return text or f"任务主管已规划完毕：\n{subtask_text}\n请选择：执行 / 修改 / 取消"
        except Exception as exc:
            _log.warning("计划润色 LLM 调用失败，回退模板：%s", exc)
            subtasks = getattr(plan, "subtasks", []) or []
            lines = [f"任务主管规划已完成：【{getattr(plan, 'task_summary', '')}】"]
            for i, sub in enumerate(subtasks, 1):
                lines.append(f"  {i}. {getattr(sub, 'name', '')} — {getattr(sub, 'goal', '')}")
            lines.append("请选择：执行 / 修改 / 取消")
            return "\n".join(lines)

    def present_exception_to_user(self, signal: Any, session_id: str) -> str:
        """子 Agent 抛异常时主 Agent 友好提示。"""

        failed = getattr(signal, "failed_subtask", "某步骤")
        summary = getattr(signal, "error_summary", "")
        candidates = getattr(signal, "candidates", ["retry", "skip", "modify", "cancel"])
        text = (
            f"呜……任务【{failed}】这边有点小问题呢。"
            f"刚才的错误如下：{summary}\n"
            f"你希望人家：【{' / '.join(candidates)}】？"
        )
        return text

    def present_result_to_user(self, result_text: str, session_id: str) -> str:
        """任务完成结果由主 Agent 润色后呈现。"""
        try:
            llm = get_llm("main_agent", self.cfg)
            prompt = (
                f"任务主管Agent为已完成工作, 返回以下任务执行结果：\n"
                f"{result_text}\n\n"
                f"请用你的人设风格向用户呈现这个结果。"
            )
            messages = [
                SystemMessage(content=self._persona_system_prompt()),
                SystemMessage(content=prompt),
            ]
            response = llm.invoke(messages)
            text = response.content if isinstance(response, BaseMessage) else str(response)
            return text or f"任务主管已完成工作：\n{result_text}\n"
        except Exception as exc:
            _log.warning("计划润色 LLM 调用失败，回退模板：%s", exc)
            return f"任务好啦，人家帮你整理好啦：\n{result_text}\n有什么想再调整的随时告诉我喵~"


# ---------- 用户消息处理入口 ----------


def handle_user_message(session_id: str, user_text: str) -> tuple[str, dict[str, Any] | None]:
    """UI 调用方统一入口；返回 (agent_reply_text, optional_task_submission_status)。"""

    agent = get_main_agent()
    store = get_user_memory_store()
    if agent.is_task_request(user_text):
        try:
            distilled = agent.distill_task(user_text)
            store.add_message(session_id=session_id, role="user", content=user_text, tokens=agent._rough_tokens(user_text))
            status = agent.submit_task(session_id, distilled)
            llm = get_llm("main_agent", agent.cfg)
            msgs = [
                SystemMessage(content=agent._persona_system_prompt()),
                HumanMessage(content=user_text),
                SystemMessage(content=f"用户提交了一个任务：「{distilled}」。该任务已提交给任务主管Agent处理。请用你的人设风格告知用户任务已收到、正在等待主管规划，不要自己执行或规划这个任务。"),
            ]
            response = llm.invoke(msgs)
            ack = response.content if isinstance(response, BaseMessage) else str(response)
            store.add_message(session_id=session_id, role="assistant", content=ack, tokens=agent._rough_tokens(ack))
            try:
                from nekoagent.observability.token_counter import estimate_tokens as _et, record_dialogue_tokens as _rdt
                _rdt(session_id, _et(user_text) + _et(ack), agent_role="main_agent")
            except Exception:
                pass
            return ack, {"distilled": distilled, "submission": status}
        except NekoAgentError as exc:
            return exc.friendly_message, None
        except Exception as exc:
            _log.exception("主 Agent /task 处理失败：%s", exc)
            return format_friendly_error(exc), None
    # 普通对话（chat_stream 内部已持久化 user + assistant 消息）
    try:
        reply = agent.chat(session_id, user_text)
        return (reply or "（主 Agent 暂时无法回复，喵~）"), None
    except NekoAgentError as exc:
        return exc.friendly_message, None
    except Exception as exc:
        return format_friendly_error(exc), None