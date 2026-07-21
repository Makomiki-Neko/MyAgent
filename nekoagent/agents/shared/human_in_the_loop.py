"""Human-in-the-loop 中间件 helper。

LangGraph 的 `interrupt(...) → Command(resume=...)` 语义为底座，本模块封装：
1. 统一 HITL 中断载荷（plan-ready / exception）
2. 中断点 → main agent pub/sub 通道（让主 Agent 即时拿到要呈现给用户的待办）
3. `resume(thread_id, decision)` 一键恢复 supervisor 子图继续推进
4. 重启后从 checkpointer 恢复的入口（按 thread_id 列出 pending HITL）

不在 supervisor/executor 节点里手工 sleep 轮询，由 LangGraph interrupt 原生完成挂起。
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from nekoagent.observability.exceptions import HITLAbortedError
from nekoagent.observability.logging import get_logger

_log = get_logger("agent.human-in_loop")

try:
    from langgraph.types import interrupt, Command  # type: ignore
except ImportError:  # pragma: no cover
    interrupt = None  # type: ignore
    Command = None  # type: ignore


class HITLKind(str, Enum):
    PLAN_READY = "plan_ready"
    EXCEPTION = "exception"


@dataclass
class HITLPayload:
    """中断时由 supervisor / executor 收集并通过 callback 推给 main agent 的载荷。"""

    thread_id: str
    kind: HITLKind
    plan: Any | None = None                       # TaskPlan 对象（kind=plan_ready）
    exception_signal: Any | None = None           # HitLExceptionSignal 对象（kind=exception）
    candidates: list[str] = field(default_factory=list)
    notes: str = ""


class HITLManager:
    """单例：登记 callback（main agent 注入），推送 pending payload，按 thread_id 索引。"""

    def __init__(self) -> None:
        self._callback: Callable[[HITLPayload], None] | None = None
        self._async_callback: Any | None = None
        self._pending: dict[str, HITLPayload] = {}        # thread_id -> latest payload

    def set_callback(self, cb: Callable[[HITLPayload], None]) -> None:
        self._callback = cb

    def set_async_callback(self, cb: Any) -> None:
        self._async_callback = cb

    def emit(self, payload: HITLPayload) -> None:
        """由 supervisor / executor 内部 interrupt 调用前调用，告知 main agent。"""

        self._pending[payload.thread_id] = payload
        if self._callback:
            try:
                self._callback(payload)
            except Exception as exc:
                _log.warning("HITL sync callback 抛错（被吞掉）：%s", exc)
        if self._async_callback:
            try:
                asyncio.create_task(self._async_callback(payload))  # type: ignore[arg-type]
            except RuntimeError:
                pass  # 没 loop 时退化为无 async callback

    def get_pending(self, thread_id: str) -> HITLPayload | None:
        return self._pending.get(thread_id)

    def clear_pending(self, thread_id: str) -> None:
        self._pending.pop(thread_id, None)

    def list_pending(self) -> list[HITLPayload]:
        return list(self._pending.values())

    def reset(self) -> None:
        self._pending.clear()
        self._callback = None
        self._async_callback = None


_hitl_manager: HITLManager | None = None


def get_hitl_manager() -> HITLManager:
    """单例入口。"""

    global _hitl_manager
    if _hitl_manager is None:
        _hitl_manager = HITLManager()
    return _hitl_manager


# ==================== 中断点辅助 ====================


def new_thread_id(prefix: str = "task") -> str:
    """构造 thread_id 供 supervisor 子图与 checkpointer 绑定。"""

    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def interrupt_for_plan(thread_id: str, plan: Any) -> Any:
    """supervisor 节点：plan 生成后挂起并要求用户决策。

    返回值 = 用户决策（"execute" | "modify" | "cancel" + 可选 modification）。
    实际 LangGraph 内部由 interrupt 完成，这里负责把 payload 推给 main agent。
    在非 LangGraph runnable context 中调用时退化为本地 mock，便于单测。
    """

    payload = HITLPayload(
        thread_id=thread_id,
        kind=HITLKind.PLAN_READY,
        plan=plan,
        candidates=["execute", "modify", "cancel"],
        notes="任务主管 Agent 已生成完整计划，等待用户决定是否执行。",
    )
    get_hitl_manager().emit(payload)
    if interrupt is None:
        _log.warning("LangGraph interrupt 不可用，回退为自动 execute（仅用于单测/无图环境）")
        return {"action": "execute", "modification": ""}
    try:
        return interrupt({"thread_id": thread_id, "payload_kind": payload.kind.value, "plan": plan})
    except RuntimeError as exc:
        # 非 LangGraph runnable context（如直接脚本调用 / 单测）→ mock 决策
        _log.warning("interrupt 在非 runnable context 下调用，回退 mock 决策：%s", exc)
        return {"action": "execute", "modification": ""}


def interrupt_for_exception(thread_id: str, signal: Any) -> Any:
    """supervisor 节点：子 Agent 抛异常时挂起并要求用户决策下一步。"""

    payload = HITLPayload(
        thread_id=thread_id,
        kind=HITLKind.EXCEPTION,
        exception_signal=signal,
        candidates=["retry", "skip", "modify", "cancel"],
        notes="子 Agent 执行时发生异常，请选择下一步。",
    )
    get_hitl_manager().emit(payload)
    if interrupt is None:
        return {"action": "cancel", "modification": ""}
    try:
        return interrupt({"thread_id": thread_id, "payload_kind": payload.kind.value, "signal": signal})
    except RuntimeError as exc:
        _log.warning("interrupt 在非 runnable context 下调用，回退 mock 决策：%s", exc)
        return {"action": "cancel", "modification": ""}


def make_resume_command(decision: dict[str, Any]) -> Any:
    """根据 UI 收到的用户决策构造 `Command(resume=...)`，推回 supervisor 子图。"""

    if Command is None:
        _log.warning("LangGraph Command 不可用，返回 plain dict 决策（仅用于单测/无图环境）")
        return decision
    return Command(resume=decision)


def decide_register_new_task(task_summary: str) -> tuple[str, str]:
    """供 supervisor 节点快速生成新 thread_id 与 initial state hook。"""

    tid = new_thread_id("task")
    _log.info("登记新任务 thread_id=%s summary=%s", tid, task_summary)
    return tid, task_summary


def assert_not_cancelled(decision: Any) -> None:
    """便利：从 interrupt 返回值取出 action；若为 cancel 则抛 HITLAbortedError 让上层短路。"""

    if isinstance(decision, dict) and decision.get("action") == "cancel":
        raise HITLAbortedError("用户取消了任务", cause=None)