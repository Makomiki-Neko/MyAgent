"""任务调度器。

集成所有上游：
- main_agent 的 `submit_task` 函数注入到 main agent
- 创建 supervisor agent 实例 + 绑定 executor dispatch + callbacks
- 主管产生的 plan/exception/complete 事件 pub/sub 推 main agent 然后润色
- 用户决策 → 通过 thread_id 路由回 supervisor resume

并发：MVP 用 asyncio.create_task + 内部任务字典；多任务上下文天然隔离（各 thread_id 单独 state）。
"""

from __future__ import annotations

import asyncio
import enum
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from sqlalchemy import insert, update, select

from nekoagent.agents.executor import dispatch_subtask
from nekoagent.agents.main.agent import get_main_agent
from nekoagent.agents.shared.human_in_the_loop import (
    HITLAbortedError,
    HITLKind,
    HITLPayload,
    get_hitl_manager,
)
from nekoagent.agents.shared.schemas import HitLExceptionSignal, TaskPlan
from nekoagent.agents.supervisor.agent import get_supervisor_agent
from nekoagent.memory.schema import get_metadata
from nekoagent.observability.exceptions import NekoAgentError
from nekoagent.observability.logging import get_logger

_log = get_logger("queue.scheduler")


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    PLANNING = "planning"
    AWAITING_USER = "awaiting_user"
    RUNNING = "running"
    DONE = "done"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class Task:
    task_id: str
    session_id: str
    thread_id: str
    task_summary: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    plan: TaskPlan | None = None
    exception_signal: HitLExceptionSignal | None = None
    final_result: str = ""
    current_subtask: str = ""
    info_request: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error_log: str = ""


class TaskScheduler:
    """单例任务调度器。

    内部：
    - `_tasks`: task_id -> Task
    - `_thread_index`: thread_id -> task_id
    - 通过 asyncio.Queue 队列化提交；worker 顺序消费但 supervisor 子图内自并发（结合 HITL 中断后挂起）
    - 用户决策经 `handle_user_decision(thread_id, decision)` 路由
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._thread_index: dict[str, str] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_status_change_callbacks: list[Callable[[Task], None]] = []
        self._on_new_plan_callbacks: list[Callable[[Task], None]] = []
        self._on_exception_callbacks: list[Callable[[Task], None]] = []
        self._on_complete_callbacks: list[Callable[[Task], None]] = []
        self._on_decision_made_callbacks: list[Callable[[Task], None]] = []
        self._on_subtask_change_callbacks: list[Callable[[Task], None]] = []
        self._on_info_request_callbacks: list[Callable[[Task], None]] = []

    # ---------------- 订阅 ----------------
    def on_status_change(self, cb: Callable[[Task], None]) -> None:
        self._on_status_change_callbacks.append(cb)

    def on_new_plan(self, cb: Callable[[Task], None]) -> None:
        self._on_new_plan_callbacks.append(cb)

    def on_exception(self, cb: Callable[[Task], None]) -> None:
        self._on_exception_callbacks.append(cb)

    def on_complete(self, cb: Callable[[Task], None]) -> None:
        self._on_complete_callbacks.append(cb)

    def on_decision_made(self, cb: Callable[[Task], None]) -> None:
        self._on_decision_made_callbacks.append(cb)

    def on_subtask_change(self, cb: Callable[[Task], None]) -> None:
        self._on_subtask_change_callbacks.append(cb)

    def on_info_request(self, cb: Callable[[Task], None]) -> None:
        self._on_info_request_callbacks.append(cb)

    # ---------------- 提交 ----------------
    def submit_task(self, session_id: str, distilled_task_summary: str) -> dict[str, Any]:
        """新任务进入调度；返回 {task_id, thread_id, status}。"""

        task_id = uuid.uuid4().hex[:16]
        thread_id = f"task-{uuid.uuid4().hex[:12]}"
        task = Task(
            task_id=task_id,
            session_id=session_id,
            thread_id=thread_id,
            task_summary=distilled_task_summary,
        )
        with self._lock:
            self._tasks[task_id] = task
            self._thread_index[thread_id] = task_id
        self._persist_task_meta(task)
        self._update_status(task, TaskStatus.PLANNING)
        self._spawn_supervisor(task)
        return {"task_id": task_id, "thread_id": thread_id, "status": task.status.value}

    # ---------------- 用户决策 ----------------
    def handle_user_decision(self, thread_id: str, decision: dict[str, Any]) -> Task:
        if thread_id not in self._thread_index:
            raise NekoAgentError(f"未找到 thread_id={thread_id}; 可能已被取消或完成。")
        task_id = self._thread_index[thread_id]
        task = self._tasks[task_id]
        try:
            get_supervisor_agent().resume_with_decision(thread_id, decision)
            # info 回应后图可能继续执行，不设置 DONE
            if decision.get("action") != "provide_info":
                task.status = TaskStatus.DONE
        except HITLAbortedError as exc:
            _log.info("任务已取消: thread_id=%s reason=%s", thread_id, exc.friendly_message)
            self._update_status(task, TaskStatus.CANCELLED)
        except NekoAgentError as exc:
            _log.warning("resume_with_decision 失败: %s", exc.friendly_message)
            self._update_status(task, TaskStatus.ERROR, err=exc.friendly_message)
        # 通知 UI 用户决策已处理
        for cb in self._on_decision_made_callbacks:
            try:
                cb(task)
            except Exception as exc:
                _log.warning("decision_made callback 抛错（被吞）：%s", exc)
        return task

    # ---------------- 状态查询 ----------------
    def query_task_status(self, task_id: str) -> dict[str, Any]:
        if task_id not in self._tasks:
            raise NekoAgentError(f"未找到 task_id={task_id}; 可能已过期或不存在。")
        t = self._tasks[task_id]
        return {
            "task_id": t.task_id,
            "thread_id": t.thread_id,
            "session_id": t.session_id,
            "summary": t.task_summary,
            "status": t.status.value,
            "progress": t.progress,
            "exception": t.exception_signal.failed_subtask if t.exception_signal else None,
            "final_result": t.final_result,
            "error_log": t.error_log,
            "created_at": t.created_at.isoformat(),
            "updated_at": t.updated_at.isoformat(),
        }

    def list_tasks_by_session(self, session_id: str) -> list[dict[str, Any]]:
        return [
            self.query_task_status(tid)
            for tid, t in self._tasks.items()
            if t.session_id == session_id
        ]

    # ---------------- 内部 spawn ----------------
    def _spawn_supervisor(self, task: Task) -> None:
        """安排 supervisor 实际跑这个任务（HITL 会暂停）。"""

        supervisor = get_supervisor_agent()
        # 注入 dispatch + 回调
        supervisor.set_executor_dispatch(dispatch_subtask)
        supervisor.set_on_plan_ready(
            lambda thread_id, plan, _task=task: self._on_plan_ready_handler(_task, plan)
        )
        supervisor.set_on_exception(
            lambda thread_id, signal, _task=task: self._on_exception_handler(_task, signal)
        )
        supervisor.set_on_complete(
            lambda thread_id, result, _task=task: self._on_complete_handler(_task, result)
        )
        supervisor.set_on_subtask_change(
            lambda thread_id, subtask_name, _task=task: self._on_subtask_change_handler(_task, subtask_name)
        )
        supervisor.set_on_info_request(
            lambda thread_id, question, _task=task: self._on_info_request_handler(_task, question)
        )

        def runner():
            try:
                supervisor.run_for_task(task.task_summary, task.session_id, thread_id=task.thread_id)
            except HITLAbortedError:
                self._update_status(task, TaskStatus.CANCELLED)
            except NekoAgentError as exc:
                self._update_status(task, TaskStatus.ERROR, err=exc.friendly_message)
            except Exception as exc:
                _log.exception("supervisor 异常：%s", exc)
                self._update_status(task, TaskStatus.ERROR, err=format_friendly_error(exc))

        threading.Thread(target=runner, daemon=True, name=f"task-{task.task_id}").start()

    # ---------------- 主管回传 → 通知 main agent + 触发 UI callbacks ----------------
    def _on_plan_ready_handler(self, task: Task, plan: TaskPlan) -> None:
        task.plan = plan
        plan_text = get_main_agent().present_plan_to_user(plan, task.session_id)
        task.plan_presented_to_user_text = plan_text  # type: ignore[attr-defined]
        self._update_status(task, TaskStatus.AWAITING_USER)
        for cb in self._on_new_plan_callbacks:
            try:
                cb(task)
            except Exception as exc:
                _log.warning("on_new_plan callback 抛错（被吞）：%s", exc)

    def _on_exception_handler(self, task: Task, signal: HitLExceptionSignal) -> None:
        task.exception_signal = signal
        exception_text = get_main_agent().present_exception_to_user(signal, task.session_id)
        task.exception_presented_to_user_text = exception_text  # type: ignore[attr-defined]
        self._update_status(task, TaskStatus.AWAITING_USER)
        for cb in self._on_exception_callbacks:
            try:
                cb(task)
            except Exception as exc:
                _log.warning("on_exception callback 抛错（被吞）：%s", exc)

    def _on_info_request_handler(self, task: Task, question: str) -> None:
        task.info_request = question
        self._update_status(task, TaskStatus.AWAITING_USER)
        for cb in self._on_info_request_callbacks:
            try:
                cb(task)
            except Exception as exc:
                _log.warning("info_request callback 抛错（被吞）：%s", exc)

    def _on_subtask_change_handler(self, task: Task, subtask_name: str) -> None:
        task.current_subtask = subtask_name
        task.status = TaskStatus.RUNNING
        for cb in self._on_subtask_change_callbacks:
            try:
                cb(task)
            except Exception as exc:
                _log.warning("subtask_change callback 抛错（被吞）：%s", exc)

    def _on_complete_handler(self, task: Task, result_text: str) -> None:
        task.final_result = result_text
        task.progress = 1.0
        result_presented = get_main_agent().present_result_to_user(result_text, task.session_id)
        task.final_result_presented = result_presented  # type: ignore[attr-defined]
        self._update_status(task, TaskStatus.DONE)
        for cb in self._on_complete_callbacks:
            try:
                cb(task)
            except Exception as exc:
                _log.warning("on_complete callback 抛错（被吞）：%s", exc)

    # ---------------- 持久化 + 状态变更 ----------------
    def _update_status(self, task: Task, new_status: TaskStatus, err: str = "") -> None:
        task.status = new_status
        task.updated_at = datetime.utcnow()
        if err:
            task.error_log = (task.error_log + "\n" + err).strip()
        self._persist_task_meta(task)
        for cb in self._on_status_change_callbacks:
            try:
                cb(task)
            except Exception as exc:
                _log.warning("on_status_change callback 抛错（被吞）：%s", exc)

    def _persist_task_meta(self, task: Task | dict[str, Any]) -> None:
        """写入 `task_meta` 表。失败仅记日志不阻塞调度。"""

        try:
            from nekoagent.memory.mysql_checkpointer import get_engine
            engine = get_engine()
            table = get_metadata().tables["task_meta"]
            with engine.begin() as conn:
                existing = conn.execute(
                    select(table).where(table.c.task_id == task.task_id)
                ).first()
                if existing is None:
                    conn.execute(insert(table).values(
                        task_id=task.task_id,
                        session_id=task.session_id,
                        thread_id=task.thread_id,
                        title=task.task_summary[:255],
                        status=task.status.value,
                        progress=task.progress,
                        plan_json=task.plan.model_dump_json() if task.plan else "",
                        error_log=task.error_log,
                        created_at=task.created_at,
                        updated_at=task.updated_at,
                    ))
                else:
                    conn.execute(
                        update(table).where(table.c.task_id == task.task_id).values(
                            status=task.status.value,
                            progress=task.progress,
                            plan_json=task.plan.model_dump_json() if task.plan else "",
                            error_log=task.error_log,
                            updated_at=task.updated_at,
                        )
                    )
        except Exception as exc:
            _log.debug("task_meta 持久化失败（已忽略）：%s", exc)

    # ---------------- 持久化恢复（重启后 list pending） ----------------
    def restore_pending_tasks(self) -> list[Task]:
        """启动时从 DB 拉取仍处于 awaiting_user / running 状态的任务，便于 UI 重新接入。"""

        try:
            from nekoagent.memory.mysql_checkpointer import get_engine
            engine = get_engine()
            table = get_metadata().tables["task_meta"]
            with engine.connect() as conn:
                rs = conn.execute(
                    select(table).where(table.c.status.in_(
                        ["awaiting_user", "running", "planning", "pending"]
                    ))
                )
                rows = [dict(r._mapping) for r in rs]
            out: list[Task] = []
            for row in rows:
                task = Task(
                    task_id=row["task_id"],
                    session_id=row["session_id"],
                    thread_id=row["thread_id"] or "",
                    task_summary=row["title"],
                    status=TaskStatus(row["status"]),
                    progress=float(row["progress"]),
                    final_result="",
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                with self._lock:
                    self._tasks[task.task_id] = task
                    self._thread_index[task.thread_id] = task.task_id
                out.append(task)
            return out
        except Exception as exc:
            _log.warning("restore_pending_tasks 失败：%s", exc)
            return []


# ---------------- 单例 + 便利 API ----------------

_scheduler_singleton: TaskScheduler | None = None


def get_scheduler() -> TaskScheduler:
    global _scheduler_singleton
    if _scheduler_singleton is None:
        _scheduler_singleton = TaskScheduler()
        # 绑定 main agent 的 submit_task 至 scheduler
        from nekoagent.agents.main.agent import register_task_submitter
        register_task_submitter(_scheduler_singleton.submit_task)
        # 让 HITL manager 额外订阅一个 adapter（用于 UI 与调度同步）
        from nekoagent.agents.shared.human_in_the_loop import get_hitl_manager
        get_hitl_manager().set_callback(_on_hitl_emit_adapter)
    return _scheduler_singleton


def _on_hitl_emit_adapter(payload: HITLPayload) -> None:
    """HITLManager 推出 payload 时这里额外收一份用于 UI 订阅。"""

    # 任务调度目前由 supervisor callback 直接给出 plan/exception，HITLManager 给 UI 用做同步镜像即可
    _log.debug("HITL payload kind=%s thread_id=%s", payload.kind, payload.thread_id)


def submit_task(session_id: str, distilled_task_summary: str) -> dict[str, Any]:
    return get_scheduler().submit_task(session_id, distilled_task_summary)


def handle_user_decision(thread_id: str, decision: dict[str, Any]) -> dict[str, Any]:
    task = get_scheduler().handle_user_decision(thread_id, decision)
    return {"task_id": task.task_id, "status": task.status.value}


def query_task_status(task_id: str) -> dict[str, Any]:
    return get_scheduler().query_task_status(task_id)


def list_user_tasks(session_id: str) -> list[dict[str, Any]]:
    return get_scheduler().list_tasks_by_session(session_id)


def format_friendly_error(exc: Exception) -> str:
    from nekoagent.observability.exceptions import format_friendly_error as _fff
    return _fff(exc)