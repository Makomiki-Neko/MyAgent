"""任务主管 Agent（含 Reflexion 循环）。

架构：
    START → router
      ├─(planning)──→ plan ──→ END (awaiting_user, 等待审批)
      ├─(running)───→ scheduler → review ──→ scheduler (pass/retry)
      │                                        └─→ END (awaiting_info, 请求用户信息)
      └─(cancelled)──→ END

- scheduler 节点执行当前子任务 → review 节点审核结果
- review 判定：pass(下一子任务) / retry(重试) / need_info(请求用户补充)
- 每子任务最多请求一次 info；再次不足则强制执行
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from nekoagent.agents.shared.human_in_the_loop import (
    HITLKind,
    interrupt_for_exception,
    new_thread_id,
)
from nekoagent.agents.shared.output_constraints import invoke_with_retry
from nekoagent.agents.shared.schemas import (
    HitLExceptionSignal,
    SubTask,
    TaskPlan,
)
from nekoagent.config.loader import Config, get_config
from nekoagent.llm.factory import get_llm
from nekoagent.memory.stores import get_proc_memory_store
from nekoagent.observability.exceptions import HITLAbortedError, NekoAgentError
from nekoagent.observability.logging import get_logger

_log = get_logger("agent.supervisor")


def _save_hitl(session_id: str, thread_id: str, kind: str, payload: dict) -> None:
    try:
        from nekoagent.memory.mysql_checkpointer import save_hitl_pending
        save_hitl_pending(session_id, thread_id, kind, payload)
    except Exception:
        pass


def _clear_hitl(thread_id: str) -> None:
    try:
        from nekoagent.memory.mysql_checkpointer import clear_hitl_pending
        clear_hitl_pending(thread_id)
    except Exception:
        pass


def _log_supervisor_event(thread_id: str, session_id: str, event: str, detail: str = "") -> None:
    try:
        from datetime import datetime
        from nekoagent.memory.mysql_checkpointer import get_engine as _ge
        from nekoagent.memory.schema import get_metadata as _gm
        eng = _ge()
        tbl = _gm().tables["supervisor_logs"]
        with eng.begin() as c:
            c.execute(tbl.insert().values(
                thread_id=thread_id, session_id=session_id,
                event=event, detail=detail[:500],
                created_at=datetime.utcnow(),
            ))
    except Exception:
        pass


class SupervisorState(TypedDict, total=False):
    task_summary: str
    thread_id: str
    session_id: str
    plan: dict[str, Any] | TaskPlan | None
    subtask_results: dict[str, Any]
    current_subtask_index: int
    exceptions: list[dict[str, Any]]
    status: str
    final_result: str
    info_request: str
    info_response: str
    info_requested_for: str
    retry_count: int


class SupervisorAgent:

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or get_config()
        self._on_plan_ready: Callable[[str, TaskPlan], None] | None = None
        self._on_exception: Callable[[str, HitLExceptionSignal], None] | None = None
        self._on_complete: Callable[[str, str], None] | None = None
        self._on_subtask_change: Callable[[str, str], None] | None = None
        self._on_info_request: Callable[[str, str], None] | None = None
        self._executor_dispatch: Callable[[str, str, SubTask], str] | None = None
        self._compiled: Any | None = None
        self._saved_states: dict[str, dict] = {}

    def set_executor_dispatch(self, dispatch: Callable[[str, str, SubTask], str]) -> None:
        self._executor_dispatch = dispatch

    def set_on_plan_ready(self, cb: Callable[[str, TaskPlan], None]) -> None:
        self._on_plan_ready = cb

    def set_on_exception(self, cb: Callable[[str, HitLExceptionSignal], None]) -> None:
        self._on_exception = cb

    def set_on_complete(self, cb: Callable[[str, str], None]) -> None:
        self._on_complete = cb

    def set_on_subtask_change(self, cb: Callable[[str, str], None]) -> None:
        self._on_subtask_change = cb

    def set_on_info_request(self, cb: Callable[[str, str], None]) -> None:
        self._on_info_request = cb

    @staticmethod
    def _router(state: SupervisorState) -> str:
        status = state.get("status", "planning")
        if status in ("running", "done_pending_aggregate", "awaiting_info"):
            return "scheduler"
        if status in ("cancelled", "done"):
            return END
        return "plan"

    # ---------------- plan 节点 ----------------
    def _plan_node(self, state: SupervisorState) -> SupervisorState:
        task_summary = state["task_summary"]
        thread_id = state["thread_id"]
        _log.info("[supervisor] plan 开始 thread_id=%s summary=%r", thread_id, task_summary[:80])

        try:
            proc_store = get_proc_memory_store()
            similar = proc_store.find_similar(task_summary, limit=1)
        except Exception as exc:
            _log.warning("程序性记忆读取失败（继续冷规划）：%s", exc)
            similar = []

        referenced = bool(similar)
        persona = self.cfg.get_persona_cached("supervisor_agent").system_prompt
        plan_hint = ""
        if referenced:
            prev = similar[0]
            plan_hint = (
                f"\n\n历史同类任务参考（请参考其拆解并酌情调整）：\n"
                f"  task_summary: {prev.get('task_summary', '')}\n"
                f"  teardown: {prev.get('teardown', '')}\n"
                f"  exception_notes: {prev.get('exception_notes', '')}\n"
            )

        try:
            from nekoagent.skills import get_skill_registry as _gsk
            _avail = list(_gsk(self.cfg).all_skills().keys())
        except Exception:
            _avail = []
        skill_hint = f"\n可用 Skill：{_avail}（仅从列表中选择，勿使用未列出的 skill）" if _avail else "\n注意：当前没有可用 Skill，子任务请勿设置 required_skills。"

        prompt = (
            f"{persona}\n\n核心任务：{task_summary}\n"
            f"请拆解为可原子执行的子任务清单（按依赖顺序），输出符合 TaskPlan 结构的 JSON 对象。"
            f"{skill_hint}{plan_hint}"
        )
        llm = get_llm("supervisor_agent", self.cfg)
        plan = invoke_with_retry(
            llm, TaskPlan,
            [SystemMessage(content=prompt), HumanMessage(content="请输出 JSON。")],
            cfg=self.cfg,
            on_giveup_message="无法生成可执行任务计划，请稍后重试。",
        )
        state["plan"] = plan
        state["status"] = "awaiting_user"
        try:
            from nekoagent.observability.token_counter import estimate_tokens, record_task_tokens
            tok = estimate_tokens(prompt) + estimate_tokens(str(plan.model_dump() if hasattr(plan, 'model_dump') else plan))
            record_task_tokens(state.get("session_id", ""), tok, agent_role="supervisor_agent")
        except Exception:
            pass
        _log_supervisor_event(thread_id, state.get("session_id", ""), "plan_ready", task_summary[:200])
        _save_hitl(state.get("session_id", ""), thread_id, "plan_ready", {
            "task_summary": task_summary, "session_id": state.get("session_id", ""),
            "plan": str(plan.model_dump() if hasattr(plan, 'model_dump') else plan),
        })
        self._saved_states[thread_id] = {
            "task_summary": task_summary,
            "session_id": state["session_id"],
            "plan": plan,
            "stage": "plan",
        }
        if self._on_plan_ready is not None:
            try:
                self._on_plan_ready(thread_id, plan)
            except Exception as exc:
                _log.warning("on_plan_ready 回调抛错（被吞）：%s", exc)
        return state

    # ---------------- scheduler 节点（只执行，不推进索引） ----------------
    def _scheduler_node(self, state: SupervisorState) -> SupervisorState:
        plan = state["plan"]
        subtasks: list[SubTask] = plan.subtasks if isinstance(plan, TaskPlan) else [
            SubTask(**s) for s in plan.get("subtasks", [])
        ]
        idx = state.get("current_subtask_index", 0)
        if idx >= len(subtasks):
            state["status"] = "done_pending_aggregate"
            return state

        current = subtasks[idx]

        # 如果是收到用户 info 后重新进入，把 info_response 拼入 task_summary
        if state.get("status") == "awaiting_info" and state.get("info_response"):
            _log.info("[supervisor] 收到用户补充信息 thread_id=%s", state["thread_id"])
            state["task_summary"] = state["task_summary"] + f"\n用户补充信息：{state['info_response']}"
            state["status"] = "running"
            state["info_response"] = ""

        deps_ok = all(dep in state.get("subtask_results", {}) for dep in (current.depends_on or []))
        if not deps_ok:
            sig = HitLExceptionSignal(
                failed_subtask=current.name,
                error_summary=f"依赖未齐：{current.depends_on}，可能某个前置子 Agent 失败。",
            )
            if self._on_exception is not None:
                self._on_exception(state["thread_id"], sig)
            decision = interrupt_for_exception(state["thread_id"], sig)
            action = decision.get("action") if isinstance(decision, dict) else "cancel"
            if action == "retry":
                return state
            if action == "skip":
                state["current_subtask_index"] = idx + 1
                return state
            if action == "modify":
                mod_text = decision.get("modification", "")
                state["task_summary"] = state["task_summary"] + f"\n用户修改意见：{mod_text}"
                state["plan"] = None
                state["status"] = "replanning"
                return state
            state["status"] = "cancelled"
            raise HITLAbortedError("用户因依赖问题取消了任务")

        if self._executor_dispatch is None:
            raise NekoAgentError("主管 Agent 未注入 executor dispatch", cause=None)
        _log_supervisor_event(state["thread_id"], state.get("session_id", ""), "subtask_start", current.name)
        if self._on_subtask_change is not None:
            try:
                self._on_subtask_change(state["thread_id"], current.name)
            except Exception:
                pass
        # 构建上下文：任务总目标 + 前置步骤结果 + 用户提供的补充信息
        ctx_parts = [f"任务总目标：{state.get('task_summary', '')}"]
        prev_results = state.get("subtask_results", {})
        if prev_results:
            ctx_parts.append("已完成步骤结果：")
            for pname, presult in prev_results.items():
                ctx_parts.append(f"  · {pname}：{presult[:300]}")
        if state.get("info_response"):
            ctx_parts.append(f"用户补充信息：{state['info_response'][:300]}")
        current.context = "\n".join(ctx_parts)
        try:
            result = self._executor_dispatch(state["thread_id"], state.get("session_id", ""), current)
            state.setdefault("subtask_results", {})[current.name] = result
            _log.info("[supervisor] 子任务完成 thread_id=%s name=%s", state["thread_id"], current.name)
        except HITLAbortedError:
            raise
        except NekoAgentError as exc:
            sig = HitLExceptionSignal(
                failed_subtask=current.name,
                error_summary=exc.friendly_message,
            )
            if self._on_exception is not None:
                self._on_exception(state["thread_id"], sig)
            decision = interrupt_for_exception(state["thread_id"], sig)
            action = decision.get("action") if isinstance(decision, dict) else "cancel"
            if action == "retry":
                return state
            if action == "skip":
                state["current_subtask_index"] = idx + 1
                state.setdefault("exceptions", []).append({"subtask": current.name, "error": exc.friendly_message})
                return state
            if action == "modify":
                mod_text = decision.get("modification", "")
                state["task_summary"] = state["task_summary"] + f"\n用户修改意见：{mod_text}"
                state["plan"] = None
                state["status"] = "replanning"
                return state
            raise HITLAbortedError("用户取消任务（异常链路）")
        return state

    # ---------------- review 节点（Reflexion） ----------------
    def _review_node(self, state: SupervisorState) -> SupervisorState:
        plan = state["plan"]
        subtasks: list[SubTask] = plan.subtasks if isinstance(plan, TaskPlan) else [
            SubTask(**s) for s in plan.get("subtasks", [])
        ]
        idx = state.get("current_subtask_index", 0)
        if idx >= len(subtasks):
            state["status"] = "done_pending_aggregate"
            return state

        current = subtasks[idx]
        result = state.get("subtask_results", {}).get(current.name, "")
        _log.info("[supervisor] review 开始 thread_id=%s subtask=%s", state["thread_id"], current.name)

        prompt = (
            f"子任务目标：{current.goal}\n"
            f"子任务结果：{result[:1500]}\n\n"
            f"请判断该结果是否符合要求。仅输出以下三者之一：\n"
            f"- PASS：结果充分满足目标\n"
            f"- RETRY：结果不符合要求，需要重试\n"
            f"- NEED_INFO：结果明确指出缺少关键信息（如需要用户提供具体数据），需要向用户询问\n"
            f"输出仅一行，不要其他内容。"
        )
        try:
            llm = get_llm("supervisor_agent", self.cfg)
            resp = llm.invoke([SystemMessage(content=prompt)])
            verdict = (resp.content if isinstance(resp, object) else str(resp)).strip().upper()
        except Exception:
            verdict = "PASS"

        already_requested = state.get("info_requested_for") == current.name

        if verdict.startswith("NEED_INFO") and not already_requested:
            info_question = f"执行子任务「{current.name}」时缺少信息：{result[:300]}"
            state["info_request"] = info_question
            state["info_requested_for"] = current.name
            state["status"] = "awaiting_info"
            self._saved_states[state["thread_id"]] = {
                "task_summary": state["task_summary"],
                "session_id": state["session_id"],
                "plan": plan,
                "current_subtask_index": idx,
                "subtask_results": dict(state.get("subtask_results", {})),
                "exceptions": list(state.get("exceptions", [])),
                "info_requested_for": current.name,
                "stage": "info",
            }
            _log_supervisor_event(state["thread_id"], state.get("session_id", ""), "info_request", info_question[:200])
            _save_hitl(state.get("session_id", ""), state["thread_id"], "info_request", {
                "question": info_question, "subtask": current.name,
            })
            _log.info("[supervisor] 请求用户补充信息 thread_id=%s subtask=%s", state["thread_id"], current.name)
            if self._on_info_request is not None:
                try:
                    self._on_info_request(state["thread_id"], info_question)
                except Exception as exc:
                    _log.warning("on_info_request 抛错（被吞）：%s", exc)
            return state

        if verdict.startswith("RETRY") or (verdict.startswith("NEED_INFO") and already_requested):
            retry_count = state.get("retry_count", 0) + 1
            state["retry_count"] = retry_count
            if retry_count <= 2:
                _log_supervisor_event(state["thread_id"], state.get("session_id", ""), "subtask_review", f"RETRY {current.name} count={retry_count}")
                _log.info("[supervisor] 重试子任务 thread_id=%s subtask=%s retry=%d", state["thread_id"], current.name, retry_count)
                state["status"] = "running"
                return state
            else:
                _log.info("[supervisor] 重试已达上限，强制推进 thread_id=%s", state["thread_id"])

        # PASS or force-proceed
        _log_supervisor_event(state["thread_id"], state.get("session_id", ""), "subtask_review", f"PASS {current.name}")
        state["current_subtask_index"] = idx + 1
        state["retry_count"] = 0
        state["info_requested_for"] = ""
        state["status"] = "running"
        return state

    # ---------------- aggregate 节点 ----------------
    def _aggregate_node(self, state: SupervisorState) -> SupervisorState:
        plan = state["plan"]
        subtasks = plan.subtasks if isinstance(plan, TaskPlan) else [SubTask(**s) for s in plan.get("subtasks", [])]
        completed = state.get("subtask_results", {})
        exceptions = state.get("exceptions", [])
        lines = [f"任务 [{state['task_summary'][:80]}] 已完成："]
        for t in subtasks:
            r = completed.get(t.name)
            if r:
                lines.append(f"  · {t.name} → {r[:200]}")
            else:
                skipped = any(e.get("subtask") == t.name for e in exceptions)
                lines.append(f"  · {t.name} → 已跳过" if skipped else f"  · {t.name} → 未执行")
        if exceptions:
            lines.append("存在被跳过的子任务，详情见日志。")
        result_text = "\n".join(lines)
        if not exceptions:
            try:
                teardown = [{"name": t.name, "goal": t.goal, "depends_on": t.depends_on} for t in subtasks]
                get_proc_memory_store().record_task(task_summary=state["task_summary"], teardown=teardown, exception_notes="")
                _log.info("[supervisor] 已写入程序性记忆 thread_id=%s", state["thread_id"])
            except Exception as exc:
                _log.warning("程序性记忆写入失败（不阻塞汇总）：%s", exc)
        state["final_result"] = result_text
        state["status"] = "done"
        if self._on_complete:
            try:
                self._on_complete(state["thread_id"], result_text)
            except Exception as exc:
                _log.warning("on_complete 抛错（被吞）：%s", exc)
        return state

    # ---------------- 子图编译 ----------------
    def _build_graph(self) -> Any:
        if self._compiled is not None:
            return self._compiled
        g = StateGraph(SupervisorState)
        g.add_node("plan", self._plan_node)
        g.add_node("scheduler", self._scheduler_node)
        g.add_node("review", self._review_node)
        g.add_node("aggregate", self._aggregate_node)

        g.add_conditional_edges(START, self._router, {"plan": "plan", "scheduler": "scheduler", END: END})
        g.add_conditional_edges(
            "plan",
            lambda s: "scheduler" if s.get("status") == "running" else
                      ("plan" if s.get("status") == "replanning" else END),
            {"scheduler": "scheduler", "plan": "plan", END: END},
        )
        g.add_conditional_edges(
            "scheduler",
            lambda s: "review" if s.get("status") == "running" else
                      ("aggregate" if s.get("status") == "done_pending_aggregate" else "scheduler"),
            {"review": "review", "aggregate": "aggregate", "scheduler": "scheduler"},
        )
        g.add_conditional_edges(
            "review",
            lambda s: "scheduler" if s.get("status") == "running" else END,
            {"scheduler": "scheduler", END: END},
        )
        g.add_edge("aggregate", END)
        self._compiled = g.compile()
        return self._compiled

    # ---------------- 同步入口 ----------------
    def run_for_task(self, task_summary: str, session_id: str, thread_id: str | None = None) -> str:
        if thread_id is None:
            thread_id = new_thread_id("task")
        runnable = self._build_graph()
        initial: SupervisorState = {
            "task_summary": task_summary,
            "thread_id": thread_id,
            "session_id": session_id,
            "plan": None,
            "subtask_results": {},
            "current_subtask_index": 0,
            "exceptions": [],
            "status": "planning",
            "final_result": "",
        }
        try:
            runnable.invoke(initial, config={"configurable": {"thread_id": thread_id}})
        except HITLAbortedError as exc:
            _log_supervisor_event(thread_id, session_id, "cancel", exc.friendly_message[:200])
            _clear_hitl(thread_id)
            _log.info("[supervisor] 任务被取消 thread_id=%s：%s", thread_id, exc.friendly_message)
            return thread_id
        except NekoAgentError:
            raise
        _clear_hitl(thread_id)
        _log_supervisor_event(thread_id, session_id, "complete", task_summary[:200])
        return thread_id

    def resume_with_decision(self, thread_id: str, decision: dict[str, Any]) -> None:
        saved = self._saved_states.get(thread_id)
        if not saved:
            raise NekoAgentError(f"未找到 thread_id={thread_id} 的已保存规划状态，可能已过期。")
        stage = saved.get("stage", "plan")
        action = decision.get("action", "execute")

        if action == "cancel":
            _log.info("[supervisor] 任务已取消 thread_id=%s", thread_id)
            _clear_hitl(thread_id)
            self._saved_states.pop(thread_id, None)
            return

        if stage == "plan":
            if action == "execute":
                initial: SupervisorState = {
                    "task_summary": saved["task_summary"],
                    "thread_id": thread_id,
                    "session_id": saved["session_id"],
                    "plan": saved["plan"],
                    "subtask_results": {},
                    "current_subtask_index": 0,
                    "exceptions": [],
                    "status": "running",
                    "final_result": "",
                }
            elif action == "modify":
                mod_text = decision.get("modification", "")
                new_summary = f"{saved['task_summary']}\n用户修改意见：{mod_text}".strip()
                initial = SupervisorState({
                    "task_summary": new_summary, "thread_id": thread_id,
                    "session_id": saved["session_id"], "plan": None,
                    "subtask_results": {}, "current_subtask_index": 0,
                    "exceptions": [], "status": "planning", "final_result": "",
                })
            else:
                raise NekoAgentError(f"未知决策 action={action}")
        elif stage == "info":
            info_response = decision.get("info", "")
            subtask_results = saved.get("subtask_results", {})
            initial = SupervisorState({
                "task_summary": saved["task_summary"], "thread_id": thread_id,
                "session_id": saved["session_id"], "plan": saved["plan"],
                "subtask_results": subtask_results,
                "current_subtask_index": saved.get("current_subtask_index", 0),
                "exceptions": saved.get("exceptions", []),
                "status": "awaiting_info",
                "info_response": info_response,
                "info_requested_for": saved.get("info_requested_for", ""),
                "final_result": "",
            })
        else:
            raise NekoAgentError(f"未知 stage={stage}")

        runnable = self._build_graph()
        try:
            runnable.invoke(initial, config={"configurable": {"thread_id": thread_id}})
        except HITLAbortedError as exc:
            _log.info("[supervisor] 任务被取消 thread_id=%s：%s", thread_id, exc.friendly_message)
        except NekoAgentError:
            raise
        finally:
            # 仅当状态未被图内节点重新保存时才弹出（e.g., info_request 后保留新状态）
            if self._saved_states.get(thread_id) is saved:
                self._saved_states.pop(thread_id, None)


_supervisor_singleton: SupervisorAgent | None = None


def get_supervisor_agent(cfg: Config | None = None) -> SupervisorAgent:
    global _supervisor_singleton
    if _supervisor_singleton is None:
        _supervisor_singleton = SupervisorAgent(cfg or get_config())
    return _supervisor_singleton


def run_supervisor_for_task(task_summary: str, session_id: str) -> str:
    return get_supervisor_agent().run_for_task(task_summary, session_id)
