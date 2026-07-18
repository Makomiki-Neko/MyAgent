import logging

from langgraph.graph import StateGraph, END
from typing import TypedDict

from app.task.task_schema import TaskPlan, TaskStatus, SubTaskStatus

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    task: TaskPlan
    context: str
    step_index: int
    next_action: str


def _check_dependencies(depends_on: list[str], subtasks: list) -> bool:
    for dep_id in depends_on:
        for st in subtasks:
            if st.id == dep_id and st.status != SubTaskStatus.COMPLETED:
                return False
    return True


def determine_step(state: WorkflowState) -> dict:
    task = state["task"]
    step_idx = state["step_index"]

    if task.status in (TaskStatus.CANCELLED, TaskStatus.FAILED):
        return {**state, "next_action": "end"}
    if step_idx >= len(task.subtasks):
        return {**state, "next_action": "summarize"}

    subtask = task.subtasks[step_idx]
    deps_met = _check_dependencies(subtask.depends_on, task.subtasks)
    if deps_met:
        return {**state, "next_action": "execute"}
    else:
        return {**state, "next_action": "wait"}


def router(state: WorkflowState) -> str:
    action = state.get("next_action", "wait")
    return action if action != "end" else END


def execute_step(state: WorkflowState) -> dict:
    task = state["task"]
    step_idx = state["step_index"]
    if step_idx < len(task.subtasks):
        subtask = task.subtasks[step_idx]
        subtask.status = SubTaskStatus.RUNNING
        task.current_subtask = subtask.id
        task.progress = (step_idx / len(task.subtasks)) * 100
    return {**state, "task": task}


def mark_done(state: WorkflowState) -> dict:
    task = state["task"]
    step_idx = state["step_index"]
    if step_idx < len(task.subtasks):
        subtask = task.subtasks[step_idx]
        if subtask.status == SubTaskStatus.RUNNING:
            subtask.status = SubTaskStatus.COMPLETED
    return {**state, "task": task, "step_index": step_idx + 1}


def wait_step(state: WorkflowState) -> dict:
    logger.info(f"Dependencies not met for step {state['step_index']}, retrying")
    return state


def summarize_step(state: WorkflowState) -> dict:
    task = state["task"]
    task.status = TaskStatus.COMPLETED
    task.progress = 100.0
    task.current_subtask = None
    logger.info(f"Task {task.task_id} workflow completed")
    return {**state, "task": task}


def build_workflow() -> StateGraph:
    workflow = StateGraph(WorkflowState)

    workflow.add_node("determine", determine_step)
    workflow.add_node("execute", execute_step)
    workflow.add_node("mark_done", mark_done)
    workflow.add_node("wait", wait_step)
    workflow.add_node("summarize", summarize_step)

    workflow.set_entry_point("determine")

    workflow.add_conditional_edges("determine", router, {
        "execute": "execute",
        "wait": "wait",
        "summarize": "summarize",
    })

    workflow.add_edge("execute", "mark_done")
    workflow.add_edge("mark_done", "determine")
    workflow.add_edge("wait", "determine")
    workflow.add_edge("summarize", END)

    return workflow.compile()
