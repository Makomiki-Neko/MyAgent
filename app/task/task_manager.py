import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from app.agents.executor_agent import ExecutorAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.config import get_task_config
from app.task.task_schema import TaskPlan, TaskStatus, SubTaskStatus

logger = logging.getLogger(__name__)


class TaskManager:
    def __init__(self, supervisor: SupervisorAgent):
        self.supervisor = supervisor
        self.tasks: dict[str, TaskPlan] = {}
        self.config = get_task_config()
        self.executor = ExecutorAgent()
        self._pool = ThreadPoolExecutor(max_workers=self.config["max_concurrent"])
        self._lock = threading.Lock()
        self._on_plan_ready: Optional[Callable[[TaskPlan], None]] = None
        self._on_task_completed: Optional[Callable[[str, str], None]] = None

    def set_plan_ready_callback(self, callback: Callable[[TaskPlan], None]):
        self._on_plan_ready = callback

    def set_task_completed_callback(self, callback: Callable[[str, str], None]):
        self._on_task_completed = callback

    def submit_task(self, objective: str) -> str:
        task = self.supervisor.plan_task(objective)
        with self._lock:
            self.tasks[task.task_id] = task
        logger.info(f"Task {task.task_id} created, status={task.status.value}")

        if self._on_plan_ready and task.status == TaskStatus.WAITING_CONFIRM:
            self._on_plan_ready(task)

        return task.task_id

    def confirm_task(self, task_id: str) -> bool:
        with self._lock:
            task = self.tasks.get(task_id)
            if not task or task.status != TaskStatus.WAITING_CONFIRM:
                return False
            task.status = TaskStatus.CONFIRMED

        self._pool.submit(self._execute_task, task_id)
        return True

    def modify_task(self, task_id: str, user_feedback: str) -> bool:
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return False

        new_task = self.supervisor.modify_plan(task, user_feedback)
        with self._lock:
            self.tasks[task_id] = new_task

        if self._on_plan_ready and new_task.status == TaskStatus.WAITING_CONFIRM:
            self._on_plan_ready(new_task)

        return True

    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            task.status = TaskStatus.CANCELLED
            for st in task.subtasks:
                if st.status == SubTaskStatus.PENDING:
                    st.status = SubTaskStatus.FAILED
        logger.info(f"Task {task_id} cancelled")
        return True

    def get_task(self, task_id: str) -> Optional[TaskPlan]:
        with self._lock:
            return self.tasks.get(task_id)

    def get_all_tasks(self) -> list[TaskPlan]:
        with self._lock:
            return list(self.tasks.values())

    def _execute_task(self, task_id: str):
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return

        try:
            task.status = TaskStatus.RUNNING
            context = task.objective
            total = len(task.subtasks)

            for i, subtask in enumerate(task.subtasks):
                with self._lock:
                    if task.status == TaskStatus.CANCELLED:
                        return

                subtask.status = SubTaskStatus.RUNNING
                task.current_subtask = subtask.id
                result = self.executor.execute(subtask, context)
                subtask.status = result.status
                subtask.result = result.result
                subtask.error = result.error
                task.progress = ((i + 1) / total) * 100

            with self._lock:
                if task.status == TaskStatus.CANCELLED:
                    return
                if all(st.status == SubTaskStatus.COMPLETED for st in task.subtasks):
                    summary = self.supervisor.summarize_result(task)
                    task.status = TaskStatus.COMPLETED
                    logger.info(f"Task {task_id} completed")
                    if self._on_task_completed:
                        self._on_task_completed(task_id, summary)
                else:
                    task.status = TaskStatus.FAILED
                    errors = [st.error for st in task.subtasks if st.error]
                    task.error = "; ".join(errors)
                    logger.error(f"Task {task_id} failed: {task.error}")

        except Exception as e:
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error = str(e)
            logger.exception(f"Task {task_id} execution error: {e}")
