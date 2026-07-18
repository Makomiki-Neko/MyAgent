from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    WAITING_CONFIRM = "waiting_confirm"
    CONFIRMED = "confirmed"
    MODIFYING = "modifying"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SubTask(BaseModel):
    id: str = Field(description="子任务唯一标识")
    description: str = Field(description="子任务描述")
    depends_on: list[str] = Field(default_factory=list, description="依赖的子任务ID列表")
    executor_type: str = Field(default="default", description="执行子Agent类型")
    status: SubTaskStatus = SubTaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None


class TaskPlan(BaseModel):
    task_id: str = Field(description="任务唯一标识")
    objective: str = Field(description="任务目标描述")
    subtasks: list[SubTask] = Field(default_factory=list, description="子任务列表")
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    current_subtask: Optional[str] = None
    error: Optional[str] = None


class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    summary: str
    details: Optional[Any] = None
    error: Optional[str] = None
