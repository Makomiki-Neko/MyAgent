"""Agent 共享结构化 schema：任务计划、子任务、执行结果、HITL 决策。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SubTask(BaseModel):
    name: str = Field(..., description="子任务名")
    goal: str = Field(..., description="子任务目标（单步可执行描述）")
    depends_on: list[str] = Field(default_factory=list, description="依赖的子任务名列表")
    required_skills: list[str] = Field(default_factory=list, description="所需 skill 名")
    required_mcp_tools: list[str] = Field(default_factory=list, description="所需 MCP 工具名（如 weather.get_weather）")
    context: str = Field(default="", description="主管提供的额外上下文（如前置任务结果、用户补充信息）")


class TaskPlan(BaseModel):
    """任务主管 Agent 提交给主 Agent（再展示给用户）的任务计划结构。"""

    task_summary: str = Field(..., description="任务简述")
    subtasks: list[SubTask] = Field(..., min_length=1)
    referenced_proc_mem: bool = Field(default=False, description="是否复用了历史程序性记忆")
    notes: str = Field(default="", description="额外说明（可选）")


class ExecutorResult(BaseModel):
    """子执行 Agent 完成单步后的结构化输出。"""

    result: str
    tokens_used: int = 0
    notes: str = ""


class HITLDecision(BaseModel):
    """用户 HITL 决策：execute / modify / cancel + 修改/补全字段。"""

    action: str = Field(..., description="execute | modify | cancel")
    modification: str = Field(default="", description="若 action=modify，附修改提示")
    notes: str = Field(default="")


class HitLExceptionSignal(BaseModel):
    """子 Agent 异常时由 supervisor 收集并交给 main agent 的中断信号载荷。"""

    failed_subtask: str
    error_summary: str
    candidates: list[str] = Field(
        default_factory=lambda: ["retry", "skip", "modify", "cancel"],
        description="向用户给出的候选操作",
    )