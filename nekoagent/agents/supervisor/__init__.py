"""主管 Agent：任务拆解 → HITL → 子 Agent 调度 → 汇总回传主 Agent。"""

from nekoagent.agents.supervisor.agent import (
    SupervisorAgent,
    get_supervisor_agent,
    run_supervisor_for_task,
)

__all__ = [
    "SupervisorAgent",
    "get_supervisor_agent",
    "run_supervisor_for_task",
]