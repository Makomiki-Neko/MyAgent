"""执行子 Agent：单步原子执行 + 懒加载 skill + 调用 YAML 授权 MCP 工具。"""

from nekoagent.agents.executor.agent import (
    ExecutorAgent,
    get_executor_agent,
    dispatch_subtask,
)

__all__ = [
    "ExecutorAgent",
    "get_executor_agent",
    "dispatch_subtask",
]