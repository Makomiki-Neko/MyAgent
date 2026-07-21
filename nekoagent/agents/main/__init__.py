"""主 Agent：用户对话人设、/task 路由 + 提炼、RAG 注入与回传润色。"""

from nekoagent.agents.main.agent import MainAgent, get_main_agent, register_task_submitter

__all__ = [
    "MainAgent",
    "get_main_agent",
    "register_task_submitter",
]