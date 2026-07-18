import json
import logging

from app.agents.base_agent import AgentRuntime
from app.config import get_agent_config, get_skills_config
from app.task.task_schema import SubTask, SubTaskStatus

logger = logging.getLogger(__name__)


class ExecutorAgent:
    def __init__(self):
        cfg = get_agent_config("executor_agent")
        self.runtime = AgentRuntime("executor_agent")
        self.runtime.init_mcp(cfg.get("mcp_servers", []))
        skills_cfg = get_skills_config()
        if skills_cfg.get("enabled", False) and skills_cfg.get("auto_load", False):
            self.runtime.init_skills(skills_cfg.get("path", "app/skills"))
        self.config = cfg

    def execute(self, subtask: SubTask, context: str = "") -> SubTask:
        tools_context = ""
        if self.runtime.mcp:
            mcp_tools = self.runtime.mcp.get_all_tools()
            if mcp_tools:
                tools_context = "\n可用MCP工具:\n" + json.dumps(
                    [{"name": t.get("name"), "description": t.get("description", "")} for t in mcp_tools],
                    ensure_ascii=False, indent=2
                )

        skills_context = ""
        if self.runtime.skills:
            skills_context = "\n可用本地技能:\n" + json.dumps(
                list(self.runtime.skills.keys()), ensure_ascii=False, indent=2
            )

        prompt = f"""{self.config['system_prompt']}

当前任务上下文: {context}

执行以下子任务:
任务ID: {subtask.id}
任务描述: {subtask.description}{tools_context}{skills_context}

请执行此子任务。如需使用MCP工具或本地技能，请直接调用。
输出执行结果（纯文字描述，不要JSON格式）。"""

        messages = [{"role": "user", "content": prompt}]
        logger.info(f"Executing subtask: {subtask.id} - {subtask.description[:60]}")

        try:
            result = self.runtime.chat(messages, system_prompt=self.config["system_prompt"])
            subtask.status = SubTaskStatus.COMPLETED
            subtask.result = result
            logger.info(f"Subtask {subtask.id} completed")
        except Exception as e:
            subtask.status = SubTaskStatus.FAILED
            subtask.error = str(e)
            logger.error(f"Subtask {subtask.id} failed: {e}")

        return subtask
