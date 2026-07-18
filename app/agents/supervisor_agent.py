import json
import logging
import uuid
from datetime import datetime

from app.agents.base_agent import AgentRuntime
from app.config import get_agent_config
from app.memory.procedural_memory import ProceduralMemory
from app.task.task_schema import TaskPlan, SubTask, TaskStatus, SubTaskStatus

logger = logging.getLogger(__name__)


class SupervisorAgent:
    def __init__(self, procedural_memory: ProceduralMemory):
        cfg = get_agent_config("supervisor_agent")
        self.runtime = AgentRuntime("supervisor_agent")
        self.runtime.init_mcp(cfg.get("mcp_servers", []))
        self.config = cfg
        self.memory = procedural_memory

    def plan_task(self, objective: str) -> TaskPlan:
        similar = self.memory.get_similar_pattern(objective)
        similar_context = ""
        if similar:
            similar_context = f"\n参考历史类似任务的拆解方案:\n{json.dumps(similar, ensure_ascii=False, indent=2)}"

        prompt = f"""{self.config['system_prompt']}
        任务目标: {objective}{similar_context}
        请将上述任务拆解为原子子任务清单，仅以 JSON 格式返回（不要包含任何其他文字、代码块标记）:
        {{"subtasks": [
        {{"id": "sub-1", "description": "子任务描述", "depends_on": [], "executor_type": "default"}}
        ]}}"""

        messages = [{"role": "user", "content": prompt}]
        response = self.runtime.chat(messages, system_prompt=self.config["system_prompt"])
        plan = self._parse_plan(response, objective)
        self.memory.save_pattern(objective, {
            "objective": objective,
            "subtasks": [s.model_dump() for s in plan.subtasks],
            "created_at": datetime.now().isoformat(),
        })
        return plan

    def _parse_plan(self, response: str, objective: str) -> TaskPlan:
        try:
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            if "{" not in json_str:
                brace_start = json_str.find("{")
                if brace_start >= 0:
                    json_str = json_str[brace_start:]
                    last_brace = json_str.rfind("}")
                    if last_brace >= 0:
                        json_str = json_str[:last_brace + 1]

            data = json.loads(json_str)
            subtasks = []
            for i, st in enumerate(data.get("subtasks", [])):
                subtasks.append(SubTask(
                    id=st.get("id", f"sub-{i+1}"),
                    description=st.get("description", ""),
                    depends_on=st.get("depends_on", []),
                    executor_type=st.get("executor_type", "default"),
                ))
            return TaskPlan(
                task_id=f"task-{uuid.uuid4().hex[:8]}",
                objective=objective,
                subtasks=subtasks,
                status=TaskStatus.WAITING_CONFIRM,
            )
        except Exception as e:
            logger.error(f"Failed to parse plan: {e}\nResponse: {response}")
            return TaskPlan(
                task_id=f"task-{uuid.uuid4().hex[:8]}",
                objective=objective,
                status=TaskStatus.FAILED,
                error=f"任务拆解失败: {str(e)}",
            )

    def modify_plan(self, task: TaskPlan, user_feedback: str) -> TaskPlan:
        prompt = f"""{self.config['system_prompt']}

        原始任务目标: {task.objective}
        原始子任务清单:
        {json.dumps([s.model_dump() for s in task.subtasks], ensure_ascii=False, indent=2)}

        修改意见: {user_feedback}

        请根据意见重新拆解任务，仅以 JSON 格式返回:
        {{"subtasks": [
        {{"id": "sub-1", "description": "子任务描述", "depends_on": [], "executor_type": "default"}}
        ]}}"""

        messages = [{"role": "user", "content": prompt}]
        response = self.runtime.chat(messages, system_prompt=self.config["system_prompt"])
        return self._parse_plan(response, task.objective)

    def summarize_result(self, task: TaskPlan) -> str:
        prompt = f"""任务目标: {task.objective}

        执行结果汇总:
        {json.dumps([s.model_dump() for s in task.subtasks], ensure_ascii=False, indent=2)}

        请生成一份简洁的任务完成报告，总结每个子任务的执行结果。不要添加额外格式。"""
        messages = [{"role": "user", "content": prompt}]
        return self.runtime.chat(messages, system_prompt=self.config["system_prompt"])
