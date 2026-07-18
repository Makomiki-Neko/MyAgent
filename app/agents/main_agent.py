import json
import logging
from datetime import datetime

from app.agents.base_agent import AgentRuntime
from app.config import get_agent_config
from app.memory.memory_manager import MemoryManager
from app.task.task_schema import TaskPlan, TaskStatus

logger = logging.getLogger(__name__)

# 废弃
_TASK_TRIGGER_KEYWORDS = [
    "计划", "规划", "安排", "帮我", "计算", "分析",
    "研究", "调查", "整理", "组织", "设计", "开发",
    "实现", "构建", "创建", "比较", "评估",
]


class MainAgent:
    def __init__(self, memory_manager: MemoryManager):
        cfg = get_agent_config("main_agent")
        self.runtime = AgentRuntime("main_agent")
        self.runtime.init_mcp(cfg.get("mcp_servers", []))
        self.runtime.init_skills()
        self.config = cfg
        self.memory = memory_manager
        self.conversation_history: list[dict] = []
        self._task_callback = None

    def set_task_callback(self, callback):
        self._task_callback = callback

    def _build_system_prompt(self) -> str:
        persona = self.config["persona_prompt"]
        memory_ctx = self.memory.build_memory_context()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.config["system_prompt_template"].format(
            persona_prompt=persona,
            memory_context=memory_ctx,
            current_time=now,
        )

    def _needs_task(self, user_input: str) -> bool:
        prompt = f"""你是一个任务提炼助手。用户输入可能包含闲聊、情感表达等不相干内容。
        请判断用户的请求中是否有需要规划执行的任务，如果需要规划执行请返回 true , 否则返回 false .

        用户: "{user_input}"

        本次回答强制规则，严格遵守：
        1. 只能输出单个单词：true 或 false（全小写英文）
        2. 禁止任何解释、标点、换行、markdown、多余描述
        3. 不允许输出中文、数字、JSON、段落
        """

        decide = False
        msgs = [{"role": "user", "content": prompt}]
        for _ in range(3): # 允许重试三次
            result = self.runtime.chat(msgs, system_prompt="你只输出最精简的决定，true 或 false 。")
            if result.strip() == "true":
                decide = True
                break
            elif result.strip() == "false":
                decide = False
                break
        
        return decide

    def _refine_task(self, user_input: str) -> str:
        prompt = f"""你是一个任务提炼助手。用户输入可能包含闲聊、情感表达等不相干内容。
        请从中提取出核心任务描述，去除所有闲聊、情感表达、人称修饰等不相干内容，返回最精简的核心任务。

        示例：
        用户: "请规划一次与Miya的双人旅行呢，想和Miya贴贴"
        输出: 请规划一次有私人空间场景的双人旅行

        用户: "帮我分析一下最近的项目数据，感觉最近好累哦"
        输出: 分析最近的项目数据

        用户: "{user_input}"
        输出:"""

        msgs = [{"role": "user", "content": prompt}]
        result = self.runtime.chat(msgs, system_prompt="你只输出最精简的任务描述，不要其他任何内容。")
        return result.strip()

    def chat(self, user_input: str) -> str:
        self.memory.record_interaction("user", user_input)

        if self._needs_task(user_input):
            core_task = self._refine_task(user_input)
            logger.info(f"Task detected. Original: '{user_input[:50]}...' Refined: '{core_task}'")
            if self._task_callback:
                task_id = self._task_callback(core_task, self.conversation_history)
                return self._build_task_submitted_response(core_task, task_id)
            return f"[已识别到复杂任务，正在处理...]\n核心任务: {core_task}"

        system_msg = {"role": "system", "content": self._build_system_prompt()}
        messages = [system_msg] + self.conversation_history[-20:] + [
            {"role": "user", "content": user_input}
        ]
        response = self.runtime.chat(messages)
        self.memory.record_interaction("assistant", response)
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def _build_task_submitted_response(self, core_task: str, task_id: str) -> str:
        prompt = f"""用户提交了一个任务，核心任务是: "{core_task}"
        请用你的人设风格告知用户任务已提交，正在等待计划生成。语气要温暖可爱。"""
        msgs = [
            {"role": "system", "content": self._build_system_prompt()},
            *self.conversation_history[-6:],
            {"role": "user", "content": prompt},
        ]
        response = self.runtime.chat(msgs)
        self.memory.record_interaction("assistant", response)
        return response

    def present_task_plan(self, task: TaskPlan) -> str:
        prompt = f"""主管Agent生成了以下任务计划，请用你的人设风格展示给用户，让用户决定是否执行。
        任务ID: {task.task_id}
        任务目标: {task.objective}
        子任务清单:
        {json.dumps([{"id": s.id, "description": s.description, "depends_on": s.depends_on} for s in task.subtasks], ensure_ascii=False, indent=2)}

        请用可爱的语气询问用户是否要执行这个计划，或者想要修改/取消。"""
        msgs = [
            {"role": "system", "content": self._build_system_prompt()},
            *self.conversation_history[-10:],
            {"role": "user", "content": prompt},
        ]
        response = self.runtime.chat(msgs)
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def notify_task_cancelled(self, task_id: str) -> str:
        prompt = f"任务 {task_id} 已被取消。请告知用户任务已取消。"
        msgs = [
            {"role": "system", "content": self._build_system_prompt()},
            *self.conversation_history[-6:],
            {"role": "user", "content": prompt},
        ]
        response = self.runtime.chat(msgs)
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def notify_task_completed(self, task_id: str, summary: str) -> str:
        prompt = f"""任务 {task_id} 已完成！
        主管Agent的汇总报告:
        {summary}

        请用你的人设风格将结果润色后告知用户，语气要温暖可爱，让用户感受到任务的完成。"""
        msgs = [
            {"role": "system", "content": self._build_system_prompt()},
            *self.conversation_history[-10:],
            {"role": "user", "content": prompt},
        ]
        response = self.runtime.chat(msgs)
        self.memory.record_interaction("assistant", response)
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def notify_task_status(self, task: TaskPlan) -> str:
        prompt = f"""用户请求查看任务状态，请用你的风格告知用户。
        任务ID: {task.task_id}
        目标: {task.objective}
        状态: {task.status.value}
        进度: {task.progress:.0f}%
        当前子任务: {task.current_subtask or '无'}
        子任务详情:
        {json.dumps([{"id": s.id, "description": s.description, "status": s.status.value} for s in task.subtasks], ensure_ascii=False, indent=2)}"""
        msgs = [
            {"role": "system", "content": self._build_system_prompt()},
            *self.conversation_history[-6:],
            {"role": "user", "content": prompt},
        ]
        response = self.runtime.chat(msgs)
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def reset_conversation(self):
        self.conversation_history = []
        logger.info("MainAgent conversation history reset")
