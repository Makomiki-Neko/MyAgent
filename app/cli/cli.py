import logging
import sys

from app.agents.main_agent import MainAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.memory.memory_manager import MemoryManager
from app.task.task_manager import TaskManager
from app.task.task_schema import TaskPlan, TaskStatus

logger = logging.getLogger(__name__)


class CLIApp:
    def __init__(self):
        self.memory = MemoryManager()
        self.main_agent = MainAgent(self.memory)
        self.supervisor = SupervisorAgent(self.memory.supervisor_procedural)
        self.task_manager = TaskManager(self.supervisor)
        self.task_manager.set_plan_ready_callback(self._on_plan_ready)
        self.task_manager.set_task_completed_callback(self._on_task_completed)
        self.main_agent.set_task_callback(self._submit_task_to_manager)
        self.running = True

    def _submit_task_to_manager(self, core_objective: str, _conversation_history=None) -> str:
        return self.task_manager.submit_task(core_objective)

    def _on_plan_ready(self, task: TaskPlan):
        response = self.main_agent.present_task_plan(task)
        print(f"\nMainAgent: {response}\n")
        print("[提示: 回复 '确认执行', '修改: ...', 或 '取消任务']")

    def _on_task_completed(self, task_id: str, summary: str):
        response = self.main_agent.notify_task_completed(task_id, summary)
        print(f"\nMainAgent: {response}\n")

    def _handle_task_decision(self, user_input: str):
        pending = [t for t in self.task_manager.get_all_tasks()
                   if t.status == TaskStatus.WAITING_CONFIRM]
        if not pending:
            return False
        task = pending[0]
        text = user_input.strip()

        if text.startswith("确认") or text.startswith("执行"):
            ok = self.task_manager.confirm_task(task.task_id)
            if ok:
                response = self.main_agent._build_task_submitted_response(task.objective, task.task_id)
                print(f"\nMainAgent: {response}\n")
            return True

        if text.startswith("修改"):
            feedback = text[2:].strip()
            if not feedback:
                feedback = input("请告诉我你想怎么修改: ").strip()
            if feedback:
                ok = self.task_manager.modify_task(task.task_id, feedback)
                if ok:
                    resp = "已重新规划任务，等待新的计划..."
                    print(f"\nMainAgent: {resp}\n")
            return True

        if text.startswith("取消"):
            ok = self.task_manager.cancel_task(task.task_id)
            if ok:
                resp = self.main_agent.notify_task_cancelled(task.task_id)
                print(f"\nMainAgent: {resp}\n")
            return True

        return False

    def run(self):
        print("\n" + "=" * 60)
        print("  多Agent用户交互系统 [MVP]")
        print("  输入 /help 查看帮助, /quit 退出")
        print("=" * 60 + "\n")

        while self.running:
            try:
                user_input = input("你: ").strip()
                if not user_input:
                    continue

                if user_input == "/quit":
                    self.running = False
                    print("再见！")
                    break
                elif user_input == "/help":
                    self._show_help()
                elif user_input == "/reset":
                    self.main_agent.reset_conversation()
                    print("对话已重置")
                elif user_input == "/tasks":
                    self._show_tasks()
                elif user_input.startswith("/task "):
                    task_id = user_input[6:].strip()
                    task = self.task_manager.get_task(task_id)
                    if task:
                        resp = self.main_agent.notify_task_status(task)
                        print(f"\nMainAgent: {resp}\n")
                    else:
                        print(f"任务 {task_id} 不存在")
                else:
                    if self._handle_task_decision(user_input):
                        continue

                    print("MainAgent: ", end="", flush=True)
                    response = self.main_agent.chat(user_input)
                    print(response)
                    print()

            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                logger.exception(f"CLI error: {e}")
                print(f"\n[系统异常] 请稍后重试")

    def _show_help(self):
        print("""
        可用命令:
        /help          显示帮助
        /tasks         查看所有任务状态
        /task <id>     查看任务详情
        /reset         重置对话历史
        /quit          退出系统

        任务决策:
        确认执行       执行任务计划
        修改: ...      修改任务计划
        取消任务       取消任务

        直接输入文字即可与 AI 助手对话。
        系统会自动识别复杂任务并转交主管Agent处理。
        """)

    def _show_tasks(self):
        tasks = self.task_manager.get_all_tasks()
        if not tasks:
            print("当前没有任务")
            return
        print(f"\n共 {len(tasks)} 个任务:")
        for t in tasks:
            print(f"  [{t.status.value.upper()}] {t.task_id}: {t.objective[:60]} ({t.progress:.0f}%)")
