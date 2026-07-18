import logging
from pathlib import Path

from app.config import get_memory_config
from app.memory.episodic_memory import EpisodicMemory
from app.memory.semantic_memory import SemanticMemory
from app.memory.procedural_memory import ProceduralMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self):
        cfg = get_memory_config()
        base_path = Path(cfg["base_path"])
        base_path.mkdir(parents=True, exist_ok=True)

        self.episodic = EpisodicMemory(
            file_path=str(base_path / cfg["episodic"]["file_name"]),
            max_history=cfg["episodic"]["max_history"],
        )
        self.semantic = SemanticMemory(
            file_path=str(base_path / cfg["semantic"]["file_name"]),
        )
        self.procedural = ProceduralMemory(
            file_path=str(base_path / cfg["procedural"]["file_name"]),
        )
        self.supervisor_procedural = ProceduralMemory(
            file_path=str(base_path / cfg["supervisor_procedural"]["file_name"]),
        )

    def build_memory_context(self) -> str:
        recent = self.episodic.get_recent(10)
        user_info = self.semantic.get_user_info()
        habits = self.procedural.all_data

        lines = []
        if user_info:
            lines.append("【用户信息】")
            for k, v in user_info.items():
                if k != "preferences":
                    lines.append(f"  {k}: {v}")
            prefs = user_info.get("preferences", {})
            if prefs:
                lines.append("【用户偏好】")
                for k, v in prefs.items():
                    lines.append(f"  {k}: {v}")

        if recent:
            lines.append("\n【最近对话】")
            for ep in recent[-5:]:
                lines.append(f"  [{ep['role']}]: {ep['content'][:100]}")

        return "\n".join(lines)

    def record_interaction(self, role: str, content: str, metadata: dict | None = None):
        self.episodic.add_episode(role, content, metadata)
        logger.debug(f"Recorded {role} interaction: {content[:50]}...")
