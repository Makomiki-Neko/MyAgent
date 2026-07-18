import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EpisodicMemory:
    def __init__(self, file_path: str, max_history: int = 100):
        self.file_path = Path(file_path)
        self.max_history = max_history
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._episodes: list[dict] = []
        self._load()

    def _load(self):
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self._episodes = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load episodic memory: {e}")
                self._episodes = []

    def _save(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self._episodes[-self.max_history:], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save episodic memory: {e}")

    def add_episode(self, role: str, content: str, metadata: dict | None = None):
        self._episodes.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        })
        if len(self._episodes) > self.max_history:
            self._episodes = self._episodes[-self.max_history:]
        self._save()

    def get_recent(self, n: int = 10) -> list[dict]:
        return self._episodes[-n:]

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        results = []
        query_lower = query.lower()
        for ep in reversed(self._episodes):
            if query_lower in ep["content"].lower():
                results.append(ep)
                if len(results) >= top_k:
                    break
        return results

    def clear(self):
        self._episodes = []
        self._save()
