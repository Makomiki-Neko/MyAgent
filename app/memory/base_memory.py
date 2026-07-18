import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BaseMemory(ABC):
    @abstractmethod
    def save(self, key: str, value: Any):
        ...

    @abstractmethod
    def load(self, key: str) -> Any:
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[Any]:
        ...

    @abstractmethod
    def clear(self):
        ...


class FileMemory(BaseMemory):
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load memory from {self.file_path}: {e}")
                self._data = {}

    def _save_to_disk(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory to {self.file_path}: {e}")

    def save(self, key: str, value: Any):
        self._data[key] = value
        self._save_to_disk()

    def load(self, key: str) -> Any:
        return self._data.get(key)

    def search(self, query: str, top_k: int = 5) -> list[Any]:
        results = []
        query_lower = query.lower()
        for key, value in self._data.items():
            if query_lower in key.lower():
                results.append({"key": key, "value": value})
            elif isinstance(value, str) and query_lower in value.lower():
                results.append({"key": key, "value": value})
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, str) and query_lower in v.lower():
                        results.append({"key": key, "value": value})
                        break
        return results[:top_k]

    def clear(self):
        self._data = {}
        self._save_to_disk()

    @property
    def all_data(self) -> dict:
        return self._data
