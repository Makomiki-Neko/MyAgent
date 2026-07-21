"""MemoryStore 抽象基类。所有 Store 子类限制访问特定表。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from sqlalchemy import Engine, select, insert, update, text
from sqlalchemy.engine import Engine as SqlEngine

from nekoagent.memory.mysql_checkpointer import get_engine


class MemoryStore(ABC):
    """Store 抽象基类：仅暴露读写接口；具体表绑定由子类决定。"""

    def __init__(self, table_name: str, engine: SqlEngine | None = None) -> None:
        self.table_name = table_name
        self._engine: SqlEngine | None = engine

    @property
    def engine(self) -> SqlEngine:
        if self._engine is None:
            self._engine = get_engine()
        return self._engine

    @abstractmethod
    def add(self, **fields: Any) -> int:
        """插入一行；返回主键 id。"""

    @abstractmethod
    def query_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """返回最近 limit 条记录。"""


def _now() -> datetime:
    return datetime.utcnow()