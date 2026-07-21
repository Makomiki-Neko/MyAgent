"""主 Agent 用户记忆存储：仅访问 `main_agent_messages` 与 `session_meta` 间接相关；禁止读写 `supervisor_proc_mem`。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import desc, select, insert, update, text

from nekoagent.memory.schema import get_metadata
from nekoagent.memory.stores.base import MemoryStore
from nekoagent.observability.exceptions import MemoryError
from nekoagent.observability.logging import get_logger

_log = get_logger("memory.stores.main")


class MainUserMemoryStore(MemoryStore):
    """主 Agent 用户记忆（情景+语义合并）：仅可读写 `main_agent_messages`。

    任何尝试访问 `supervisor_proc_mem` 的调用 MUST 直接拒绝。
    """

    FORBIDDEN_TABLE = "supervisor_proc_mem"

    def __init__(self, engine: Any | None = None) -> None:
        super().__init__("main_agent_messages", engine=engine)

    # ---- 写 ----
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tokens: int = 0,
        archived: int = 0,
    ) -> int:
        if role not in ("user", "assistant", "system", "tool"):
            raise MemoryError(f"非法 role：{role}")
        table = get_metadata().tables[self.table_name]
        with self.engine.begin() as conn:
            result = conn.execute(
                insert(table).values(
                    session_id=session_id,
                    role=role,
                    content=content,
                    tokens=tokens,
                    archived=archived,
                    created_at=datetime.utcnow(),
                )
            )
            return int(result.inserted_primary_key[0])

    def archive_message(self, message_id: int) -> None:
        table = get_metadata().tables[self.table_name]
        with self.engine.begin() as conn:
            conn.execute(
                update(table).where(table.c.id == message_id).values(archived=1)
            )

    def replace_with_summary(self, archived_ids: list[int], summary: str, tokens: int, session_id: str, summary_of: int | None = None) -> int:
        """把指定 archived_ids 标记为已归档，并插入一行 summary 角色 message 作为压缩结果。"""

        table = get_metadata().tables[self.table_name]
        with self.engine.begin() as conn:
            if archived_ids:
                conn.execute(
                    update(table).where(table.c.id.in_(archived_ids)).values(archived=1)
                )
            result = conn.execute(
                insert(table).values(
                    session_id=session_id,
                    role="system",
                    content=summary,
                    tokens=tokens,
                    archived=0,
                    created_at=datetime.utcnow(),
                    summary_of=summary_of,
                )
            )
            return int(result.inserted_primary_key[0])

    # ---- 读 ----
    def fetch_active_messages(self, session_id: str, limit: int = 100) -> list[dict[str, Any]]:
        table = get_metadata().tables[self.table_name]
        with self.engine.connect() as conn:
            rs = conn.execute(
                select(table)
                .where(table.c.session_id == session_id)
                .where(table.c.archived == 0)
                .order_by(desc(table.c.id))
                .limit(limit)
            )
            rows = [dict(r._mapping) for r in rs]
        rows.reverse()
        return rows

    def fetch_all_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        table = get_metadata().tables[self.table_name]
        with self.engine.connect() as conn:
            rs = conn.execute(
                select(table).order_by(desc(table.c.id)).limit(limit)
            )
            return [dict(r._mapping) for r in rs]

    # ---- 实现 abstractmethod ----
    def add(self, **fields: Any) -> int:
        return self.add_message(
            session_id=fields["session_id"],
            role=fields["role"],
            content=fields["content"],
            tokens=fields.get("tokens", 0),
            archived=fields.get("archived", 0),
        )

    def query_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.fetch_all_recent(limit)

    # ---- 隔离访问校验 ----
    def assert_not_proc_mem(self) -> None:
        """主动校验：本 Store 永不访问 supervisor_proc_mem。"""

        if self.table_name == self.FORBIDDEN_TABLE:
            raise MemoryError("违反隔离约束：主 Agent Store 不得接触 supervisor_proc_mem")