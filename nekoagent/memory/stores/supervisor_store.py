"""主管 Agent 程序性记忆存储：仅访问 `supervisor_proc_mem`；禁止读写 `main_agent_messages`。"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from sqlalchemy import desc, select, insert, update

from nekoagent.memory.schema import get_metadata
from nekoagent.memory.stores.base import MemoryStore
from nekoagent.observability.exceptions import MemoryError

import json


class SupervisorProcMemoryStore(MemoryStore):
    """主管 Agent 程序性记忆：同类任务拆解/调度方案/异常处理。

    仅可读写 `supervisor_proc_mem`。任何尝试访问 `main_agent_messages` MUST 直接拒绝。
    """

    FORBIDDEN_TABLE = "main_agent_messages"

    def __init__(self, engine: Any | None = None) -> None:
        super().__init__("supervisor_proc_mem", engine=engine)

    @staticmethod
    def task_signature(task_summary: str) -> str:
        """对任务意图文本生成稳定的短签名；同类任务签名应接近。

        MVP 采用：对前置分词/长度截断后取 SHA1。子图成熟后可替换为嵌入向量聚类。
        """

        normalized = " ".join(task_summary.lower().split())[:256]
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]

    def record_task(
        self,
        task_summary: str,
        teardown: list[dict[str, Any]] | str,
        exception_notes: str = "",
    ) -> int:
        teardown_json = teardown if isinstance(teardown, str) else json.dumps(teardown, ensure_ascii=False)
        table = get_metadata().tables[self.table_name]
        sig = self.task_signature(task_summary)
        with self.engine.begin() as conn:
            result = conn.execute(
                insert(table).values(
                    task_signature=sig,
                    task_summary=task_summary,
                    teardown=teardown_json,
                    exception_notes=exception_notes,
                    created_at=datetime.utcnow(),
                    last_referenced_at=datetime.utcnow(),
                )
            )
            return int(result.inserted_primary_key[0])

    def touch(self, record_id: int) -> None:
        table = get_metadata().tables[self.table_name]
        with self.engine.begin() as conn:
            conn.execute(
                update(table).where(table.c.id == record_id).values(last_referenced_at=datetime.utcnow())
            )

    def find_similar(self, task_summary: str, limit: int = 5) -> list[dict[str, Any]]:
        """MVP：按签名精确匹配。如签名无匹配返回空列表。"""

        sig = self.task_signature(task_summary)
        table = get_metadata().tables[self.table_name]
        with self.engine.connect() as conn:
            rs = conn.execute(
                select(table)
                .where(table.c.task_signature == sig)
                .order_by(desc(table.c.last_referenced_at))
                .limit(limit)
            )
            rows = [dict(r._mapping) for r in rs]
        return rows

    # ---- 实现 abstractmethod ----
    def add(self, **fields: Any) -> int:
        return self.record_task(
            task_summary=fields["task_summary"],
            teardown=fields["teardown"],
            exception_notes=fields.get("exception_notes", ""),
        )

    def query_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        table = get_metadata().tables[self.table_name]
        with self.engine.connect() as conn:
            rs = conn.execute(
                select(table).order_by(desc(table.c.id)).limit(limit)
            )
            return [dict(r._mapping) for r in rs]

    def assert_not_main_messages(self) -> None:
        if self.table_name == self.FORBIDDEN_TABLE:
            raise MemoryError("违反隔离约束：主管 Store 不得接触 main_agent_messages")