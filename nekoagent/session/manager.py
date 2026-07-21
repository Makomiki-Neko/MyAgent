"""会话管理实现。

- `session_meta` 表持久化（SessionManager 重启后可读回已存会话）。
- 临时对话上下文（messages 缓存）由 MainAgent 通过 `main_agent_messages.session_id` 直接驱动；
  SessionManager 只维护元数据：title / 活跃态 / created/updated。
- 全局长期记忆跨会话共享：通过 RAG pipeline 的 `rag_search(query)` 而非 per-session 内存读取。
- 重启后 `restore_sessions()` 拉 DB 未删除的会话回内存索引。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import delete as sql_delete
from sqlalchemy import insert, select, update

from nekoagent.memory.schema import get_metadata
from nekoagent.observability.exceptions import NekoAgentError
from nekoagent.observability.logging import get_logger

_log = get_logger("session")


@dataclass
class Session:
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    is_active: bool = True


class SessionManager:
    def __init__(self) -> None:
        self._cache: dict[str, Session] = {}
        self._active_id: str | None = None

    # ----- CRUD -----
    def create_session(self, title: str = "新会话") -> Session:
        session_id = uuid.uuid4().hex[:16]
        session = Session(
            session_id=session_id,
            title=title,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self._persist(session, mode="insert")
        self._cache[session_id] = session
        self._active_id = session_id
        _log.info("新建会话 id=%s title=%s", session_id, title)
        return session

    def list_sessions(self) -> list[Session]:
        return sorted(
            self._cache.values(),
            key=lambda s: s.updated_at,
            reverse=True,
        )

    def switch_session(self, session_id: str) -> Session:
        if session_id not in self._cache:
            raise NekoAgentError(f"会话 {session_id} 不存在，无法切换。")
        self._active_id = session_id
        session = self._cache[session_id]
        session.is_active = True
        session.updated_at = datetime.utcnow()
        self._persist(session, mode="update")
        return session

    def rename_session(self, session_id: str, new_title: str) -> Session:
        if session_id not in self._cache:
            raise NekoAgentError(f"会话 {session_id} 不存在，无法重命名。")
        session = self._cache[session_id]
        session.title = new_title.strip() or session.title
        session.updated_at = datetime.utcnow()
        self._persist(session, mode="update")
        return session

    def update_session_tokens(self, session_id: str, dialogue_tokens: int, task_tokens: int) -> None:
        try:
            from nekoagent.memory.mysql_checkpointer import get_engine
            engine = get_engine()
            table = get_metadata().tables["session_meta"]
            with engine.begin() as conn:
                conn.execute(
                    update(table).where(table.c.session_id == session_id).values(
                        dialogue_tokens=dialogue_tokens,
                        task_tokens=task_tokens,
                    )
                )
        except Exception as exc:
            _log.debug("session token 更新失败（忽略）：%s", exc)

    def delete_session(self, session_id: str) -> None:
        if session_id not in self._cache:
            _log.warning("待删除会话不存在：%s", session_id)
            return
        del self._cache[session_id]
        if self._active_id == session_id:
            self._active_id = None
        # 同步删除 DB 行（软删除：is_active=0，保留消息和 token_stats 以便历史查询）
        try:
            from nekoagent.memory.mysql_checkpointer import get_engine
            engine = get_engine()
            table = get_metadata().tables["session_meta"]
            with engine.begin() as conn:
                conn.execute(
                    update(table).where(table.c.session_id == session_id).values(
                        is_active=0, updated_at=datetime.utcnow()
                    )
                )
            _log.info("会话已软删除 id=%s", session_id)
        except Exception as exc:
            _log.warning("会话软删除持久化失败（不阻塞）：%s", exc)

    # ----- current active -----
    @property
    def active_session_id(self) -> str | None:
        return self._active_id

    def get_session(self, session_id: str) -> Session | None:
        return self._cache.get(session_id)

    # ----- 持久化 / 恢复 -----
    def _persist(self, session: Session, *, mode: str) -> None:
        try:
            from nekoagent.memory.mysql_checkpointer import get_engine
            engine = get_engine()
            table = get_metadata().tables["session_meta"]
            with engine.begin() as conn:
                if mode == "insert":
                    conn.execute(insert(table).values(
                        session_id=session.session_id,
                        title=session.title,
                        created_at=session.created_at,
                        updated_at=session.updated_at,
                        is_active=1,
                    ))
                else:
                    conn.execute(
                        update(table).where(table.c.session_id == session.session_id).values(
                            title=session.title,
                            updated_at=session.updated_at,
                            is_active=1 if session.is_active else 0,
                        )
                    )
        except Exception as exc:
            _log.warning("session 持久化失败（忽略）：%s", exc)

    def restore_sessions(self) -> list[Session]:
        try:
            from nekoagent.memory.mysql_checkpointer import get_engine
            engine = get_engine()
            table = get_metadata().tables["session_meta"]
            with engine.connect() as conn:
                rs = conn.execute(
                    select(table).where(table.c.is_active == 1)
                    .order_by(table.c.updated_at.desc())
                )
                rows = [dict(r._mapping) for r in rs]
            for row in rows:
                session = Session(
                    session_id=row["session_id"],
                    title=row["title"] or "未命名",
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    is_active=True,
                )
                self._cache[session.session_id] = session
            self._active_id = None if not self._cache else next(iter(self._cache))
            _log.info("session 恢复：找回 %d 个会话", len(rows))
            return list(self._cache.values())
        except Exception as exc:
            _log.warning("session 恢复失败（仅在内存运行）：%s", exc)
            return []


# 默认单例
_session_singleton: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _session_singleton
    if _session_singleton is None:
        mgr = SessionManager()
        mgr.restore_sessions()
        if not mgr.list_sessions():
            mgr.create_session()
        _session_singleton = mgr
    return _session_singleton


def create_session(title: str = "新会话") -> Session:
    return get_session_manager().create_session(title)


def list_sessions() -> list[Session]:
    return get_session_manager().list_sessions()


def switch_session(session_id: str) -> Session:
    return get_session_manager().switch_session(session_id)


def rename_session(session_id: str, new_title: str) -> Session:
    return get_session_manager().rename_session(session_id, new_title)


def delete_session(session_id: str) -> None:
    get_session_manager().delete_session(session_id)