"""Token 双管线统计：对话 vs 任务，按 session 累加 + 全局总览。

- `dialogue` 管线计主 Agent 对话消耗
- `task` 管线计 supervisor/executor/MCP 工具往返消耗
- 双管线持久化到 `token_stats` 表，重启后总览可恢复
- `record_dialogue_tokens / record_task_tokens` 是 main_agent 与 executor/mcp 包装器调用的入口
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select

from nekoagent.memory.schema import get_metadata
from nekoagent.observability.logging import get_logger

_log = get_logger("observability.token")


def estimate_tokens(text: str) -> int:
    """中文 1 字符 ≈ 2 token 的粗估；tic缺失时回退。"""

    if not text:
        return 0
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text) * 2.0)


def _record(session_id: str, batch_kind: str, tokens: int, agent_role: str = "") -> None:
    if tokens <= 0:
        return
    try:
        from sqlalchemy import update as _update, select as _select, func as _func
        from nekoagent.memory.mysql_checkpointer import get_engine
        engine = get_engine()
        table = get_metadata().tables["token_stats"]
        meta_table = get_metadata().tables["session_meta"]
        with engine.begin() as conn:
            conn.execute(insert_row(table, session_id, batch_kind, tokens, agent_role))
            # 计算该会话累计 token 并更新 session_meta
            d = conn.execute(
                _select(_func.coalesce(_func.sum(table.c.tokens), 0))
                .where(table.c.session_id == session_id).where(table.c.batch_kind == "dialogue")
            ).scalar()
            t = conn.execute(
                _select(_func.coalesce(_func.sum(table.c.tokens), 0))
                .where(table.c.session_id == session_id).where(table.c.batch_kind == "task")
            ).scalar()
            try:
                conn.execute(
                    _update(meta_table).where(meta_table.c.session_id == session_id).values(
                        dialogue_tokens=int(d or 0), task_tokens=int(t or 0),
                    )
                )
            except Exception:
                pass  # session_meta 可能缺少新字段
    except Exception as exc:
        _log.debug("token_stats 记录失败（忽略）：%s", exc)


def record_dialogue_tokens(session_id: str, tokens: int, agent_role: str = "main_agent") -> None:
    _record(session_id, "dialogue", tokens, agent_role)


def record_task_tokens(session_id: str, tokens: int, agent_role: str = "") -> None:
    _record(session_id, "task", tokens, agent_role)


def get_session_stats(session_id: str) -> dict[str, int]:
    """返回 {dialogue_tokens, task_tokens, total}"""

    try:
        from nekoagent.memory.mysql_checkpointer import get_engine
        engine = get_engine()
        table = get_metadata().tables["token_stats"]
        with engine.connect() as conn:
            dialogue = conn.execute(
                select(func.coalesce(func.sum(table.c.tokens), 0))
                .where(table.c.session_id == session_id)
                .where(table.c.batch_kind == "dialogue")
            ).scalar()
            task = conn.execute(
                select(func.coalesce(func.sum(table.c.tokens), 0))
                .where(table.c.session_id == session_id)
                .where(table.c.batch_kind == "task")
            ).scalar()
        d = int(dialogue or 0)
        t = int(task or 0)
        return {"dialogue_tokens": d, "task_tokens": t, "total": d + t}
    except Exception as exc:
        _log.debug("token_stats 读取失败（返回 0）：%s", exc)
        return {"dialogue_tokens": 0, "task_tokens": 0, "total": 0}


def get_session_totals(session_id: str) -> dict[str, int]:
    return get_session_stats(session_id)


def get_global_totals() -> dict[str, int]:
    """跨所有会话累计总览。"""

    try:
        from nekoagent.memory.mysql_checkpointer import get_engine
        engine = get_engine()
        table = get_metadata().tables["token_stats"]
        with engine.connect() as conn:
            dialogue = conn.execute(
                select(func.coalesce(func.sum(table.c.tokens), 0))
                .where(table.c.batch_kind == "dialogue")
            ).scalar()
            task = conn.execute(
                select(func.coalesce(func.sum(table.c.tokens), 0))
                .where(table.c.batch_kind == "task")
            ).scalar()
        d = int(dialogue or 0)
        t = int(task or 0)
        return {"dialogue_tokens": d, "task_tokens": t, "total": d + t}
    except Exception as exc:
        _log.debug("token_stats global 读取失败：%s", exc)
        return {"dialogue_tokens": 0, "task_tokens": 0, "total": 0}


def insert_row(table: Any, session_id: str, batch_kind: str, tokens: int, agent_role: str) -> Any:
    """SQLAlchemy insert 表达式，便于在 begin 内复用。"""

    from sqlalchemy import insert
    return insert(table).values(
        session_id=session_id,
        batch_kind=batch_kind,
        tokens=tokens,
        agent_role=agent_role,
        ts=datetime.utcnow(),
    )