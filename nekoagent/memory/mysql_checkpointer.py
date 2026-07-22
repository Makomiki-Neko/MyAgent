"""MySQL checkpointer 集成 + SQLite 透明回退。

- `database.backend=mysql` 时使用 `langgraph.checkpoint.mysql.MySQLSaver`
- `database.backend=sqlite` 时使用 `langgraph.checkpoint.sqlite.SqliteSaver`（实验性回退）
- 启动期：先建业务表（schema.py），再返回 checkpointer
- 全程惰性初始化，避免未配置 MySQL 时 import 失败
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine

from nekoagent.config.loader import Config, get_config
from nekoagent.memory.schema import init_business_tables
from nekoagent.observability.exceptions import MemoryError
from nekoagent.observability.logging import get_logger

_log = get_logger("memory.checkpointer")

_engine: Any = None
_saver: Any = None
_saver_ctx: Any = None


def get_engine(cfg: Config | None = None):
    """返回 SQLAlchemy Engine 单例。首次调用时同步初始化业务表，确保任何后续 SELECT 不会因表缺失而失败。"""

    global _engine
    if _engine is not None:
        return _engine
    cfg = cfg or get_config()
    db_cfg = cfg.database
    try:
        _engine = create_engine(db_cfg.url, echo=db_cfg.echo, future=True)
    except Exception as exc:
        raise MemoryError(f"无法建立数据库连接：{exc}", cause=exc) from exc

    # 关键：建一次表后再返回。这样 restore_pending_tasks / session restore 不会因 1146 报错。
    try:
        init_business_tables(_engine)
        _migrate_session_meta(_engine)
        _log.info("业务表已就绪（auto-create）")
    except Exception as exc:
        # 不抛错——让上层 graceful skip；不可能表都建不出来仍要求 scheduler 启动成功
        _log.warning("业务表自动创建失败（将以下游逻辑忽略表查询错误）：%s", exc)
    return _engine


def _migrate_session_meta(engine: Any) -> None:
    """为已有 session_meta 表添加新字段（如不存在）。"""
    try:
        from sqlalchemy import text as _sql
        with engine.begin() as c:
            for col, col_type in [("dialogue_tokens", "INTEGER DEFAULT 0"), ("task_tokens", "INTEGER DEFAULT 0")]:
                try:
                    c.execute(_sql(f"ALTER TABLE session_meta ADD COLUMN {col} {col_type}"))
                except Exception:
                    pass  # 列已存在
    except Exception:
        pass


def save_hitl_pending(session_id: str, thread_id: str, kind: str, payload: dict) -> None:
    """持久化 HITL 待处理状态到数据库。"""
    try:
        from datetime import datetime
        from nekoagent.memory.schema import get_metadata as _gm
        import json
        engine = get_engine()
        tbl = _gm().tables["hitl_pending"]
        with engine.begin() as c:
            c.execute(tbl.insert().values(
                session_id=session_id, thread_id=thread_id, kind=kind,
                payload=json.dumps(payload, ensure_ascii=False, default=str),
                created_at=datetime.utcnow(),
            ))
    except Exception:
        pass


def load_hitl_pending(session_id: str) -> list[dict]:
    """加载指定会话的所有待处理 HITL 状态。"""
    try:
        from sqlalchemy import select as _select, desc as _desc
        import json
        engine = get_engine()
        tbl = get_metadata().tables["hitl_pending"]
        with engine.connect() as c:
            rs = c.execute(
                _select(tbl).where(tbl.c.session_id == session_id)
                .order_by(_desc(tbl.c.created_at))
            )
            results = []
            for r in rs:
                row = dict(r._mapping)
                row["payload"] = json.loads(row.get("payload") or "{}")
                results.append(row)
            return results
    except Exception:
        return []


def clear_hitl_pending(thread_id: str) -> None:
    """清除指定 thread_id 的待处理 HITL 状态（任务完成后）。"""
    try:
        from sqlalchemy import delete as _delete
        engine = get_engine()
        tbl = get_metadata().tables["hitl_pending"]
        with engine.begin() as c:
            c.execute(_delete(tbl).where(tbl.c.thread_id == thread_id))
    except Exception:
        pass


def ensure_tables(cfg: Config | None = None) -> None:
    """显式触发一次业务表创建 + 迁移。可在 app 启动前调用以便失败时立即报错。"""

    engine = get_engine(cfg)
    init_business_tables(engine)
    _migrate_session_meta(engine)


# alias 兼容命名差异
engine_ensure_tables = ensure_tables


def get_checkpointer(cfg: Config | None = None):
    """返回 LangGraph checkpointer 单例；首次调用同时落地业务表。"""

    global _saver, _saver_ctx
    if _saver is not None:
        return _saver
    cfg = cfg or get_config()
    backend = cfg.database.backend.lower()

    engine = get_engine(cfg)
    try:
        init_business_tables(engine)
    except Exception as exc:
        raise MemoryError(f"业务表创建失败：{exc}", cause=exc) from exc

    try:
        if backend == "mysql":
            from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver  # type: ignore
            _saver_ctx = PyMySQLSaver.from_conn_string(cfg.database.url)
            _saver = _saver_ctx.__enter__()
            _saver.setup()
        elif backend == "sqlite":
            from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
            _saver = SqliteSaver.from_engine(engine) if hasattr(SqliteSaver, "from_engine") else SqliteSaver(engine)
        else:
            raise MemoryError(f"unsupported database.backend: {backend}")
    except ImportError as exc:
        raise MemoryError(
            f"未安装 LangGraph {backend} checkpointer 依赖；pip install langgraph-checkpoint-{backend}",
            cause=exc,
        ) from exc
    except Exception as exc:
        raise MemoryError(f"checkpointer 初始化失败：{exc}", cause=exc) from exc

    _log.info("checkpointer 已就绪 backend=%s", backend)
    return _saver


def close_checkpointer() -> None:
    """关闭 checkpointer + engine；热更新或退出时调用。"""

    global _saver, _saver_ctx, _engine
    try:
        if _saver_ctx is not None:
            _saver_ctx.__exit__(None, None, None)
    except Exception:
        pass
    try:
        if _saver is not None and hasattr(_saver, "close"):
            _saver.close()
    except Exception:
        pass
    try:
        if _engine is not None:
            _engine.dispose()
    except Exception:
        pass
    _saver = None
    _saver_ctx = None
    _engine = None