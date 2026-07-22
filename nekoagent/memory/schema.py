"""SQL schema：补充 LangGraph checkpointer 默认表之外的 NekoAgent 业务表。

依据 design D4：
- langgraph_state / langgraph_writes / langgraph_migrations ← 由 langgraph MySQLSaver 自动创建
- main_agent_messages        ← 用户对话原始消息 + 归档标记
- supervisor_proc_mem        ← 主管 Agent 程序性记忆
- session_meta               ← 会话元数据
- task_meta                  ← 任务元数据（主管子图 id、状态等）
- token_stats                ← 双管线 Token 累计（dialogue / task 分离）
"""

from __future__ import annotations

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    TIMESTAMP,
    Table,
)
from sqlalchemy.engine import Engine


def build_metadata() -> MetaData:
    """构造 NekoAgent 业务表元数据。create_all(engine) 后即落地。"""

    md = MetaData()

    Table(
        "main_agent_messages",
        md,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("session_id", String(64), nullable=False, index=True),
        Column("role", String(16), nullable=False),           # user | assistant | system | tool
        Column("content", Text, nullable=False),
        Column("tokens", Integer, default=0),
        Column("archived", Integer, default=0),               # 0/1：是否被总结裁剪归档
        Column("created_at", TIMESTAMP, nullable=False),
        Column("summary_of", Integer, ForeignKey("main_agent_messages.id"), nullable=True),
        Index("ix_main_agent_messages_session_archived", "session_id", "archived"),
    )

    Table(
        "supervisor_proc_mem",
        md,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("task_signature", String(255), nullable=False, index=True),  # 同类任务签名（如任务意图相似度哈希）
        Column("task_summary", Text, nullable=False),
        Column("teardown", Text, nullable=False),                            # JSON: 任务拆解结构
        Column("exception_notes", Text, default=""),
        Column("created_at", TIMESTAMP, nullable=False),
        Column("last_referenced_at", TIMESTAMP, nullable=True),
        Index("ix_supervisor_proc_mem_signature", "task_signature"),
    )

    Table(
        "session_meta",
        md,
        Column("session_id", String(64), primary_key=True),
        Column("title", String(255), default="新会话"),
        Column("created_at", TIMESTAMP, nullable=False),
        Column("updated_at", TIMESTAMP, nullable=False),
        Column("is_active", Integer, default=1),
        Column("dialogue_tokens", Integer, default=0),
        Column("task_tokens", Integer, default=0),
    )

    Table(
        "task_meta",
        md,
        Column("task_id", String(64), primary_key=True),
        Column("session_id", String(64), ForeignKey("session_meta.session_id"), nullable=False, index=True),
        Column("thread_id", String(128), nullable=True),                     # LangGraph thread / checkpointer key
        Column("title", String(255), default="未命名任务"),
        Column("status", String(32), nullable=False, default="pending"),     # pending | planning | awaiting_user | running | done | cancelled | error
        Column("progress", Float, default=0.0),
        Column("plan_json", Text, default=""),
        Column("created_at", TIMESTAMP, nullable=False),
        Column("updated_at", TIMESTAMP, nullable=False),
        Column("error_log", Text, default=""),
    )

    Table(
        "token_stats",
        md,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("session_id", String(64), ForeignKey("session_meta.session_id"), nullable=False, index=True),
        Column("batch_kind", String(32), nullable=False),       # dialogue | task
        Column("tokens", Integer, default=0),
        Column("agent_role", String(32), default=""),            # main_agent | supervisor_agent | executor_agent | mcp
        Column("ts", DateTime, nullable=False),
        Index("ix_token_stats_session_kind", "session_id", "batch_kind"),
    )

    Table(
        "hitl_pending",
        md,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("session_id", String(64), ForeignKey("session_meta.session_id"), nullable=False, index=True),
        Column("thread_id", String(128), nullable=False),
        Column("kind", String(32), nullable=False),           # plan_ready | info_request
        Column("payload", Text, default=""),                  # JSON 序列化的完整状态
        Column("created_at", TIMESTAMP, nullable=False),
    )

    Table(
        "executor_logs",
        md,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("thread_id", String(128), nullable=False, index=True),
        Column("subtask_name", String(255), nullable=False),
        Column("subtask_goal", Text, default=""),
        Column("skills_used", Text, default=""),
        Column("tools_used", Text, default=""),
        Column("result", Text, default=""),
        Column("status", String(32), default="done"),           # done | retry | error
        Column("created_at", TIMESTAMP, nullable=False),
    )

    Table(
        "supervisor_logs",
        md,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("thread_id", String(128), nullable=False, index=True),
        Column("session_id", String(64), nullable=True),
        Column("event", String(64), nullable=False),            # plan_ready | subtask_start | subtask_review | info_request | complete | cancel
        Column("detail", Text, default=""),
        Column("created_at", TIMESTAMP, nullable=False),
    )

    return md


_metadata: MetaData | None = None


def get_metadata() -> MetaData:
    global _metadata
    if _metadata is None:
        _metadata = build_metadata()
    return _metadata


def init_business_tables(engine: Engine) -> None:
    """在 PyLance/SQLAlchemy engine 上创建业务表。"""

    md = get_metadata()
    md.create_all(engine, checkfirst=True)