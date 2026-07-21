"""会话管理：CRUD + 临时上下文隔离 + 全局长期记忆跨会话共享。"""

from nekoagent.session.manager import (
    Session,
    SessionManager,
    get_session_manager,
    create_session,
    list_sessions,
    switch_session,
    rename_session,
    delete_session,
)

__all__ = [
    "Session",
    "SessionManager",
    "get_session_manager",
    "create_session",
    "list_sessions",
    "switch_session",
    "rename_session",
    "delete_session",
]