"""MemoryStore 抽象 + Main / Supervisor 隔离实现。

设计原则：
- 主 Agent 用户记忆 与 主管 Agent 程序性记忆 **物理分表**（不同表）+ **逻辑访问隔离**（不同 Store 类型）。
- 不允许跨 Agent 直接读写对方表。
- 跨 Agent 查询必须经由主 Agent 提供的标准化查询接口（Task 相关上下文）。
"""

from nekoagent.memory.stores.base import MemoryStore
from nekoagent.memory.stores.main_store import MainUserMemoryStore
from nekoagent.memory.stores.supervisor_store import SupervisorProcMemoryStore

_main_store: MainUserMemoryStore | None = None
_sup_store: SupervisorProcMemoryStore | None = None


def get_user_memory_store() -> MainUserMemoryStore:
    global _main_store
    if _main_store is None:
        _main_store = MainUserMemoryStore()
    return _main_store


def get_proc_memory_store() -> SupervisorProcMemoryStore:
    global _sup_store
    if _sup_store is None:
        _sup_store = SupervisorProcMemoryStore()
    return _sup_store


__all__ = [
    "MemoryStore",
    "MainUserMemoryStore",
    "SupervisorProcMemoryStore",
    "get_user_memory_store",
    "get_proc_memory_store",
]