"""受控异常框架。

面向用户：
- NekoAgentError 与其子类产生**友好中文文案**，UI 直接展示。
- 禁止将原始堆栈抛给用户。

面向开发者：
- 完整 traceback 通过 logging 系统写入文件日志（dev log）。
- format_friendly_error() 是 UI 调用方统一入口。
"""

from __future__ import annotations

import logging
import traceback
from typing import Any


class NekoAgentError(Exception):
    """所有 NekoAgent 受控异常基类。friend_message 为对外友好文案。"""

    friendly_message: str = "发生未知问题，请稍后重试或查看日志。"
    error_code: str = "NEKO_UNKNOWN"

    def __init__(self, friendly_message: str | None = None, *, cause: BaseException | None = None) -> None:
        self.friendly_message = friendly_message or self.friendly_message
        super().__init__(self.friendly_message)
        if cause is not None:
            self.__cause__ = cause
        logging.getLogger("nekoagent").error(
            "[%s] %s\n%s",
            self.error_code,
            self.friendly_message,
            traceback.format_exc() if cause is None else "".join(traceback.format_exception(type(cause), cause, cause.__traceback__)),
        )

    def to_user_text(self) -> str:
        return self.friendly_message


class ConfigError(NekoAgentError):
    friendly_message = "系统配置出错，请检查 YAML 文件后重试。"
    error_code = "NEKO_CONFIG"


class LLMConnectionError(NekoAgentError):
    friendly_message = "无法连接到模型服务，请稍后重试或检查 API 配置。"
    error_code = "NEKO_LLM_CONNECTION"


class LLMConfigError(NekoAgentError):
    friendly_message = "模型配置缺失或非法，请在配置面板补充后保存。"
    error_code = "NEKO_LLM_CONFIG"


class MemoryError(NekoAgentError):
    friendly_message = "记忆存储读写失败，已记录日志，请稍后重试。"
    error_code = "NEKO_MEMORY"


class RAGError(NekoAgentError):
    friendly_message = "记忆检索失败，本轮对话将不使用长期记忆。"
    error_code = "NEKO_RAG"


class MCPError(NekoAgentError):
    friendly_message = "外部工具调用失败，已暂停任务，请确认后重试。"
    error_code = "NEKO_MCP"


class SkillError(NekoAgentError):
    friendly_message = "可加载技能出现问题，已暂停任务，请确认后重试。"
    error_code = "NEKO_SKILL"


class HITLAbortedError(NekoAgentError):
    friendly_message = "任务已被用户取消。"
    error_code = "NEKO_HITL_ABORT"


def format_friendly_error(exc: BaseException) -> str:
    """UI 调用方入口：把任意异常转为友好文案；dev log 已写入完整堆栈。"""

    if isinstance(exc, NekoAgentError):
        return f"[{exc.error_code}] {exc.friendly_message}"

    logging.getLogger("nekoagent").error("未受控异常：%s\n%s", exc, traceback.format_exc())
    return "发生未知问题，请稍后重试或查看日志。"


def safe_call(func, *args: Any, **kwargs: Any):
    """简单包装：捕获任意异常并返回友好错误对象，不抛栈给调用方。"""

    try:
        return func(*args, **kwargs), None
    except NekoAgentError as exc:
        return None, exc
    except Exception as exc:  # pragma: no cover - 兜底
        return None, NekoAgentError(cause=exc)