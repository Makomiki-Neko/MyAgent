"""上下文总结裁剪中间件。

策略：
- 输入：当前会话消息列表（LangChain 消息格式，含 id 引用）+ 超过阈值即触发。
- 步骤：
  1. 估算当前上下文 token 数（tiktoken 优先；不可用则按 1 token ≈ 2 中文字符粗估）
  2. 若超过 `memory.summarize_token_threshold`：
     a. 取中段（保留首尾若干消息作为最近上下文）messages。
     b. 调用 `get_summary_llm()` 生成结构化总结。
     c. 调用 MainUserMemoryStore.replace_with_summary(...) 在 DB 中归档中段 + 插入总结 message。
- 返回裁剪后消息列表；调用方（main-agent）用其再调用 LLM。

LangChain 中间件形态：可装饰为 `Runnable` 的 pre-hook；MVP 暴露纯函数 + 装饰器入口。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from nekoagent.config.loader import Config, get_config
from nekoagent.llm.factory import get_summary_llm
from nekoagent.memory.stores import get_user_memory_store
from nekoagent.observability.exceptions import MemoryError
from nekoagent.observability.logging import get_logger

_log = get_logger("memory.summarize")

SUMMARY_SYSTEM_PROMPT = (
    "请将以下多轮对话压缩为紧凑的结构化总结。要求：\n"
    "1. 输出 JSON：{\"facts\": [\"...\"], \"preferences\": [\"...\"], \"events\": [\"...\"]}\n"
    "2. 保留关键事实、用户偏好、重要事件；丢弃寒暄与重复内容。\n"
    "3. 中文输出，单字段不超过 200 字。\n"
)

# 估算 token 的兜底常量（中文）：1 中文字符 ≈ 2 token
_CHAR_TOKEN_RATIO = 2.0


def _estimate_tokens(messages: list[BaseMessage]) -> int:
    """尽力估算 messages 的总 token 数。优先使用 tiktoken，不可用时回退字符数比例。"""

    total_chars = 0
    for m in messages:
        if isinstance(m.content, str):
            total_chars += len(m.content)
        elif isinstance(m.content, list):
            for part in m.content:
                if isinstance(part, dict) and "text" in part:
                    total_chars += len(str(part["text"]))
                elif isinstance(part, str):
                    total_chars += len(part)
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        joined = "\n".join(
            m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
            for m in messages
        )
        return len(enc.encode(joined))
    except Exception:
        return int(total_chars * _CHAR_TOKEN_RATIO)


def _classify_messages(
    messages: list[BaseMessage], keep_recent: int = 6
) -> tuple[list[BaseMessage], list[BaseMessage], list[BaseMessage]]:
    """拆为 (head, middle, tail)：head=system 与头几条，tail=最近 keep_recent 条。"""

    if len(messages) <= keep_recent + 2:
        return [], messages, []
    head = []
    rest = list(messages)
    while rest and isinstance(rest[0], SystemMessage):
        head.append(rest.pop(0))
    tail = rest[-keep_recent:]
    middle = rest[:-keep_recent]
    return head, middle, tail


def summarize_if_needed(
    messages: list[BaseMessage],
    *,
    session_id: str,
    cfg: Config | None = None,
) -> list[BaseMessage]:
    """若 token 超阈值则做中段总结并替换；否则原样返回。"""

    cfg = cfg or get_config()
    threshold = cfg.memory.summarize_token_threshold
    tokens = _estimate_tokens(messages)
    if tokens <= threshold:
        return messages

    head, middle, tail = _classify_messages(messages)
    if not middle:
        return messages

    _log.info("总结中间件触发：估算 tokens=%s 阈值=%s 中段=%s 条", tokens, threshold, len(middle))

    try:
        llm = get_summary_llm(cfg)
        joined = "\n".join(
            f"[{m.type}] {m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)}"
            for m in middle
        )
        summary_msg = llm.invoke([SystemMessage(content=SUMMARY_SYSTEM_PROMPT), HumanMessage(content=joined)])
        summary_text = summary_msg.content if isinstance(summary_msg.content, str) else json.dumps(summary_msg.content, ensure_ascii=False)
    except MemoryError:
        raise
    except Exception as exc:
        raise MemoryError(f"总结中间件调用 summary LLM 失败：{exc}", cause=exc) from exc

    try:
        store = get_user_memory_store()
        # 中段消息映射回 DB：通过消息 id 关联（若 messages 带元数据 nekoagent_msg_id）
        archived_ids: list[int] = []
        for m in middle:
            mid = _get_msg_db_id(m)
            if mid is not None:
                archived_ids.append(mid)
        summary_id = store.replace_with_summary(
            archived_ids=archived_ids,
            session_id=session_id,
            summary=f"[系统总结] {summary_text}",
            tokens=_estimate_tokens([summary_msg]),
        )
    except Exception as exc:
        raise MemoryError(f"总结写入主 Agent 记忆失败：{exc}", cause=exc) from exc

    summarized_msg = SystemMessage(content=f"[系统总结] {summary_text}", id=f"summary-{summary_id}")
    return head + [summarized_msg] + tail


def _get_msg_db_id(m: BaseMessage) -> int | None:
    """从消息附加元数据中取出 DB 行 id（如有）。"""

    md = getattr(m, "additional_kwargs", {}) or {}
    return md.get("nekoagent_msg_id")