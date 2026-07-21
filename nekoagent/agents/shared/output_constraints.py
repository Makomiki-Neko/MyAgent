"""Agent 输出约束 + 校验失败重试策略。

设计：
- `bind_structured(llm, schema)` 返回 `llm.with_structured_output(schema)`，由 LangChain 自选最优策略。
- `invoke_with_retry(llm, schema, messages, cfg)` 用 tenacity 装饰；失败时带前一次错误回灌模型再试。
- 重试耗尽抛 `NekoAgentError`（友好文案），交由 HITL 中间件处理。
"""

from __future__ import annotations

from typing import Any, Type, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from nekoagent.config.loader import Config, get_config
from nekoagent.observability.exceptions import NekoAgentError
from nekoagent.observability.logging import get_logger

_log = get_logger("agent.output_constraints")

T = TypeVar("T", bound=BaseModel)


def bind_structured(llm: Any, schema: Type[BaseModel]):
    """对 LLM 绑定结构化输出（优先 JSON Schema，回退 function_calling）。"""

    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError, TypeError) as exc:
        _log.warning("LLM 不支持结构化输出绑定（OpenAI tools），回退到纯 prompt 模式：%s", exc)
        return None


def _try_parse(content: str, schema: Type[T]) -> T | None:
    """从自由文本响应中尽力解析出 schema 对象。"""

    if not content:
        return None
    try:
        import json
        # 尝试从文本中抠出 JSON 块
        text = content.strip()
        if "```" in text:
            start = text.find("```")
            end = text.rfind("```")
            chunk = text[start:end].strip().strip("`")
            if chunk.startswith("json"):
                chunk = chunk[4:]
            candidate = chunk.strip()
        else:
            candidate = text
        return schema.model_validate_json(candidate)
    except (ValidationError, ValueError, json.JSONDecodeError):
        return None


def invoke_with_retry(
    llm: Any,
    schema: Type[T],
    messages: list[BaseMessage],
    cfg: Config | None = None,
    *,
    on_giveup_message: str = "Agent 输出无法满足结构化约束，已触发友好回退。",
) -> T:
    """调用支持结构化输出的 LLM；失败带修正提示自动重试，最终抛 NekoAgentError。

    实现：先用 `with_structured_output`。若该 LLM/Profile 不支持（如本地 ollama），
    则退化为"自然语言响应 → JSON 解析 + schema 校验"，仍走重试链路。
    """

    cfg = cfg or get_config()
    retry_cfg = cfg.output_retry
    bound = bind_structured(llm, schema)
    last_error: Exception | None = None
    current_messages = list(messages)

    for attempt in range(1, retry_cfg.max_attempts + 1):
        try:
            if bound is not None:
                obj = bound.invoke(current_messages)
            else:
                raw = llm.invoke(current_messages)
                content = raw.content if isinstance(raw, BaseMessage) else str(raw)
                obj = _try_parse(content if isinstance(content, str) else str(content), schema)
                if obj is None:
                    raise ValueError("Failed to parse structured output from natural language response")
            if isinstance(obj, schema):
                return obj
            # 某些 LangChain 实现返回 dict
            if isinstance(obj, dict):
                return schema.model_validate(obj)
            raise ValueError(f"Unexpected structured output type: {type(obj).__name__}")
        except (ValidationError, ValueError, Exception) as exc:
            last_error = exc
            _log.warning("结构化输出尝试 %d/%d 失败：%s", attempt, retry_cfg.max_attempts, exc)
            if attempt >= retry_cfg.max_attempts:
                break
            # 带修正提示回灌再试
            correction = (
                f"上一次响应不符合要求的结构化 schema `{schema.__name__}`。"
                f"错误：{exc}。请仅输出合规的 JSON 对象，不要附加自然语言解释，不要包裹 markdown 代码块以外的文字。"
            )
            try:
                last_ai = current_messages[-1] if current_messages else None
                if isinstance(last_ai, AIMessage):
                    current_messages = current_messages[:-1]
                current_messages.append(SystemMessage(content=correction))
            except Exception:
                pass
            # tenacity 退避由 sleep 模拟以保持同步语义
            import time
            delay = min(retry_cfg.backoff_initial * (2 ** (attempt - 1)), retry_cfg.backoff_max)
            time.sleep(delay)

    raise NekoAgentError(
        on_giveup_message + f"（已尝试 {retry_cfg.max_attempts} 次；最近一次错误：{last_error}）",
        cause=last_error,
    )