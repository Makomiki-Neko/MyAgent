"""LLM profile loader：根据 YAML 中 `llm_profiles.<name>` 解析 LangChain ChatModel。

设计要点：
- 惰性 import langchain_openai，避免未安装时影响 config 加载。
- 同一 profile 仅实例化一次（按 agent_name + profile_name 缓存）。
- 失败时抛 LLMConfigError / LLMConnectionError，UI 收到友好文案而非堆栈。
- 辅助模型独立入口：`get_summary_llm` 使用 `auxiliary_models.summary_llm_profile`。
"""

from __future__ import annotations

import os
from typing import Any

from nekoagent.config.loader import Config, get_config, resolve_api_key
from nekoagent.observability.exceptions import LLMConfigError, LLMConnectionError
from nekoagent.observability.logging import get_logger

_log = get_logger("llm")


def build_llm_for_profile(profile_name: str, cfg: Config | None = None, **overrides: Any):
    """按 profile 名解析 + 实例化一个 LangChain ChatModel（OpenAI 兼容）。"""

    cfg = cfg or get_config()
    profile = cfg.llm_profiles.get(profile_name)
    if profile is None:
        raise LLMConfigError(f"LLM profile '{profile_name}' 未在 config.yaml 的 llm_profiles 中定义")

    api_key = resolve_api_key(profile_name, cfg)
    if not api_key:
        raise LLMConfigError(
            f"LLM profile '{profile_name}' 未获得可用 api_key（环境变量 {profile.api_key_env} 未设置 或 字面值缺失）"
        )

    kwargs: dict[str, Any] = dict(
        base_url=profile.base_url,
        api_key=api_key,
        model=profile.model,
        temperature=profile.temperature,
    )
    if profile.max_tokens:
        kwargs["max_tokens"] = profile.max_tokens
    kwargs.update(overrides)

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise LLMConfigError(
            "未安装 langchain_openai，请先 `pip install langchain-openai`",
            cause=exc,
        ) from exc

    try:
        return ChatOpenAI(**kwargs)
    except Exception as exc:
        _log.warning("ChatOpenAI 实例化失败 profile=%s error=%s", profile_name, exc)
        raise LLMConnectionError(
            f"无法连接模型服务（profile={profile_name}, base_url={profile.base_url}）",
            cause=exc,
        ) from exc