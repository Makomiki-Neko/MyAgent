"""LLM factory：按 Agent 名实例化 + 缓存 ChatModel。

入口：
- `get_llm(agent_name)`        — 对话型 Agent 模型（main/supervisor/executor）
- `get_summary_llm()`          — 总结裁剪中间件专用模型
- `clear_cache()`              — 热更新 model/profile 时刷新缓存
"""

from __future__ import annotations

from typing import Any

from nekoagent.config.loader import Config, get_config
from nekoagent.llm.profile_loader import build_llm_for_profile
from nekoagent.observability.exceptions import LLMConfigError
from nekoagent.observability.logging import get_logger

_log = get_logger("llm.factory")

_cache: dict[tuple[str, str], Any] = {}


def get_llm(agent_name: str, cfg: Config | None = None, **overrides: Any):
    """返回某 Agent 绑定的 LangChain ChatModel。带 profile 级缓存。"""

    cfg = cfg or get_config()
    agent = cfg.agents.get(agent_name)
    if agent is None:
        raise LLMConfigError(f"config.yaml 中未定义 agent '{agent_name}'")

    cache_key = (agent_name, agent.llm_profile)
    cached = _cache.get(cache_key)
    if cached is not None and not overrides:
        return cached

    llm = build_llm_for_profile(agent.llm_profile, cfg, **overrides)
    if not overrides:
        _cache[cache_key] = llm
        _log.debug("LLM 已缓存 agent=%s profile=%s", agent_name, agent.llm_profile)
    return llm


def get_summary_llm(cfg: Config | None = None):
    """总结裁剪中间件使用的独立模型（来自 auxiliary_models.summary_llm_profile）。"""

    cfg = cfg or get_config()
    profile_name = cfg.auxiliary_models.summary_llm_profile
    cache_key = ("__summary__", profile_name)
    if cache_key in _cache:
        return _cache[cache_key]
    llm = build_llm_for_profile(profile_name, cfg)
    _cache[cache_key] = llm
    _log.debug("总结 LLM 已缓存 profile=%s", profile_name)
    return llm


def clear_cache() -> None:
    """热更新后调用以释放旧实例。"""

    _cache.clear()
    _log.info("LLM 缓存已清空")