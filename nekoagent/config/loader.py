"""配置加载器。

设计：
- `load_config(path)` 读取 YAML → 保留 raw dict（用于 writeback 保序）+ 同步构造 Pydantic RootConfig。
- `get_config()` 返回单例；若未加载则用默认路径 ./config.yaml。
- 人设文件由 `load_persona(path)` 单独加载（解耦主配置）。
- env 变量在 `LLMProfile` 解析后注入：实际 api_key 由 `resolve_api_key()` 获取，避免明文落 Pydantic 模型。
- 失败时统一抛 ConfigError（友好文案 + dev log）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from nekoagent.config.schema import PersonaConfig, RootConfig
from nekoagent.observability.exceptions import ConfigError
from nekoagent.observability.logging import get_logger

_log = get_logger("config")

_DEFAULT_PATH = "./config.yaml"

_config_singleton: "Config | None" = None


class Config:
    """持有 raw YAML dict + 已校验的 Pydantic 模型 + persona 缓存。"""

    def __init__(self, raw: dict[str, Any], validated: RootConfig, source_path: str) -> None:
        self.raw = raw
        self.validated = validated
        self.source_path = source_path
        self._persona_cache: dict[str, PersonaConfig] = {}

    # ---- 透明代理：常用字段直接转发 ----
    @property
    def root(self) -> RootConfig:
        return self.validated

    @property
    def llm_profiles(self):
        return self.validated.llm_profiles

    @property
    def agents(self):
        return self.validated.agents

    @property
    def rag(self):
        return self.validated.rag

    @property
    def memory(self):
        return self.validated.memory

    @property
    def database(self):
        return self.validated.database

    @property
    def auxiliary_models(self):
        return self.validated.auxiliary_models

    @property
    def mcp_servers(self):
        return self.validated.mcp_servers

    @property
    def skills_dir(self):
        return self.validated.skills_dir

    @property
    def logging_cfg(self):
        return self.validated.logging

    @property
    def task_queue(self):
        return self.validated.task_queue

    @property
    def output_retry(self):
        return self.validated.output_retry

    # ---- 人设 ----
    def load_persona(self, agent_name: str) -> PersonaConfig:
        """加载/缓存某个 Agent 的人设 YAML。每次调用重读磁盘以便热更新。"""

        agent_cfg = self.validated.agents.get(agent_name)
        if agent_cfg is None:
            raise ConfigError(f"未在 config.yaml 中找到 agent '{agent_name}' 的人设配置")
        path = agent_cfg.system_prompt_file
        try:
            persona = load_persona(path)
        except ConfigError as exc:
            raise ConfigError(f"无法为 Agent '{agent_name}' 加载人设 ({path})：{exc.friendly_message}") from exc
        self._persona_cache[agent_name] = persona
        return persona

    def get_persona_cached(self, agent_name: str) -> PersonaConfig:
        if agent_name in self._persona_cache:
            return self._persona_cache[agent_name]
        return self.load_persona(agent_name)


def load_config(path: str = _DEFAULT_PATH) -> Config:
    """从 YAML 加载并验证配置，更新单例。"""

    global _config_singleton

    abs_path = os.path.abspath(path)
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"配置文件未找到：{abs_path}，请确认启动目录。", cause=exc) from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"配置文件 YAML 语法错误：{exc}", cause=exc) from exc

    try:
        validated = RootConfig.model_validate(raw)
    except ValidationError as exc:
        messages = _format_pydantic_errors(exc)
        raise ConfigError(
            f"配置校验失败：\n{messages}\n请检查 config.yaml 后重启或在 UI 配置面板修正。"
        ) from exc

    _config_singleton = Config(raw=raw, validated=validated, source_path=abs_path)
    _log.info("配置加载完成：%s（version=%s）", abs_path, validated.version)
    return _config_singleton


def reload_config(path: str | None = None) -> Config:
    """重新从磁盘加载（用于热更新或写回后校验）。"""

    return load_config(path or (_config_singleton.source_path if _config_singleton else _DEFAULT_PATH))


def get_config() -> Config:
    """获取当前单例；若未加载则用默认路径加载。"""

    global _config_singleton
    if _config_singleton is None:
        return load_config(_DEFAULT_PATH)
    return _config_singleton


def set_config_for_tests(cfg: Config | None) -> None:
    """测试可注入/清空单例。生产代码勿用。"""

    global _config_singleton
    _config_singleton = cfg


def load_persona(path: str) -> PersonaConfig:
    """独立加载人设 YAML（用于人设热更新与单测）。"""

    abs_path = os.path.abspath(path)
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"人设文件未找到：{abs_path}", cause=exc) from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"人设 YAML 语法错误：{abs_path}：{exc}", cause=exc) from exc
    try:
        return PersonaConfig.model_validate(raw)
    except ValidationError as exc:
        messages = _format_pydantic_errors(exc)
        raise ConfigError(f"人设校验失败：{abs_path}\n{messages}") from exc


def resolve_api_key(profile_name: str, cfg: Config | None = None) -> str | None:
    """根据 profile 配置解析 api_key：优先使用明文 api_key；若仅有 api_key_env 则从环境取；仍未取到时回退 dummy。"""

    cfg = cfg or get_config()
    profile = cfg.llm_profiles.get(profile_name)
    if profile is None:
        raise ConfigError(f"未找到 LLM profile '{profile_name}'")
    if profile.api_key:
        return profile.api_key
    if profile.api_key_env:
        env_val = os.environ.get(profile.api_key_env)
        if env_val:
            return env_val
    # 本地兼容服务（如 Ollama）允许 dummy key
    return "dummy-not-set"


def _format_pydantic_errors(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msg = err.get("msg", "")
        lines.append(f"  - {loc or '(root)'}: {msg}")
    return "\n".join(lines)


def find_config_path() -> str:
    """暴露默认配置路径供 UI 读 / 写。"""

    return _DEFAULT_PATH