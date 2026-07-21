"""YAML 写回。

Tradeoff：使用 PyYAML dump 会丢失原文件注释。MVP 阶段可接受，
未来可引入 ruamel.yaml 以保留格式与注释。

策略：
- 校验后再写：写回前先 `RootConfig.model_validate(new_raw)`，校验不通过则拒绝。
- 写回使用临时文件 + 原子替换，避免半写状态破坏原配置。
- 人设单独写：`save_persona(path, persona_dict)`，便于 UI 人设热更新。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from nekoagent.config.loader import Config, get_config, load_config
from nekoagent.observability.exceptions import ConfigError
from nekoagent.observability.logging import get_logger
from nekoagent.config.schema import PersonaConfig, RootConfig

_log = get_logger("config.writeback")


def save_config(new_raw: dict[str, Any], path: str | None = None) -> Config:
    """校验 + 原子写回 + 重新加载单例。返回更新后的 Config。"""

    target = path or get_config().source_path
    abs_target = os.path.abspath(target)

    try:
        validated = RootConfig.model_validate(new_raw)
    except ValidationError as exc:
        from nekoagent.config.loader import _format_pydantic_errors
        raise ConfigError(
            "配置校验失败，未写入磁盘。\n" + _format_pydantic_errors(exc)
        ) from exc

    try:
        Path(abs_target).parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".nekoagent_cfg_", suffix=".yaml", dir=str(Path(abs_target).parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    new_raw,
                    f,
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False,
                )
            os.replace(tmp_path, abs_target)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError as exc:
        raise ConfigError(f"无法写入配置文件 {abs_target}：{exc}", cause=exc) from exc

    _log.info("配置已写回并重新加载：%s", abs_target)
    return load_config(abs_target)


def save_persona(path: str, persona_raw: dict[str, Any]) -> PersonaConfig:
    """人设保存：先校验后写回 + 加载新 Persona 实例。"""

    abs_target = os.path.abspath(path)
    try:
        validated = PersonaConfig.model_validate(persona_raw)
    except ValidationError as exc:
        from nekoagent.config.loader import _format_pydantic_errors
        raise ConfigError("人设校验失败，未写入磁盘。\n" + _format_pydantic_errors(exc)) from exc

    try:
        Path(abs_target).parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".nekoagent_persona_", suffix=".yaml", dir=str(Path(abs_target).parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    persona_raw,
                    f,
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False,
                )
            os.replace(tmp_path, abs_target)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError as exc:
        raise ConfigError(f"无法写入人设文件 {abs_target}：{exc}", cause=exc) from exc

    _log.info("人设已保存：%s", abs_target)
    return validated


def reset_persona_cache(agent_name: str | None = None) -> None:
    """保存后让 loader 在下次调用时重读。"""

    cfg = get_config()
    if agent_name:
        cfg._persona_cache.pop(agent_name, None)
    else:
        cfg._persona_cache.clear()