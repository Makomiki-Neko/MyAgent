"""统一日志配置。

依据 config.logging.* 初始化 Python logging：
- 控制台展示用户友好级别（默认 INFO 摘要）
- 文件日志保留完整 DEBUG 与堆栈（dev log）
- 滚动文件 / 备份按配置生效
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Any


_CONFIGURED = False


def configure_logging(log_config: dict[str, Any] | None = None) -> logging.Logger:
    """根据 config.logging 段初始化全局 logging。可重复调用（仅首次生效）。"""

    global _CONFIGURED
    if _CONFIGURED:
        return logging.getLogger("nekoagent")

    cfg = log_config or {}
    level_name = str(cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    file_path = cfg.get("file", "./logs/nekoagent.log")
    rotate = bool(cfg.get("rotate", True))
    max_bytes = int(cfg.get("max_bytes", 10 * 1024 * 1024))
    backup_count = int(cfg.get("backup_count", 5))
    expose_stacktrace = bool(cfg.get("expose_stacktrace_to_dev_log", True))

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_console_formatter())
        handler.setLevel(level)
        root.addHandler(handler)

    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        if rotate:
            file_handler: logging.Handler = logging.handlers.RotatingFileHandler(
                file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
        else:
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG if expose_stacktrace else level)
        file_handler.setFormatter(_file_formatter())
        root.addHandler(file_handler)
    except OSError as exc:
        root.warning("无法初始化文件日志（%s），仅保留控制台日志。", exc)

    root.setLevel(logging.DEBUG)
    logger = logging.getLogger("nekoagent")
    logger.info("NekoAgent 日志系统已就绪，level=%s", level_name)
    _CONFIGURED = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """获取子模块 logger；统一命名前缀 nekoagent.*"""

    if not name or name == "nekoagent":
        return logging.getLogger("nekoagent")
    if name.startswith("nekoagent."):
        return logging.getLogger(name)
    return logging.getLogger(f"nekoagent.{name}")


def _console_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _file_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _env_default(key: str, default: str) -> str:
    """读取环境变量作为兜底默认，避免 config 未加载时产生异常路径。"""

    return os.environ.get(key, default)