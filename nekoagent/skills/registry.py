"""Skill registry：启动期扫描 skills_dir 仅登记，运行期按需懒加载。

Skill YAML schema（与样例一致）：
```yaml
name: <skill-name>
description: <skill 用途>
type: prompt         # prompt | tool_alias
system_prompt: |
  <加载到子 Agent 上下文的提示>
tools: [<server.tool>]  # 可选；仅 type=tool_alias 时引用 MCP 工具
```

隔离：SkillRegistry 单例持有扫描结果；具体加载呈单次（任务内缓存由 supervisor/executor 自行控制是否复用）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from nekoagent.config.loader import Config, get_config
from nekoagent.config.schema import PersonaConfig
from nekoagent.observability.exceptions import SkillError
from nekoagent.observability.logging import get_logger

_log = get_logger("skills")


class SkillRegistry:
    def __init__(self, skills_dir: str) -> None:
        self.skills_dir = skills_dir
        self._index: dict[str, dict[str, Any]] = {}    # name -> {path, summary, meta}
        self._cache: dict[str, dict[str, Any]] = {}

    # ---------------- 启动期扫描 ----------------
    def scan(self) -> None:
        """扫 skills 目录，仅登记每个 skill 的 path + description；不读文件内容。"""

        self._index.clear()
        root = Path(self.skills_dir)
        if not root.exists():
            _log.warning("skills 目录不存在：%s", root)
            return
        for sub in root.iterdir():
            if not sub.is_dir():
                continue
            meta = sub / "skill.yaml"
            if not meta.exists():
                continue
            try:
                with meta.open("r", encoding="utf-8") as f:
                    partial = yaml.safe_load(f) or {}
                name = partial.get("name") or sub.name
                description = partial.get("description", "")
                self._index[name] = {
                    "path": str(meta.resolve()),
                    "summary": description,
                    "loaded": False,
                }
            except yaml.YAMLError as exc:
                _log.warning("skill 文件 YAML 解析失败 %s: %s", meta, exc)
        _log.info("SkillRegistry 已登记 %d 个 skill：%s",
                  len(self._index), list(self._index.keys()))

    def all_skills(self) -> dict[str, dict[str, Any]]:
        return dict(self._index)

    def knows(self, name: str) -> bool:
        return name in self._index

    # ---------------- 运行期懒加载 ----------------
    def load(self, name: str) -> dict[str, Any] | None:
        """按需读取 skill YAML 全文；缓存复用。"""

        if name in self._cache:
            return self._cache[name]
        if name not in self._index:
            _log.warning("请求未登记 skill：%s", name)
            return None
        meta_path = self._index[name]["path"]
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise SkillError(f"skill '{name}' 文件读取失败：{exc}", cause=exc) from exc
        # 简易校验：必含 system_prompt
        if "system_prompt" not in content:
            raise SkillError(f"skill '{name}' 缺少必填 system_prompt 字段")
        if "name" not in content:
            content["name"] = name
        self._cache[name] = content
        return content

    def clear_cache(self) -> None:
        self._cache.clear()


_registry_singleton: SkillRegistry | None = None


def get_skill_registry(cfg: Config | None = None) -> SkillRegistry:
    global _registry_singleton
    if _registry_singleton is None:
        cfg = cfg or get_config()
        _registry_singleton = SkillRegistry(cfg.skills_dir)
        _registry_singleton.scan()
    return _registry_singleton


def reset_singletons_for_test() -> None:
    global _registry_singleton
    _registry_singleton = None