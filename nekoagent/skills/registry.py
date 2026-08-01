"""Skill registry：启动期扫描 skills_dir 仅登记，运行期按需懒加载。

支持两种格式：
- .yaml（标准 schema：name / description / system_prompt / tools）
- .md（YAML front matter + markdown 正文，正文作为 system_prompt）

隔离：SkillRegistry 单例持有扫描结果；具体加载呈单次（任务内缓存由 supervisor/executor 自行控制是否复用）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from nekoagent.config.loader import Config, get_config
from nekoagent.config.schema import PersonaConfig
from nekoagent.observability.exceptions import SkillError
from nekoagent.observability.logging import get_logger

_log = get_logger("skills")

_FRONT_MATCH_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)


def _parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """解析 YAML front matter，返回 (metadata_dict, body_text)。无 front matter 时 metadata 为空 dict。"""
    m = _FRONT_MATCH_RE.match(text)
    if not m:
        return {}, text
    meta = yaml.safe_load(m.group(1)) or {}
    body = text[m.end():]
    return meta, body


class SkillRegistry:
    def __init__(self, skills_dir: str) -> None:
        self.skills_dir = skills_dir
        self._index: dict[str, dict[str, Any]] = {}    # name -> {path, summary, format}
        self._cache: dict[str, dict[str, Any]] = {}

    # ---------------- 启动期扫描 ----------------
    def scan(self) -> None:
        """扫 skills 目录，仅登记每个 skill 的 path + description；不读文件正文。"""

        self._index.clear()
        root = Path(self.skills_dir)
        if not root.exists():
            _log.warning("skills 目录不存在：%s", root)
            return
        for sub in root.iterdir():
            if not sub.is_dir():
                continue
            # 优先 skill.yaml，其次 skill.md
            skill_file = sub / "skill.yaml"
            fmt = "yaml"
            if not skill_file.exists():
                skill_file = sub / "skill.md"
                fmt = "md"
            if not skill_file.exists():
                continue
            try:
                raw = skill_file.read_text(encoding="utf-8")
                if fmt == "yaml":
                    partial = yaml.safe_load(raw) or {}
                    name = partial.get("name") or sub.name
                    description = partial.get("description", "")
                else:
                    meta, _ = _parse_front_matter(raw)
                    name = meta.get("name") or sub.name
                    description = meta.get("description", "")
                self._index[name] = {
                    "path": str(skill_file.resolve()),
                    "summary": description,
                    "format": fmt,
                }
                _log.info("Skill 识别: name=%s description=%s format=%s path=%s",
                          name, description, fmt, skill_file.resolve())
            except (yaml.YAMLError, Exception) as exc:
                _log.warning("skill 文件解析失败 %s: %s", skill_file, exc)
        _log.info("Skill 识别完成: 共 %d 个", len(self._index))

    def all_skills(self) -> dict[str, dict[str, Any]]:
        return dict(self._index)

    def knows(self, name: str) -> bool:
        return name in self._index

    # ---------------- 运行期懒加载 ----------------
    def load(self, name: str) -> dict[str, Any] | None:
        """按需读取 skill 全文；缓存复用。支持 .yaml 和 .md 格式。"""

        if name in self._cache:
            return self._cache[name]
        if name not in self._index:
            _log.warning("请求未登记 skill：%s", name)
            return None
        entry = self._index[name]
        meta_path = entry["path"]
        fmt = entry.get("format", "yaml")
        try:
            raw = Path(meta_path).read_text(encoding="utf-8")
        except OSError as exc:
            raise SkillError(f"skill '{name}' 文件读取失败：{exc}", cause=exc) from exc

        if fmt == "yaml":
            content = yaml.safe_load(raw) or {}
            if "system_prompt" not in content:
                raise SkillError(f"skill '{name}' 缺少必填 system_prompt 字段")
            if "name" not in content:
                content["name"] = name
        else:
            meta, body = _parse_front_matter(raw)
            content = dict(meta)
            content["system_prompt"] = body.strip()
            if "name" not in content:
                content["name"] = name
            if not content.get("system_prompt"):
                raise SkillError(f"skill '{name}' (.md) 缺少正文内容")
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