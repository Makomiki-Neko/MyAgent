"""Skill 自动识别与懒加载 registry。"""

from nekoagent.skills.registry import (
    SkillRegistry,
    get_skill_registry,
    reset_singletons_for_test,
)

__all__ = [
    "SkillRegistry",
    "get_skill_registry",
    "reset_singletons_for_test",
]