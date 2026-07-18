import importlib.util
import inspect
import logging
import os
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def load_skills(skills_path: str = "app/skills") -> dict[str, Callable]:
    skills = {}
    path = Path(skills_path)
    if not path.exists():
        logger.info(f"Skills directory '{skills_path}' does not exist, skipping")
        return skills

    for file in path.glob("*.py"):
        if file.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(file.stem, file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name, obj in inspect.getmembers(module):
                    if callable(obj) and hasattr(obj, "_skill_meta"):
                        skills[name] = obj
                        logger.info(f"Loaded skill '{name}' from {file.name}")

                if not any(callable(obj) and hasattr(obj, "_skill_meta") for obj in module.__dict__.values()):
                    for name, obj in inspect.getmembers(module, inspect.isfunction):
                        if not name.startswith("_"):
                            skills[name] = obj
                            logger.info(f"Loaded function skill '{name}' from {file.name}")

        except Exception as e:
            logger.warning(f"Failed to load skill from {file}: {e}")

    logger.info(f"Loaded {len(skills)} skills from '{skills_path}'")
    return skills


def skill_meta(name: str = "", description: str = ""):
    def decorator(func):
        func._skill_meta = {"name": name or func.__name__, "description": description or func.__doc__ or ""}
        return func
    return decorator
