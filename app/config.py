import yaml
from pathlib import Path
from typing import Optional


_CONFIG: Optional[dict] = None


def load_config(path: Optional[str] = None) -> dict:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get_agent_llm_config(agent_name: str) -> dict:
    cfg = load_config()
    return cfg["agents"][agent_name]["llm"]


def get_agent_mcp_servers(agent_name: str) -> list[dict]:
    cfg = load_config()
    servers = cfg["agents"][agent_name].get("mcp_servers", [])
    resolved = []
    for s in servers:
        if isinstance(s, str) and s in cfg.get("mcp_servers", {}):
            resolved.append(cfg["mcp_servers"][s])
        else:
            resolved.append(s)
    return resolved


def get_agent_config(agent_name: str) -> dict:
    cfg = load_config()
    return cfg["agents"][agent_name]


def get_memory_config() -> dict:
    cfg = load_config()
    return cfg["memory"]


def get_task_config() -> dict:
    cfg = load_config()
    return cfg["task"]


def get_logging_config() -> dict:
    cfg = load_config()
    return cfg["logging"]


def get_skills_config() -> dict:
    cfg = load_config()
    return cfg.get("skills", {"enabled": False, "path": "app/skills", "auto_load": False})


def get_mcp_server_config(name: str = "default") -> Optional[dict]:
    cfg = load_config()
    return cfg.get("mcp_servers", {}).get(name)
