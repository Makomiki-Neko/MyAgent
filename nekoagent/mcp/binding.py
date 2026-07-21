"""MCP 客户端：按 YAML 中 agents.<name>.mcp_servers 绑定可访问的 MCP 服务器清单。

设计：
- 启动期不预连接；首次调用工具时 lazy fetch `mcp_servers.<srv>` 配置 + 启动子进程/HTTP 客户端。
- 每个 Agent 持有独立 AgentMCPClient 实例，限制可访问服务器集合。
- 调用方仅接触 `call_tool(server, tool, args)` 与 `allowed_servers()`，无权直接持参客户端。
- `mcp` Python SDK 未安装时友好降级：AgentMCPClient.allowed_servers 返回空，所有 call_tool 抛 MCPError。
"""

from __future__ import annotations

from typing import Any

from nekoagent.config.loader import Config, get_config
from nekoagent.observability.exceptions import MCPError
from nekoagent.observability.logging import get_logger

_log = get_logger("mcp.binding")


class AgentMCPClient:
    def __init__(self, agent_name: str, cfg: Config | None = None) -> None:
        self.agent_name = agent_name
        self.cfg = cfg or get_config()
        agent_cfg = self.cfg.agents.get(agent_name)
        if agent_cfg is None:
            raise MCPError(f"未在 config.yaml 中找到 agent '{agent_name}'")
        self._allowed = list(agent_cfg.mcp_servers)
        self._clients: dict[str, Any] = {}

    def allowed_servers(self) -> list[str]:
        return list(self._allowed)

    def _ensure_client(self, server_name: str) -> Any:
        if server_name in self._clients:
            return self._clients[server_name]
        if server_name not in self._allowed:
            raise MCPError(f"server '{server_name}' 未授权给 agent '{self.agent_name}'")
        srv_cfg = self.cfg.mcp_servers.get(server_name)
        if srv_cfg is None:
            raise MCPError(f"MCP server '{server_name}' 未在 config.yaml 定义")
        try:
            if srv_cfg.transport == "stdio":
                try:
                    from mcp import ClientSession, StdioServerParameters  # type: ignore
                    from mcp.client.stdio import stdio_client  # type: ignore
                except ImportError as exc:
                    raise MCPError(
                        "未安装 `mcp` Python SDK；pip install mcp",
                        cause=exc,
                    ) from exc
                params = StdioServerParameters(
                    command=srv_cfg.command,
                    args=list(srv_cfg.args),
                    env=dict(srv_cfg.env) or None,
                )
                ctx = stdio_client(params)
                # 同步上下文管理器：进 ctx 取到 stream，再 wrap 为 ClientSession
                read_stream, write_stream = ctx.__enter__()
                session = ClientSession(read_stream, write_stream)
                session.__enter__()
                self._clients[server_name] = session
                _log.info("MCP stdio client connected: %s", server_name)
                return session
            elif srv_cfg.transport in ("http", "sse", "websocket"):
                # MVP 简化：仅 stdio；HTTP 远程 MCP 留作扩展
                raise MCPError(f"MCP transport '{srv_cfg.transport}' 暂未支持（MVP 仅 stdio）")
            else:
                raise MCPError(f"未知 MCP transport: {srv_cfg.transport}")
        except MCPError:
            raise
        except Exception as exc:
            raise MCPError(f"MCP 客户端启动失败 server={server_name}: {exc}", cause=exc) from exc

    def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
        session = self._ensure_client(server_name)
        try:
            result = session.call_tool(tool_name, arguments=arguments or {})
        except Exception as exc:
            raise MCPError(f"MCP call_tool 失败 server={server_name} tool={tool_name}: {exc}", cause=exc) from exc
        # 归一化结果为文本：MCP SDK 可能返回 CallToolResult 对象
        try:
            return _stringify_mcp_result(result)
        except Exception:
            return str(result)

    def close_all(self) -> None:
        for name, session in list(self._clients.items()):
            try:
                session.__exit__(None, None, None)
            except Exception as exc:
                _log.warning("MCP session close error server=%s: %s", name, exc)
        self._clients.clear()


def _stringify_mcp_result(result: Any) -> str:
    """尽力把 MCP CallToolResult 拍平为字符串，方便给到下游 Agent 上下文。"""

    if isinstance(result, str):
        return result
    content = getattr(result, "content", None)
    if content is None and isinstance(result, (list, tuple)):
        content = result
    if content is None:
        return str(result)
    rendered: list[str] = []
    for item in content:
        if isinstance(item, str):
            rendered.append(item)
            continue
        text_attr = getattr(item, "text", None)
        if text_attr:
            rendered.append(text_attr)
            continue
        rendered.append(str(item))
    return "\n".join(rendered)


_client_cache: dict[str, AgentMCPClient] = {}


def get_mcp_client_for_agent(agent_name: str, cfg: Config | None = None) -> AgentMCPClient:
    if agent_name in _client_cache:
        return _client_cache[agent_name]
    client = AgentMCPClient(agent_name, cfg or get_config())
    _client_cache[agent_name] = client
    return client


def clear_client_cache() -> None:
    for client in _client_cache.values():
        try:
            client.close_all()
        except Exception:
            pass
    _client_cache.clear()