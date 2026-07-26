"""MCP 客户端绑定 + 池化 + per-Agent 授权校验 + 工具自动发现。"""

from nekoagent.mcp.binding import (
    AgentMCPClient,
    get_mcp_client_for_agent,
    clear_client_cache,
    refresh_mcp_tool_registry,
    get_mcp_tool_registry,
    get_mcp_tools_for_agent,
    bind_mcp_tools_to_llm,
    bind_agent_tools_to_llm,
)

__all__ = [
    "AgentMCPClient",
    "get_mcp_client_for_agent",
    "clear_client_cache",
    "refresh_mcp_tool_registry",
    "get_mcp_tool_registry",
    "get_mcp_tools_for_agent",
    "bind_mcp_tools_to_llm",
    "bind_agent_tools_to_llm",
]