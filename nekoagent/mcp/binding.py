"""MCP 客户端：按 YAML 中 agents.<name>.mcp_servers 绑定可访问的 MCP 服务器清单。

设计：
- stdio 传输：使用同步 subprocess + JSON-RPC 实现。
- SSE 传输：使用 httpx 同步客户端 + 后台 SSE 读取线程实现。
- 每个 Agent 持有独立 AgentMCPClient 实例，限制可访问服务器集合。
- 调用方仅接触 `call_tool(server, tool, args)` 与 `allowed_servers()`，无权直接持参客户端。

工具自动发现：
- refresh_mcp_tool_registry() 在配置加载/热更新时被调用，连接所有已配置的 MCP 服务器并调用 tools/list 获取工具清单。
- get_mcp_tools_for_agent() / bind_agent_tools_to_llm() 供 Agent 层将 MCP 工具注入 LLM 的函数调用列表。
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import urllib.parse
from typing import Any

from nekoagent.config.loader import Config, get_config
from nekoagent.observability.exceptions import MCPError
from nekoagent.observability.logging import get_logger

_log = get_logger("mcp.binding")

# ──────────────────────────────────────────
# 全局 MCP 工具注册表
# 结构: {server_name: {tool_name: ToolDef}}
# ToolDef = {"name", "description", "inputSchema", "server"}
# ──────────────────────────────────────────
_mcp_tool_registry: dict[str, dict[str, dict]] = {}
_tool_registry_initialized: bool = False


# ─────────── SSE MCP Session ───────────


class SyncSseSession:
    """基于 httpx + 后台 SSE 线程的同步 SSE MCP 会话。

    MCP over SSE 协议：
    - GET {base_url}/sse → 服务端推送事件流（含 session_id / endpoint）
    - POST {base_url}/messages?session_id=xxx → 发送 JSON-RPC 请求
    - 响应通过 SSE 事件流返回

    使用 per-request Event + 条件变量的方式避免竞态条件。
    """

    def __init__(self, server_name: str, base_url: str) -> None:
        self._server_name = server_name
        self._base_url = base_url.rstrip("/")
        self._rpc_id = 0
        self._responses: dict[int, Any] = {}
        self._pending_events: dict[int, threading.Event] = {}
        self._lock = threading.Lock()
        self._session_id: str | None = None
        self._msg_endpoint: str | None = None
        self._error: str | None = None
        self._endpoint_event = threading.Event()

        # 启动 SSE 读取线程
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._sse_reader, daemon=True)
        self._thread.start()

        # 等待 session_id（最多 5 秒）
        if not self._endpoint_event.wait(timeout=5):
            self._stop_event.set()
            raise MCPError(
                f"MCP SSE server '{server_name}' 连接超时 (5s), "
                f"请确认 {base_url}/sse 可访问"
            )
        if self._error:
            raise MCPError(
                f"MCP SSE server '{server_name}' 连接失败: {self._error}"
            )

        # MCP 协议初始化握手
        self._initialize()

    def _sse_reader(self) -> None:
        """后台线程：读取 SSE 事件流。"""
        import httpx

        try:
            with httpx.Client(base_url=self._base_url, timeout=None) as client:
                with client.stream("GET", "/sse") as resp:
                    if resp.status_code != 200:
                        self._error = f"SSE 连接返回 {resp.status_code}"
                        self._endpoint_event.set()
                        return

                    event_type = ""
                    data_lines: list[str] = []
                    for line in resp.iter_lines():
                        if self._stop_event.is_set():
                            break
                        line = line.strip()
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[5:].strip())
                        elif line == "":
                            if data_lines:
                                self._handle_sse_event(event_type, "".join(data_lines))
                            event_type = ""
                            data_lines = []
        except Exception as exc:
            if not self._stop_event.is_set():
                self._error = str(exc)
                self._endpoint_event.set()

    def _handle_sse_event(self, event_type: str, data: str) -> None:
        if event_type == "endpoint":
            self._msg_endpoint = data
            if "?" in data:
                qs = data.split("?", 1)[1]
                for pair in qs.split("&"):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        if k == "session_id":
                            self._session_id = v
            self._endpoint_event.set()
            _log.info(
                "MCP SSE 会话已建立 server=%s session_id=%s endpoint=%s",
                self._server_name, self._session_id, self._msg_endpoint,
            )
            return

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return

        rpc_id = parsed.get("id")
        if rpc_id is not None:
            rpc_id = int(rpc_id)
            with self._lock:
                self._responses[rpc_id] = parsed
                event = self._pending_events.pop(rpc_id, None)
                if event:
                    event.set()

    def _send_request(self, method: str, params: dict | None = None) -> dict:
        import httpx

        if not self._msg_endpoint:
            raise MCPError(f"SSE session '{self._server_name}' 未收到 endpoint")

        self._rpc_id += 1
        rpc_id = self._rpc_id
        req = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": method,
        }
        if params:
            req["params"] = params

        # 注册 per-request event（必须在 POST 之前，避免竞态）
        event = threading.Event()
        with self._lock:
            self._pending_events[rpc_id] = event

        msg_url = f"{self._msg_endpoint}"
        _log.debug("MCP SSE >> %s", json.dumps(req, ensure_ascii=False)[:200])

        try:
            with httpx.Client(base_url=self._base_url, timeout=10) as client:
                resp = client.post(msg_url, json=req)
                resp.raise_for_status()
        except Exception as exc:
            with self._lock:
                self._pending_events.pop(rpc_id, None)
            raise MCPError(
                f"MCP SSE POST 失败 server={self._server_name}: {exc}",
                cause=exc,
            ) from exc

        # 等待该请求的唯一响应
        if not event.wait(timeout=10):
            with self._lock:
                self._pending_events.pop(rpc_id, None)
            raise MCPError(f"MCP server '{self._server_name}' 请求超时 (10s)")

        with self._lock:
            result = self._responses.pop(rpc_id, None)

        if result is None:
            raise MCPError(f"MCP server '{self._server_name}' 未返回有效响应")

        _log.debug("MCP SSE << %s", json.dumps(result, ensure_ascii=False)[:500])

        if "error" in result:
            err = result["error"]
            raise MCPError(
                f"MCP server '{self._server_name}' 错误: [{err.get('code')}] {err.get('message')}"
            )
        return result.get("result", {})

    def _initialize(self) -> None:
        """MCP 协议初始化握手（通过 SSE POST 通道）。"""
        self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "nekoagent", "version": "0.1.0"},
        })
        _log.info("MCP SSE 初始化握手完成: %s", self._server_name)

    def list_tools(self) -> list[dict]:
        result = self._send_request("tools/list")
        raw = result.get("tools", [])
        if not isinstance(raw, list):
            return []
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "inputSchema": t.get("inputSchema", {}),
            }
            for t in raw
        ]

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        result = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments or {},
        })
        return _SyncCallToolResult(result)

    def close(self) -> None:
        self._stop_event.set()
        _log.info("MCP SSE 会话已关闭: %s", self._server_name)


# ─────────── 同步 stdio MCP Session ───────────


class SyncStdioSession:
    """基于 subprocess + JSON-RPC 的同步 MCP stdio 会话。

    与 mcp SDK 的 ClientSession 接口兼容（list_tools / call_tool），
    但完全同步，无 anyio / asyncio 依赖。
    """

    def __init__(self, server_name: str, command: str, args: list[str], env: dict[str, str] | None) -> None:
        self._server_name = server_name
        self._proc = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )
        self._rpc_id = 0
        # 建立连接后先初始化握手
        self._initialize()

    def _send_request(self, method: str, params: dict | None = None) -> dict:
        self._rpc_id += 1
        req = {
            "jsonrpc": "2.0",
            "id": self._rpc_id,
            "method": method,
        }
        if params:
            req["params"] = params
        line = json.dumps(req, ensure_ascii=False)
        _log.debug("MCP >> %s", line[:200])
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            stderr = self._collect_stderr()
            raise MCPError(
                f"MCP server '{self._server_name}' 进程断开了连接: {exc}. stderr: {stderr}",
                cause=exc,
            ) from exc

        # 读响应 — 逐行读到完整 JSON
        resp_line = self._proc.stdout.readline()
        if not resp_line:
            stderr = self._collect_stderr()
            raise MCPError(
                f"MCP server '{self._server_name}' 无响应 (stdout 已关闭). stderr: {stderr}"
            )
        try:
            resp = json.loads(resp_line.strip())
        except json.JSONDecodeError as exc:
            raise MCPError(
                f"MCP server '{self._server_name}' 返回非法 JSON: {resp_line[:200]}",
                cause=exc,
            ) from exc

        _log.debug("MCP << %s", json.dumps(resp, ensure_ascii=False)[:500])

        if "error" in resp:
            err = resp["error"]
            raise MCPError(f"MCP server '{self._server_name}' 错误: [{err.get('code')}] {err.get('message')}")
        return resp.get("result", {})

    def _collect_stderr(self) -> str:
        try:
            return (self._proc.stderr.read() or "")[:500]
        except Exception:
            return ""

    def _initialize(self) -> None:
        """MCP 初始化握手。"""
        self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "nekoagent", "version": "0.1.0"},
        })
        _log.info("MCP 初始化握手完成: %s", self._server_name)

    def list_tools(self) -> list[dict]:
        """返回工具定义列表 [{"name":..., "description":..., "inputSchema":...}, ...]"""
        result = self._send_request("tools/list")
        raw = result.get("tools", [])
        if not isinstance(raw, list):
            return []
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "inputSchema": t.get("inputSchema", {}),
            }
            for t in raw
        ]

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        """调用 MCP 工具并返回原生结果对象（与 mcp.ClientSession.call_tool 返回值类似）。"""
        result = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments or {},
        })
        return _SyncCallToolResult(result)

    def close(self) -> None:
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
        _log.info("MCP 会话已关闭: %s", self._server_name)


class _SyncCallToolResult:
    """模拟 mcp SDK 的 CallToolResult，兼容 _stringify_mcp_result。"""

    def __init__(self, raw: dict) -> None:
        self._raw = raw
        self.content = raw.get("content", [])

    def __str__(self) -> str:
        return json.dumps(self._raw, ensure_ascii=False)


# ─────────── 客户端类 ───────────


class AgentMCPClient:
    def __init__(self, agent_name: str, cfg: Config | None = None) -> None:
        self.agent_name = agent_name
        self.cfg = cfg or get_config()
        agent_cfg = self.cfg.agents.get(agent_name)
        if agent_cfg is None:
            raise MCPError(f"未在 config.yaml 中找到 agent '{agent_name}'")
        self._allowed = list(agent_cfg.mcp_servers)
        self._clients: dict[str, SyncStdioSession] = {}

    def allowed_servers(self) -> list[str]:
        return list(self._allowed)

    def _ensure_client(self, server_name: str) -> SyncStdioSession:
        if server_name in self._clients:
            return self._clients[server_name]
        if server_name not in self._allowed:
            raise MCPError(f"server '{server_name}' 未授权给 agent '{self.agent_name}'")
        srv_cfg = self.cfg.mcp_servers.get(server_name)
        if srv_cfg is None:
            raise MCPError(f"MCP server '{server_name}' 未在 config.yaml 定义")
        return _connect_server(server_name, srv_cfg, self._clients)

    def list_tools(self, server_name: str) -> list[dict]:
        """调用 MCP 服务器的 tools/list，返回工具定义列表。"""
        session = self._ensure_client(server_name)
        try:
            return session.list_tools()
        except Exception as exc:
            raise MCPError(f"list_tools 失败 server={server_name}: {exc}", cause=exc) from exc

    def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
        session = self._ensure_client(server_name)
        try:
            result = session.call_tool(tool_name, arguments=arguments or {})
        except Exception as exc:
            raise MCPError(f"MCP call_tool 失败 server={server_name} tool={tool_name}: {exc}", cause=exc) from exc
        try:
            return _stringify_mcp_result(result)
        except Exception:
            return str(result)

    def close_all(self) -> None:
        for name, session in list(self._clients.items()):
            try:
                session.close()
            except Exception as exc:
                _log.warning("MCP session close error server=%s: %s", name, exc)
        self._clients.clear()


# ─────────── 底层连接（不依赖 agent 权限） ───────────


def _connect_server(server_name: str, srv_cfg: Any, client_cache: dict[str, Any]) -> Any:
    """建立到 MCP 服务器的连接并缓存会话。"""
    try:
        if srv_cfg.transport == "stdio":
            if not srv_cfg.command:
                raise MCPError(f"MCP server '{server_name}' 的 stdio 配置缺少 command")
            session = SyncStdioSession(
                server_name=server_name,
                command=srv_cfg.command,
                args=list(srv_cfg.args),
                env=dict(srv_cfg.env) or None,
            )
            client_cache[server_name] = session
            _log.info("MCP stdio 同步客户端已连接: %s", server_name)
            return session
        elif srv_cfg.transport in ("sse", "http"):
            base_url = srv_cfg.url or f"http://{srv_cfg.host or '127.0.0.1'}:{srv_cfg.port or 8002}"
            session = SyncSseSession(server_name=server_name, base_url=base_url)
            client_cache[server_name] = session
            _log.info("MCP SSE 客户端已连接: %s (%s)", server_name, base_url)
            return session
        else:
            raise MCPError(f"未知 MCP transport: {srv_cfg.transport}")
    except MCPError:
        raise
    except Exception as exc:
        raise MCPError(f"MCP 客户端启动失败 server={server_name}: {exc}", cause=exc) from exc


# ─────────── 工具注册表 ───────────


def refresh_mcp_tool_registry(cfg: Config | None = None) -> dict[str, dict[str, dict]]:
    """连接所有已配置的 MCP 服务器，发现工具并更新全局注册表。

    在配置加载（load_config）和热更新（reload_config）时自动调用。
    """
    global _mcp_tool_registry, _tool_registry_initialized

    cfg = cfg or get_config()

    if not cfg.mcp_servers:
        _log.info("MCP 工具发现跳过：未配置任何 MCP 服务器")
        _mcp_tool_registry = {}
        _tool_registry_initialized = True
        return {}

    _log.info("MCP 工具发现开始，共 %d 个服务器待扫描", len(cfg.mcp_servers))

    discovered: dict[str, dict[str, dict]] = {}
    errors: list[str] = []

    for server_name in cfg.mcp_servers:
        srv_cfg = cfg.mcp_servers[server_name]
        local_cache: dict[str, Any] = {}
        try:
            session = _connect_server(server_name, srv_cfg, local_cache)
            raw_tools = session.list_tools()
            tools = {}
            for t in raw_tools:
                raw_name = t["name"]
                safe_name = _sanitize_tool_name(raw_name)
                tools[safe_name] = {
                    "name": safe_name,
                    "description": t.get("description", ""),
                    "inputSchema": t.get("inputSchema", {}),
                    "server": server_name,
                }
            discovered[server_name] = tools
            tool_names = list(tools.keys())
            _log.info(
                "MCP 服务器 '%s' 工具发现完成: %d 个工具 %s",
                server_name, len(tools), tool_names,
            )
        except Exception as exc:
            errors.append(f"{server_name}: {exc}")
            _log.warning("MCP 工具发现失败 server=%s: %s", server_name, exc)
        finally:
            for s in local_cache.values():
                try:
                    s.close()
                except Exception:
                    pass

    _mcp_tool_registry = discovered
    _tool_registry_initialized = True

    total_tools = sum(len(v) for v in discovered.values())
    _log.info(
        "MCP 工具发现结束: %d/%d 服务器成功, 共 %d 个工具",
        len(discovered), len(cfg.mcp_servers), total_tools,
    )

    if errors:
        _log.warning("MCP 工具发现部分失败: %s", "; ".join(errors))
    return discovered


def get_mcp_tool_registry() -> dict[str, dict[str, dict]]:
    """返回全局 MCP 工具注册表（只读）。"""
    return _mcp_tool_registry


def get_mcp_tools_for_agent(agent_name: str, cfg: Config | None = None) -> list[Any]:
    """返回指定 Agent 可调用的 MCP LangChain 工具列表。

    返回值可直接用于 llm.bind_tools(tools)。
    """
    cfg = cfg or get_config()
    agent_cfg = cfg.agents.get(agent_name)
    if agent_cfg is None:
        _log.debug("get_mcp_tools_for_agent: agent '%s' 不存在", agent_name)
        return []
    allowed = set(agent_cfg.mcp_servers)

    if not _tool_registry_initialized:
        _log.info("MCP 注册表尚未初始化，跳过工具获取 agent=%s", agent_name)
        return []

    tools: list[Any] = []
    for server_name, tool_map in _mcp_tool_registry.items():
        if server_name not in allowed:
            _log.debug("服务器 %s 未授权给 %s，跳过", server_name, agent_name)
            continue
        for tool_name, tool_def in tool_map.items():
            tool = _build_langchain_tool(server_name, tool_name, tool_def)
            if tool is not None:
                tools.append(tool)

    _log.info(
        "Agent '%s' 可调用 %d 个 MCP 工具（来自 %d 个服务器）",
        agent_name, len(tools),
        sum(1 for s in _mcp_tool_registry if s in allowed),
    )
    return tools


def bind_mcp_tools_to_llm(llm, agent_name: str = "main_agent", cfg: Config | None = None):
    """将 Agent 可见的 MCP 工具注入到 LLM 的 tool calling 列表中。

    警告: 如果 llm 已绑定过其他工具（如 RAG），此调用会覆盖而非合并。
    建议优先使用 bind_agent_tools_to_llm() 一次性绑定 RAG + MCP。
    """
    tools = get_mcp_tools_for_agent(agent_name, cfg)
    if not tools:
        _log.debug("bind_mcp_tools_to_llm: agent '%s' 无 MCP 工具可绑定", agent_name)
        return llm
    try:
        bound = llm.bind_tools(tools)
        _log.info(
            "MCP 工具已绑定到 LLM agent=%s, 工具列表: %s",
            agent_name, [t.name for t in tools],
        )
        return bound
    except Exception as exc:
        _log.warning("MCP 工具绑定失败 agent=%s: %s", agent_name, exc)
        return llm


def bind_agent_tools_to_llm(llm, agent_name: str = "main_agent", cfg: Config | None = None):
    """一次性将 RAG 和 MCP 工具合并绑定到 LLM，避免先后调用 bind_tools 造成的覆盖问题。

    这是推荐的入口函数，替代分别调用 bind_rag_tools_to_llm + bind_mcp_tools_to_llm。
    """
    from nekoagent.rag_tools import get_rag_tools_for_agent

    rag_tools = get_rag_tools_for_agent(agent_name)
    mcp_tools = get_mcp_tools_for_agent(agent_name, cfg)

    all_tools = list(rag_tools) + list(mcp_tools)
    if not all_tools:
        _log.debug("bind_agent_tools_to_llm: agent '%s' 无任何工具可绑定", agent_name)
        return llm

    try:
        bound = llm.bind_tools(all_tools)
        _log.info(
            "Agent '%s' 全部工具绑定完成: %d 个 (RAG %d + MCP %d), 列表: %s",
            agent_name, len(all_tools), len(rag_tools), len(mcp_tools),
            [t.name for t in all_tools],
        )
        return bound
    except Exception as exc:
        _log.warning("Agent '%s' 工具绑定失败: %s", agent_name, exc)
        return llm


# ─────────── 工具名净化（OpenAI 要求 ^[a-zA-Z0-9_-]+$） ───────────


def _sanitize_tool_name(name: str) -> str:
    """将工具名中不合规字符替换为下划线；若结果为空则回退为 hash。"""
    import hashlib, re
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    sanitized = sanitized.strip('_')
    if not sanitized or len(sanitized) < 2:
        suffix = hashlib.md5(name.encode('utf-8')).hexdigest()[:8]
        sanitized = f"tool_{suffix}"
    if sanitized != name:
        _log.info("工具名 '%s' 已自动修正为 '%s'", name, sanitized)
    return sanitized


# ─────────── LangChain 工具构造 ───────────


def _build_langchain_tool(server_name: str, tool_name: str, tool_def: dict) -> Any:
    """将 MCP 工具定义为 LangChain StructuredTool。"""
    tool_name = _sanitize_tool_name(tool_name)
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        _log.warning("langchain_core 未安装，无法构建 MCP LangChain 工具")
        return None

    input_schema = tool_def.get("inputSchema", {})
    description = tool_def.get("description", "") or f"MCP tool: {server_name}.{tool_name}"

    args_schema = _json_schema_to_pydantic(tool_name, input_schema)
    param_names = list(input_schema.get("properties", {}).keys())

    if args_schema is None:

        def _call(**kwargs: Any) -> str:
            return _do_call_mcp_tool(server_name, tool_name, kwargs)

        tool = StructuredTool.from_function(
            func=_call,
            name=tool_name,
            description=description,
        )
    else:

        def _typed_call(**kwargs: Any) -> str:
            return _do_call_mcp_tool(server_name, tool_name, kwargs)

        tool = StructuredTool.from_function(
            func=_typed_call,
            name=tool_name,
            description=description,
            args_schema=args_schema,
        )

    _log.debug("构建 MCP 工具 '%s' (server=%s, params=%s)", tool_name, server_name, param_names)
    return tool


def _do_call_mcp_tool(server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
    """实际的 MCP 工具调用，由 LangChain tool 函数触发。"""
    client = _get_or_create_discovery_client()
    try:
        return client.call_tool(server_name, tool_name, arguments)
    except Exception as exc:
        return f"[MCP 错误] {exc}"


def _get_or_create_discovery_client() -> AgentMCPClient:
    cfg = get_config()
    return AgentMCPClient("main_agent", cfg)


def _json_schema_to_pydantic(tool_name: str, schema: dict) -> Any | None:
    """将 JSON Schema 转换为 Pydantic 模型，用于 StructuredTool.args_schema。"""
    from pydantic import BaseModel, Field, create_model

    properties = schema.get("properties", {})
    if not properties:
        return None
    required = set(schema.get("required", []))

    type_map = {
        "string": (str, ...),
        "integer": (int, ...),
        "number": (float, ...),
        "boolean": (bool, ...),
        "array": (list, ...),
        "object": (dict, ...),
    }

    fields = {}
    for prop_name, prop in properties.items():
        json_type = prop.get("type", "string")
        py_type = type_map.get(json_type, (str, ...))[0]
        default = ... if prop_name in required else None
        field_kwargs: dict[str, Any] = {}
        desc = prop.get("description", "")
        if desc:
            field_kwargs["description"] = desc
        if json_type == "array":
            items = prop.get("items", {})
            item_type = type_map.get(items.get("type", "string"), (str, ...))[0]
            py_type = list[item_type]
            field_kwargs["description"] = desc or f"List of {item_type.__name__}"
        if py_type is list:
            py_type = list[str]
        fields[prop_name] = (py_type, Field(default=default, **field_kwargs))

    return create_model(f"{tool_name}_args", **fields)


# ─────────── 结果序列化 ───────────


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
        if isinstance(item, dict):
            text_attr = item.get("text", "")
            if text_attr:
                rendered.append(text_attr)
                continue
        rendered.append(str(item))
    return "\n".join(rendered)


# ─────────── 全局缓存 ───────────

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
