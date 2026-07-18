import json
import logging
import subprocess
import threading
from typing import Any

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self, name: str = "default", transport: str = "stdio",
                 command: str = "", args: list[str] | None = None,
                 env: dict[str, str] | None = None):
        self.name = name
        self.transport = transport
        self.command = command
        self.args = args or []
        self.env = env or {}
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._tools: list[dict] = []

    def connect(self):
        if not self.command:
            logger.info(f"MCP client '{self.name}' has no command configured, skipping")
            return
        try:
            env = {**self.env}
            self._process = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**env},
                text=True,
            )
            logger.info(f"MCP client '{self.name}' connected via {self.transport}")
            self._discover_tools()
        except Exception as e:
            logger.warning(f"MCP client '{self.name}' connection failed: {e}")

    def _discover_tools(self):
        tools = []
        try:
            tools_response = self._send_request("list_tools", {})
            if tools_response and "tools" in tools_response:
                tools = tools_response["tools"]
        except Exception as e:
            logger.debug(f"MCP tool discovery failed: {e}")
        self._tools = tools
        logger.info(f"MCP '{self.name}' discovered {len(tools)} tools")

    def _send_request(self, method: str, params: dict) -> dict | None:
        if not self._process:
            return None
        req = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        with self._lock:
            try:
                self._process.stdin.write(json.dumps(req) + "\n")
                self._process.stdin.flush()
                line = self._process.stdout.readline()
                if line:
                    return json.loads(line.strip())
            except Exception as e:
                logger.error(f"MCP request failed ({method}): {e}")
        return None

    def call_tool(self, tool_name: str, arguments: dict) -> Any:
        logger.info(f"MCP calling tool '{tool_name}' with args: {arguments}")
        resp = self._send_request("call_tool", {"name": tool_name, "arguments": arguments})
        if resp and "result" in resp:
            return resp["result"]
        if resp and "error" in resp:
            return {"error": resp["error"]}
        return {"error": f"Tool '{tool_name}' call failed"}

    def list_tools(self) -> list[dict]:
        return self._tools

    def close(self):
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None
            logger.info(f"MCP client '{self.name}' closed")


class MCPManager:
    def __init__(self, server_configs: list[dict]):
        self.clients: dict[str, MCPClient] = {}
        for cfg in server_configs:
            name = cfg.get("name", "default")
            client = MCPClient(
                name=name,
                transport=cfg.get("transport", "stdio"),
                command=cfg.get("command", ""),
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
            )
            client.connect()
            self.clients[name] = client

    def get_all_tools(self) -> list[dict]:
        tools = []
        for client in self.clients.values():
            tools.extend(client.list_tools())
        return tools

    def call_tool(self, tool_name: str, arguments: dict) -> Any:
        for client in self.clients.values():
            for t in client.list_tools():
                if t.get("name") == tool_name:
                    return client.call_tool(tool_name, arguments)
        return {"error": f"Tool '{tool_name}' not found in any MCP server"}

    def close_all(self):
        for client in self.clients.values():
            client.close()
