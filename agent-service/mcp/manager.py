"""MCP Manager - connects to MCP servers, discovers tools, provides unified tool list."""

import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client

from config import settings

logger = logging.getLogger("agent.mcp")


class MCPServerConnection:
    """Holds connection state for a single MCP server."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.session: ClientSession | None = None
        self.tools: list[StructuredTool] = []
        self.connected = False
        self._cm = None

    async def connect(self):
        server_type = self.config.get("type", "sse")
        try:
            if server_type == "sse":
                await self._connect_sse()
            elif server_type == "stdio":
                await self._connect_stdio()
            else:
                logger.error("Unknown server type: %s for %s", server_type, self.name)
                return

            tools_result = await self.session.list_tools()
            self.tools = [
                self._wrap_tool(t.name, t.description or "", t.inputSchema or {})
                for t in tools_result.tools
            ]

            self.connected = True
            logger.info("Connected to MCP '%s': %d tools", self.name, len(self.tools))
        except Exception as e:
            logger.error("Failed to connect to MCP '%s': %s", self.name, e)
            self.connected = False

    async def _connect_sse(self):
        url = self.config["url"]
        self._cm = sse_client(url=url)
        self._read, self._write = await self._cm.__aenter__()
        self.session = ClientSession(self._read, self._write)
        await self.session.__aenter__()
        await self.session.initialize()

    async def _connect_stdio(self):
        command = self.config["command"]
        args = self.config.get("args", [])
        env = {**os.environ, **self._resolve_env(self.config.get("env", {}))}
        params = StdioServerParameters(command=command, args=args, env=env)
        self._cm = stdio_client(params)
        self._read, self._write = await self._cm.__aenter__()
        self.session = ClientSession(self._read, self._write)
        await self.session.__aenter__()
        await self.session.initialize()

    @staticmethod
    def _resolve_env(env_overrides: dict) -> dict:
        """Resolve environment variable references like ${VAR_NAME}."""
        resolved = {}
        for key, value in env_overrides.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                resolved[key] = os.environ.get(value[2:-1], "")
            else:
                resolved[key] = value
        return resolved

    def _wrap_tool(self, tool_name: str, description: str, schema: dict) -> StructuredTool:
        namespaced = f"{self.name}__{tool_name}"
        session = self.session

        async def _invoke(**kwargs: Any) -> str:
            try:
                result = await session.call_tool(tool_name, arguments=kwargs)
                parts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                    else:
                        parts.append(str(item))
                return "\n".join(parts) if parts else "(no output)"
            except Exception as e:
                return f"Error: {e}"

        return StructuredTool.from_function(
            coroutine=_invoke,
            name=namespaced,
            description=f"[{self.name}] {description}",
        )

    async def disconnect(self):
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception:
                pass
        if self._cm:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                pass
        self.connected = False
        self.tools = []


class MCPManager:
    """Manages connections to all configured MCP servers."""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or settings.mcp_servers_path
        self.servers: dict[str, MCPServerConnection] = {}

    async def start(self):
        config = self._load_config()
        for name, server_config in config.get("servers", {}).items():
            conn = MCPServerConnection(name, server_config)
            await conn.connect()
            self.servers[name] = conn
        total = sum(len(s.tools) for s in self.servers.values())
        logger.info("MCP Manager ready: %d servers, %d tools", len(self.servers), total)

    async def stop(self):
        for conn in self.servers.values():
            await conn.disconnect()
        self.servers.clear()

    def get_all_tools(self) -> list[StructuredTool]:
        tools = []
        for conn in self.servers.values():
            if conn.connected:
                tools.extend(conn.tools)
        return tools

    def get_server_status(self) -> list[dict]:
        return [
            {
                "name": name,
                "type": conn.config.get("type"),
                "connected": conn.connected,
                "tools_count": len(conn.tools),
                "description": conn.config.get("description", ""),
            }
            for name, conn in self.servers.items()
        ]

    async def reload(self):
        await self.stop()
        await self.start()

    async def add_server(self, name: str, config: dict):
        conn = MCPServerConnection(name, config)
        await conn.connect()
        self.servers[name] = conn
        self._save_config()

    async def remove_server(self, name: str):
        if name in self.servers:
            await self.servers[name].disconnect()
            del self.servers[name]
            self._save_config()

    def _load_config(self) -> dict:
        path = Path(self.config_path)
        if path.exists():
            return json.loads(path.read_text())
        return {"servers": {}}

    def _save_config(self):
        config = {"servers": {name: conn.config for name, conn in self.servers.items()}}
        Path(self.config_path).write_text(json.dumps(config, indent=2))
