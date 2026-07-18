import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain.agents import create_agent
from app.config import get_agent_llm_config, get_agent_mcp_servers
from app.mcp.client import MCPManager
from app.skills.loader import load_skills

logger = logging.getLogger(__name__)


def _build_chat_model(agent_name: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    cfg = get_agent_llm_config(agent_name)
    return ChatOpenAI(
        model=cfg["model"],
        temperature=cfg.get("temperature", 0.7),
        max_tokens=cfg.get("max_tokens", 4096),
        api_key=cfg.get("api_key", "sk-placeholder"),
        base_url=cfg.get("base_url", "http://localhost:11434/v1"),
    )


class AgentRuntime:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.model = _build_chat_model(agent_name)
        self.mcp: MCPManager | None = None
        self.skills: dict[str, Any] = {}
        self._tools: list[BaseTool] = []

    def init_mcp(self, server_configs: list[dict]):
        if server_configs:
            self.mcp = MCPManager(server_configs)
            logger.info(f"Agent '{self.agent_name}' MCP initialized with {len(server_configs)} server(s)")

    def init_skills(self, skills_path: str = "app/skills"):
        self.skills = load_skills(skills_path)
        if self.skills:
            logger.info(f"Agent '{self.agent_name}' loaded {len(self.skills)} skill(s)")

    def get_mcp_tools(self) -> list[BaseTool]:
        tools = []
        if not self.mcp:
            return tools

        for t in self.mcp.get_all_tools():
            name = t.get("name", "unknown")
            description = t.get("description", "")
            input_schema = t.get("input_schema", {})

            async def _make_async_call(tool_name=name, schema=input_schema):
                def tool_func(**kwargs):
                    return self.mcp.call_tool(tool_name, kwargs)
                return tool_func

            import asyncio
            tool_func = asyncio.run(_make_async_call(name))
            try:
                from langchain_core.tools import tool as lc_tool
                wrapped = lc_tool(description=description)(tool_func)
                wrapped.name = name
                tools.append(wrapped)
            except Exception as e:
                logger.debug(f"Skipping MCP tool '{name}': {e}")

        return tools

    def build_agent(self, system_prompt: str, tools: list | None = None):
        all_tools = tools or []
        all_tools.extend(self.get_mcp_tools())
        agent = create_agent(
            model=self.model,
            tools=all_tools,
            system_prompt=system_prompt,
        )
        return agent

    def chat(self, messages: list[dict], system_prompt: str = "", tools: list | None = None) -> str:
        agent = self.build_agent(system_prompt, tools)
        langchain_messages = []
        if system_prompt:
            langchain_messages.append(SystemMessage(content=system_prompt))
        for m in messages:
            if m["role"] == "user":
                langchain_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                langchain_messages.append(AIMessage(content=m["content"]))
            elif m["role"] == "system":
                langchain_messages.append(SystemMessage(content=m["content"]))

        try:
            result = agent.invoke({"messages": langchain_messages})
            last_msg = result["messages"][-1]
            if hasattr(last_msg, "content"):
                return last_msg.content or ""
            return str(last_msg)
        except Exception as e:
            logger.error(f"Agent '{self.agent_name}' chat failed: {e}")
            return f"[Agent 调用异常: {str(e)}]"

    def cleanup(self):
        if self.mcp:
            self.mcp.close_all()
