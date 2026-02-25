"""Supervisor Agent - LangGraph ReAct StateGraph with MCP tools."""

import logging
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from config import settings

logger = logging.getLogger("agent.supervisor")

SYSTEM_PROMPT = """\
You are the Pocharlies Agent Orchestrator - a production-grade AI assistant for \
Pocharlies, an airsoft e-commerce business.

## Your capabilities
You have access to tools from connected MCP servers. These tools let you:
- Search and manage the RAG knowledge base (web pages, products, competitors)
- Crawl websites and index their content
- Search the Shopify product catalog
- Crawl and analyze competitor websites
- Translate content across 20+ languages
- Manage vector collections in Qdrant

## How you operate
- You receive instructions via chat, scheduled triggers [SCHEDULED:...], or events [EVENT:...]
- For simple queries, use tools directly and respond
- For complex multi-step tasks, break them down and execute step by step
- Always report what you did and what you found
- If an action is destructive or expensive (large crawls, bulk operations), \
explain what you plan to do and ask for confirmation before proceeding

## Formatting
- Be concise and direct
- Use bullet points for lists
- Include relevant numbers (pages crawled, products found, etc.)
"""


def create_supervisor(
    checkpointer: AsyncPostgresSaver,
    tools: list[StructuredTool],
) -> StateGraph:
    """Build and compile the supervisor ReAct agent graph."""
    model = ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model or "default",
        temperature=settings.llm_temperature,
    )

    model_with_tools = model.bind_tools(tools) if tools else model

    def supervisor_node(state: MessagesState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "__end__"

    builder = StateGraph(MessagesState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", should_continue, ["tools", "__end__"])
    builder.add_edge("tools", "supervisor")

    graph = builder.compile(checkpointer=checkpointer)
    logger.info("Supervisor graph compiled with %d tools", len(tools))
    return graph
