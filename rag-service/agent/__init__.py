"""
Agent package — OpenAI Agents SDK wrapper for RAG service.
Replaces the hand-rolled agent loop from agent_legacy.py.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Awaitable
from datetime import datetime, timezone

from openai import OpenAI, AsyncOpenAI
from agents import Agent, ModelSettings
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

logger = logging.getLogger(__name__)


# ── AgentTask (carried over from agent_legacy.py, minus cancel/inject) ──


@dataclass
class AgentTask:
    """Tracks an autonomous agent task."""

    task_id: str
    prompt: str
    status: str = "running"  # running | completed | failed
    steps: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    model_id: Optional[str] = None
    tools_called: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    error: Optional[str] = None
    source: str = "web"  # "web" | "cli"

    # Optional async callbacks for Redis persistence (fire-and-forget)
    _on_log: Optional[Callable[[str, str], Awaitable[None]]] = field(default=None, repr=False)
    _on_step: Optional[Callable[[str, Dict], Awaitable[None]]] = field(default=None, repr=False)

    MAX_LOGS = 3000

    def log(self, msg: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.logs.append(entry)
        if len(self.logs) > self.MAX_LOGS:
            self.logs = self.logs[:50] + self.logs[-(self.MAX_LOGS - 50):]
        if self._on_log:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._on_log(self.task_id, entry))
            except RuntimeError:
                pass  # No event loop (sync context)

    def add_step(self, step_type: str, content: str):
        step = {
            "type": step_type,
            "content": content[:2000],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.steps.append(step)
        if self._on_step:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._on_step(self.task_id, step))
            except RuntimeError:
                pass

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "status": self.status,
            "steps": self.steps[-50:],
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "model_id": self.model_id,
            "tools_called": self.tools_called,
            "summary": self.summary,
            "error": self.error,
            "log_count": len(self.logs),
            "source": self.source,
        }


# ── AgentServices context (passed to tools via RunContextWrapper) ──


@dataclass
class AgentServices:
    """Shared services passed to all tools via RunContextWrapper[AgentServices]."""
    web_indexer: Any  # WebIndexer
    product_indexer: Any  # Optional[ProductIndexer]
    llm_client: Any  # OpenAI
    task: Optional[AgentTask] = None


# ── System prompt (unchanged from agent_legacy.py) ──


SYSTEM_PROMPT = """You are an autonomous AI agent managing a multi-purpose RAG (Retrieval-Augmented Generation) system for an airsoft e-commerce business. You have tools to:

**Web & Content:**
1. **web_search** — Search the internet to find relevant URLs
2. **analyze_site** — Check if a URL is accessible and crawlable (detects Cloudflare, SPAs, bot protection)
3. **crawl_website** — Crawl and index a website into the RAG vector database
4. **search_indexed** — Search already-indexed content (web pages)
5. **list_collections** — View database statistics for all collections
6. **list_sources** — See what domains are already indexed
7. **delete_source** — Remove a domain's content from the index

**Product Catalog:**
8. **search_products** — Search the Shopify product catalog. REQUIRES `query` parameter (always pass a search string).

## IMPORTANT RULES — Tool Usage

**Always pass ALL required parameters.** Every tool call MUST include its required arguments:
- `search_products` REQUIRES `query` (a text string). If you want to browse by category, use the category name as the query: `search_products(query="pistol", category="pistol")`.
- `search_indexed` REQUIRES `query`.
- `crawl_website` REQUIRES `url`.
- `web_search` REQUIRES `query`.

**Before crawling, ALWAYS analyze the site first:**
1. Call `analyze_site(url)` — check the `crawlable` field and `verdict`
2. If verdict says BLOCKED or SPA DETECTED, do NOT attempt to crawl — it will fail
3. If blocked by Cloudflare, try alternative URLs or inform the user
4. If SPA detected, the crawler cannot render JavaScript — suggest a different approach

**Crawl failure recovery:**
- If a crawl returns 0 pages indexed, READ the diagnostics and suggestions carefully
- If smart_mode analysis failed, retry with `smart_mode=false`
- If 0 pages visited, the site blocks bots — do NOT retry the same URL
- If pages were visited but 0 indexed, content was below minimum length or filtered as boilerplate

**Workflow:**
1. Check `list_sources` first — don't re-crawl indexed domains
2. Use `analyze_site` before crawling to check accessibility
3. Only crawl if the site is marked `crawlable: true`
4. After crawling, use `search_indexed` to verify content was indexed
5. Report progress and findings clearly

When finished, provide a clear summary of what you accomplished."""


# ── Agent factory ──


def create_agent(vllm_base_url: str, api_key: str = "none") -> tuple:
    """Create the agent and return (agent, model_id).

    Returns:
        Tuple of (Agent, str) — the agent instance and discovered model ID.
    """
    from .tools import ALL_TOOLS

    async_client = AsyncOpenAI(
        base_url=vllm_base_url,
        api_key=api_key,
        default_headers={"X-Source-Service": "pocharlies-rag:agent"},
    )

    model_id = os.environ.get("AGENT_MODEL", "smart")

    agent = Agent(
        name="pocharlies-rag",
        instructions=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
        model=OpenAIChatCompletionsModel(model=model_id, openai_client=async_client),
        model_settings=ModelSettings(temperature=0.3, max_tokens=4096),
    )

    logger.info(f"Created agent with model: {model_id}")
    return agent, model_id
