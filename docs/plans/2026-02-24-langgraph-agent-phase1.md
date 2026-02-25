# LangGraph Agent Orchestrator — Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production-grade LangGraph agent-service with supervisor agent, dynamic MCP tool loading, Postgres persistence, REST API with Swagger, WebSocket chat, and a chat UI.

**Architecture:** A new `agent-service` Docker container running FastAPI + LangGraph. The supervisor agent (ReAct StateGraph) reasons over dynamically-loaded MCP tools. AsyncPostgresSaver provides durable checkpointing. A chat UI served as static files allows real-time interaction over WebSocket.

**Tech Stack:** LangGraph, FastAPI, PostgreSQL 16, Redis, langchain-openai, MCP Python SDK, APScheduler (Postgres job store), Pydantic v2, vanilla HTML/JS/CSS.

---

### Task 1: Scaffold the agent-service project

**Files:**
- Create: `agent-service/requirements.txt`
- Create: `agent-service/config.py`
- Create: `agent-service/main.py`
- Create: `agent-service/Dockerfile`

**Step 1: Create requirements.txt**

```txt
# LangGraph core + Postgres persistence
langgraph>=0.4.0
langgraph-checkpoint-postgres>=2.0.0

# LangChain LLM integration (OpenAI-compatible via LiteLLM)
langchain-openai>=0.3.0
langchain-core>=0.3.0

# MCP client
mcp>=1.8.0

# API layer
fastapi>=0.115.0
uvicorn>=0.34.0
websockets>=14.0

# Database
asyncpg>=0.30.0
psycopg[binary]>=3.2.0
sqlalchemy[asyncio]>=2.0.0

# Scheduler
apscheduler>=4.0.0a5

# Redis
redis[hiredis]>=5.0.0

# Utilities
pydantic>=2.5.0
pydantic-settings>=2.0.0
httpx>=0.27.0
python-dotenv>=1.0.0
```

**Step 2: Create config.py**

```python
# agent-service/config.py
"""Application configuration from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://agent:agent@localhost:5432/langgraph"
    database_url_sync: str = ""
    redis_url: str = "redis://localhost:6379/1"
    rag_service_url: str = "http://localhost:5000"
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "none"
    llm_model: str = ""
    llm_temperature: float = 0.2
    mcp_servers_path: str = "mcp/mcp_servers.json"
    host: str = "0.0.0.0"
    port: int = 8100

    class Config:
        env_file = ".env"
        extra = "ignore"

    def model_post_init(self, __context):
        if not self.database_url_sync:
            self.database_url_sync = self.database_url.replace(
                "postgresql+asyncpg://", "postgresql://"
            )


settings = Settings()
```

**Step 3: Create main.py (minimal shell)**

```python
# agent-service/main.py
"""Pocharlies Agent Orchestrator - FastAPI entrypoint."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Agent service starting...")
    yield
    logger.info("Agent service shutting down...")


app = FastAPI(
    title="Pocharlies Agent Orchestrator",
    description=(
        "Production-grade LangGraph agent for Pocharlies airsoft e-commerce.\n\n"
        "## Capabilities\n"
        "- **Chat**: Converse with the agent to give instructions\n"
        "- **Workflows**: Automated multi-step operations\n"
        "- **Schedules**: Cron jobs for recurring tasks\n"
        "- **MCP**: Dynamic tool loading from any MCP server\n"
        "- **Events**: React to system events in real-time\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/api/health", tags=["System"])
async def health():
    return {"status": "ok", "service": "agent-orchestrator"}
```

**Step 4: Create Dockerfile**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8100
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
```

**Step 5: Verify**

Run: `cd agent-service && pip install -r requirements.txt && python -c "from main import app; print('OK')"`

**Step 6: Commit**

```bash
git add agent-service/
git commit -m "feat(agent): scaffold agent-service with FastAPI + config"
```

---

### Task 2: Add PostgreSQL to Docker Compose + DB schema

**Files:**
- Modify: `docker-compose.yml`
- Create: `agent-service/state/__init__.py`
- Create: `agent-service/state/models.py`
- Create: `agent-service/state/database.py`
- Create: `agent-service/state/checkpointer.py`

**Step 1: Add postgres + agent-service to docker-compose.yml**

Add after `rag-service` block:

```yaml
  postgres:
    image: postgres:16-alpine
    container_name: pocharlies-postgres
    restart: unless-stopped
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: langgraph
      POSTGRES_USER: agent
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-agent-secret}
    networks:
      - skirmshop-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agent -d langgraph"]
      interval: 10s
      timeout: 5s
      retries: 5

  agent-service:
    build:
      context: ./agent-service
      dockerfile: Dockerfile
    container_name: pocharlies-agent
    restart: unless-stopped
    ports:
      - "127.0.0.1:8100:8100"
    environment:
      - DATABASE_URL=postgresql+asyncpg://agent:${POSTGRES_PASSWORD:-agent-secret}@pocharlies-postgres:5432/langgraph
      - REDIS_URL=redis://pocharlies-redis:6379/1
      - RAG_SERVICE_URL=http://pocharlies-rag:5000
      - LLM_BASE_URL=${LLM_BASE_URL:-http://host.docker.internal:8000/v1}
      - LLM_API_KEY=${LLM_API_KEY:-none}
      - LLM_MODEL=${LLM_MODEL:-}
    networks:
      - skirmshop-network
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
      rag-service:
        condition: service_healthy
```

Add `postgres_data:` to the `volumes:` section.

**Step 2: Create state/__init__.py, state/database.py**

```python
# agent-service/state/__init__.py
```

```python
# agent-service/state/database.py
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from config import settings

engine: AsyncEngine = create_async_engine(settings.database_url, pool_size=5, max_overflow=10)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
```

**Step 3: Create state/models.py**

```python
# agent-service/state/models.py
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Task(Base):
    __tablename__ = "tasks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(String, nullable=False)
    trigger = Column(String, nullable=False)
    trigger_ref = Column(String, nullable=True)
    status = Column(String, default="pending")
    prompt = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    tools_used = Column(JSONB, server_default=text("'[]'::jsonb"))
    started_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSONB, server_default=text("'{}'::jsonb"))
    logs = relationship("TaskLog", back_populates="task", cascade="all, delete-orphan")


class TaskLog(Base):
    __tablename__ = "task_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=False)
    level = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    data = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    task = relationship("Task", back_populates="logs")
```

**Step 4: Create state/checkpointer.py**

```python
# agent-service/state/checkpointer.py
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from config import settings

_pg_uri = settings.database_url_sync

async def get_checkpointer() -> AsyncPostgresSaver:
    checkpointer = AsyncPostgresSaver.from_conn_string(_pg_uri)
    await checkpointer.setup()
    return checkpointer
```

**Step 5: Wire into main.py lifespan**

Update lifespan to init DB + checkpointer.

**Step 6: Commit**

```bash
git add agent-service/state/ docker-compose.yml
git commit -m "feat(agent): add Postgres schema + LangGraph checkpointer"
```

---

### Task 3: MCP Client Manager

**Files:**
- Create: `agent-service/mcp/__init__.py`
- Create: `agent-service/mcp/manager.py`
- Create: `agent-service/mcp/mcp_servers.json`

**Step 1: Create mcp_servers.json**

```json
{
  "servers": {
    "pocharlies-rag": {
      "type": "sse",
      "url": "http://pocharlies-rag:5000/mcp/sse",
      "description": "Pocharlies RAG - web crawling, products, competitors, translation"
    }
  }
}
```

**Step 2: Create mcp/__init__.py**

Empty file.

**Step 3: Create mcp/manager.py**

The MCP Manager connects to configured servers, discovers tools via `tools/list`, wraps each as a LangChain `StructuredTool` (namespaced as `server__tool_name`), and provides a unified `get_all_tools()` method.

Key classes:
- `MCPServerConnection`: holds connection state for one server (SSE or stdio transport), handles connect/disconnect/tool wrapping
- `MCPManager`: loads config, manages all connections, provides `start()`, `stop()`, `reload()`, `add_server()`, `remove_server()`, `get_all_tools()`, `get_server_status()`

Each tool call routes through the MCP session: `session.call_tool(tool_name, arguments=kwargs)` and returns concatenated text content from the response.

Config supports `${ENV_VAR}` expansion in env fields for stdio servers.

**Step 4: Verify**

Run: `cd agent-service && python -c "from mcp.manager import MCPManager; print('OK')"`

**Step 5: Commit**

```bash
git add agent-service/mcp/
git commit -m "feat(agent): add MCP client manager with dynamic tool loading"
```

---

### Task 4: Supervisor Agent Graph

**Files:**
- Create: `agent-service/graphs/__init__.py`
- Create: `agent-service/graphs/supervisor.py`

**Step 1: Create graphs/supervisor.py**

The supervisor is a LangGraph ReAct StateGraph:
- Uses `MessagesState` (built-in message list with `add_messages` reducer)
- `supervisor_node`: prepends system prompt, invokes LLM with bound tools
- `should_continue`: routes to `ToolNode` if tool calls present, else END
- `ToolNode`: LangGraph prebuilt node that executes tool calls
- Compiled with `AsyncPostgresSaver` checkpointer

LLM setup:
```python
model = ChatOpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    model=settings.llm_model or "default",
    temperature=settings.llm_temperature,
)
```

System prompt covers: capabilities, tool usage, destructive action confirmation, formatting guidelines.

**Step 2: Wire into main.py lifespan** (after MCP manager starts)

```python
tools = mcp_manager.get_all_tools()
graph = create_supervisor(checkpointer, tools)
app.state.supervisor = graph
```

**Step 3: Verify graph compiles**

Run with MemorySaver and no tools to confirm structure.

**Step 4: Commit**

```bash
git add agent-service/graphs/
git commit -m "feat(agent): add supervisor ReAct agent graph"
```

---

### Task 5: Chat API (REST + WebSocket)

**Files:**
- Create: `agent-service/api/__init__.py`
- Create: `agent-service/api/chat.py`
- Modify: `agent-service/main.py` (register router)

**Step 1: Create api/chat.py**

Two endpoints:
- `POST /api/chat` (REST): sends message, returns full response (non-streaming)
- `WS /api/chat/ws` (WebSocket): streams tool calls, LLM tokens, and results in real-time

WebSocket protocol:
- Client sends: `{"type": "message", "content": "...", "thread_id": "..."}`
- Server streams: `{"type": "message"|"tool_call"|"tool_result"|"done"|"error", ...}`

Uses `graph.astream_events(..., version="v2")` for streaming.

**Step 2: Register router**

**Step 3: Commit**

```bash
git add agent-service/api/
git commit -m "feat(agent): add chat REST + WebSocket endpoints"
```

---

### Task 6: Tasks, MCP, & Health API routers

**Files:**
- Create: `agent-service/api/tasks.py`
- Create: `agent-service/api/mcp_api.py`
- Create: `agent-service/api/health.py`
- Modify: `agent-service/main.py` (register routers)

**Step 1: Create api/tasks.py**

Endpoints: `GET /api/tasks`, `GET /api/tasks/{id}`, `GET /api/tasks/{id}/logs`
Uses SQLAlchemy async queries against the tasks/task_logs tables.

**Step 2: Create api/mcp_api.py**

Endpoints:
- `GET /api/mcp/servers` - list connected servers with status
- `POST /api/mcp/servers` - add new MCP server
- `DELETE /api/mcp/servers/{name}` - remove server
- `GET /api/mcp/servers/{name}/tools` - list tools from one server
- `GET /api/mcp/tools` - all tools unified
- `POST /api/mcp/servers/reload` - hot-reload all + rebuild supervisor graph

**Step 3: Create api/health.py**

Endpoints:
- `GET /api/health` - simple OK check
- `GET /api/health/dependencies` - checks Postgres, Redis, RAG service, MCP servers
- `GET /api/stats` - task counts + MCP server/tool counts

**Step 4: Register all routers in main.py, remove standalone health endpoint**

**Step 5: Commit**

```bash
git add agent-service/api/
git commit -m "feat(agent): add tasks, MCP, and health API routers"
```

---

### Task 7: Chat UI (static files)

**Files:**
- Create: `agent-service/static/index.html`
- Create: `agent-service/static/style.css`
- Create: `agent-service/static/app.js`
- Modify: `agent-service/main.py` (mount static files)

**Step 1: Create index.html**

Layout: sidebar (threads + MCP status) | main chat area | input | status bar.
Minimal semantic HTML. Links to style.css and app.js.

**Step 2: Create style.css**

Dark theme with CSS variables. Monospace font. Styled for: sidebar, messages (user/assistant), tool calls with results, interrupt approve/reject buttons, input area, status bar.

**Step 3: Create app.js**

IMPORTANT: Use safe DOM methods only. Use `textContent` for all untrusted content. Use `document.createElement` + `appendChild` pattern instead of innerHTML. Never set innerHTML with user or server content.

Key functions:
- `connect()`: WebSocket to `/api/chat/ws`, auto-reconnect on close
- `handleServerMessage()`: switch on message type, create DOM elements safely
- `addMessage(role, content)`: creates `.message` div with `textContent`
- `addToolCall(tool)`: creates `.tool-call` div
- `addInterrupt(data)`: creates approve/reject buttons via `document.createElement`
- `send()`: sends JSON over WebSocket
- Thread management: new thread, switch thread, sidebar rendering
- Auto-resize textarea, Enter to send

**Step 4: Mount static files in main.py**

```python
app.mount("/static", StaticFiles(directory="static"), name="static")
```

Plus a catch-all `GET /` that returns `index.html`.

**Step 5: Commit**

```bash
git add agent-service/static/
git commit -m "feat(agent): add chat UI with WebSocket streaming"
```

---

### Task 8: Add SSE transport to existing MCP server

**Files:**
- Modify: `mcp-server/server.py`

**Step 1: Update the server entry point**

At the bottom of `mcp-server/server.py`, make the transport configurable:

```python
if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)
```

This allows stdio (Claude Code) and SSE (agent-service) modes.

**Step 2: Commit**

```bash
git add mcp-server/server.py
git commit -m "feat(mcp): support SSE transport for agent-service connection"
```

---

### Task 9: Wire everything together — final main.py

**Files:**
- Modify: `agent-service/main.py` (final consolidated version)

**Step 1: Write final main.py with complete lifespan**

Lifespan order:
1. Create database tables (`Base.metadata.create_all`)
2. Initialize LangGraph checkpointer (`get_checkpointer()`)
3. Start MCP Manager (`mcp_manager.start()`)
4. Build supervisor graph (`create_supervisor(checkpointer, tools)`)
5. Store all on `app.state`

Shutdown: `mcp_manager.stop()`, `engine.dispose()`

Register routers: chat, tasks, mcp, health.
Mount static files. Serve index.html at `/`.

**Step 2: Verify full app loads**

Run: `python -c "from main import app; print([r.path for r in app.routes if hasattr(r, 'path')])"`

**Step 3: Docker build test**

Run: `docker compose build agent-service`

**Step 4: Commit**

```bash
git add agent-service/
git commit -m "feat(agent): complete Phase 1 - supervisor + MCP + chat UI + Swagger"
```

---

### Task 10: End-to-end smoke test

**Step 1:** `docker compose up -d postgres redis rag-service agent-service`

**Step 2:** `curl http://localhost:8100/api/health`
Expected: `{"status":"ok","service":"agent-orchestrator","version":"1.0.0"}`

**Step 3:** Open `http://localhost:8100/docs` — verify Swagger shows all tag groups

**Step 4:** `curl http://localhost:8100/api/mcp/servers` — verify MCP connection status

**Step 5:** Open `http://localhost:8100/` — verify chat UI loads

**Step 6:** Test chat:
```bash
curl -X POST http://localhost:8100/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What tools do you have available?"}'
```

**Step 7:** Fix any issues and commit:
```bash
git add -A && git commit -m "fix(agent): smoke test fixes for Phase 1"
```
