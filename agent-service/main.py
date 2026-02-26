"""Pocharlies Agent Orchestrator - FastAPI entrypoint."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from graphs.supervisor import create_supervisor
from mcp_client.manager import MCPManager
from state.checkpointer import get_checkpointer_cm
from state.database import engine
from state.models import Base

logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Agent service starting...")

    # 1. Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready")

    # 2. Initialize LangGraph checkpointer (context manager)
    checkpointer_cm = get_checkpointer_cm()
    checkpointer = await checkpointer_cm.__aenter__()
    await checkpointer.setup()
    app.state.checkpointer = checkpointer
    logger.info("LangGraph checkpointer ready")

    # 3. Start MCP Manager
    mcp_manager = MCPManager()
    await mcp_manager.start()
    app.state.mcp_manager = mcp_manager

    # 4. Build supervisor graph
    tools = mcp_manager.get_all_tools()
    graph = create_supervisor(checkpointer, tools)
    app.state.supervisor = graph
    logger.info("Supervisor graph ready with %d tools", len(tools))

    yield

    # Shutdown
    logger.info("Agent service shutting down...")
    await mcp_manager.stop()
    await checkpointer_cm.__aexit__(None, None, None)
    await engine.dispose()
    logger.info("Cleanup complete")


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

# Register API routers
from api.chat import router as chat_router
from api.health import router as health_router
from api.mcp_api import router as mcp_router
from api.tasks import router as tasks_router

app.include_router(chat_router)
app.include_router(tasks_router)
app.include_router(mcp_router)
app.include_router(health_router)

# Serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    """Serve the chat UI."""
    return FileResponse(STATIC_DIR / "index.html")
