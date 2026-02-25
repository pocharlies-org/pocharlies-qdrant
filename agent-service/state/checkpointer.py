"""LangGraph AsyncPostgresSaver setup."""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from config import settings

_pg_uri = settings.database_url_sync


async def get_checkpointer() -> AsyncPostgresSaver:
    """Create and initialize the Postgres checkpointer."""
    checkpointer = AsyncPostgresSaver.from_conn_string(_pg_uri)
    await checkpointer.setup()
    return checkpointer
