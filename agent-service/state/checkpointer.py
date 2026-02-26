"""LangGraph AsyncPostgresSaver setup."""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from config import settings

_pg_uri = settings.database_url_sync


def get_checkpointer_cm():
    """Return the async context manager for the Postgres checkpointer."""
    return AsyncPostgresSaver.from_conn_string(_pg_uri)
