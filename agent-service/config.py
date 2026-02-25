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
    mcp_servers_path: str = "mcp_client/mcp_servers.json"
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
