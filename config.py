
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # file m data kese read hoga 
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Multi-Agent Research Analyst"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    # LLM configuration
    llm_provider: Literal["gemini"] = "gemini"
    gemini_api_key: SecretStr | None = None
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # Embeddings
    embedding_model: str = "models/text-embedding-004"

    # Search
    exa_api_key: SecretStr | None = None

    # Vector database
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None

    # Database
    database_url: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/research_analyst"
    )

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Observability
    langsmith_api_key: SecretStr | None = None
    langsmith_tracing: bool = False

    # Reliability
    request_timeout_seconds: float = Field(default=20.0, gt=0)
    max_revisions: int = Field(default=3, ge=0, le=3)

    def require_gemini_credentials(self) -> str:

        if (
            self.gemini_api_key is None
            or not self.gemini_api_key.get_secret_value().strip()
        ):
            raise ValueError(
                "GEMINI_API_KEY is missing. "
                "Add your Gemini API key to the .env file."
            )

        return self.gemini_api_key.get_secret_value()

    def require_exa_credentials(self) -> str:
        if (
            self.exa_api_key is None
            or not self.exa_api_key.get_secret_value().strip()
        ):
            raise ValueError(
                "EXA_API_KEY is missing. "
                "Add your Exa API key to the .env file."
            )

        return self.exa_api_key.get_secret_value()

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
