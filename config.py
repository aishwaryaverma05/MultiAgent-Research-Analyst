
"""Application configuration and environment settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings."""

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
    llm_provider: Literal["gemini", "groq"] = "groq"
    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    llm_model: str = "openai/gpt-oss-120b"
    llm_fast_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    # Embeddings
    embedding_model: str = "models/gemini-embedding-001"

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
        """Return Gemini API key or raise an error."""

        if (
            self.gemini_api_key is None
            or not self.gemini_api_key.get_secret_value().strip()
        ):
            raise ValueError(
                "GEMINI_API_KEY is missing. Add your Gemini API key to the .env file."
            )

        return self.gemini_api_key.get_secret_value()

    def require_groq_credentials(self) -> str:
        """Return Groq API key or raise an error."""

        if (
            self.groq_api_key is None
            or not self.groq_api_key.get_secret_value().strip()
        ):
            raise ValueError(
                "GROQ_API_KEY is missing. Add your Groq API key to the .env file."
            )

        return self.groq_api_key.get_secret_value()

    def require_exa_credentials(self) -> str:
        """Return Exa API key or raise an error."""

        if (
            self.exa_api_key is None
            or not self.exa_api_key.get_secret_value().strip()
        ):
            raise ValueError(
                "EXA_API_KEY is missing. Add your Exa API key to the .env file."
            )

        return self.exa_api_key.get_secret_value()

    def require_qdrant_credentials(self) -> tuple[str, str]:
        """Return Qdrant URL and API key or raise an error."""

        if self.qdrant_url is None or not self.qdrant_url.strip():
            raise ValueError(
                "QDRANT_URL is missing. Add your Qdrant URL to the .env file."
            )

        if (
            self.qdrant_api_key is None
            or not self.qdrant_api_key.get_secret_value().strip()
        ):
            raise ValueError(
                "QDRANT_API_KEY is missing. Add your Qdrant API key to the .env file."
            )

        return (
            self.qdrant_url,
            self.qdrant_api_key.get_secret_value(),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
