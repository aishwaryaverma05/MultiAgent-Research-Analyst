
"""Factory for configured Groq chat models."""

from typing import Literal

from langchain_groq import ChatGroq

from config import get_settings


def create_chat_model(
    tier: Literal["primary", "fast"] = "primary",
) -> ChatGroq:
    """
    Initialize the configured Groq chat model.

    Primary: Higher-reasoning model.
    Fast: Lower-cost, faster model.
    """

    settings = get_settings()
    api_key = settings.require_groq_credentials()

    model_name = (
        settings.llm_fast_model
        if tier == "fast"
        else settings.llm_model
    )

    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=settings.llm_temperature,
        timeout=settings.request_timeout_seconds,
        max_retries=2,
    )
