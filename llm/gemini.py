
from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_settings


def create_chat_model() -> ChatGoogleGenerativeAI:
    """
    Initialize the configured Gemini chat model.

    Credentials are validated only when a live model is requested.
    """

    settings = get_settings()  #config.py  se data fetch

    api_key = settings.require_gemini_credentials()

    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=api_key,
        temperature=settings.llm_temperature,
        timeout=settings.request_timeout_seconds,
        max_retries=2,
    )
