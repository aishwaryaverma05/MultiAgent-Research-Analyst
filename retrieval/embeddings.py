"""
Gemini embedding model initialization.

This module creates embeddings for documents and user queries.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import get_settings


def create_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """
    Initialize the configured Gemini embedding model.

    Credentials are validated only when embeddings are requested.
    """

    settings = get_settings()
    api_key = settings.require_gemini_credentials()

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        api_key=api_key,
    )
