
"""Adapters for using existing Gemini models with Ragas."""

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from llm.gemini import create_chat_model
from retrieval.embeddings import create_embeddings_model


def create_ragas_models():
    """Create Ragas-compatible Gemini LLM and embedding wrappers."""

    llm = LangchainLLMWrapper(create_chat_model())
    embeddings = LangchainEmbeddingsWrapper(create_embeddings_model())

    return llm, embeddings
