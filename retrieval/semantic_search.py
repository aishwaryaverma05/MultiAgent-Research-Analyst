"""Semantic document indexing and retrieval using Gemini and Qdrant."""

from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient

from retrieval.embeddings import create_embeddings_model
from retrieval.vector_store import (
    DEFAULT_COLLECTION_NAME,
    ensure_collection,
    search_vectors,
    upsert_vectors,
)


def index_documents(
    client: QdrantClient,
    documents: list[dict[str, Any]],
    *,
    embedding_model: Any = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> list[str]:
    """Generate document embeddings and store them in Qdrant."""

    if not documents:
        return []

    texts = []
    payloads = []

    for document in documents:
        text = document.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Every document must contain non-empty text.")

        texts.append(text)
        payloads.append(
            {
                **{key: value for key, value in document.items() if key != "text"},
                "text": text,
            }
        )

    model = embedding_model or create_embeddings_model()
    vectors = model.embed_documents(texts)
    if not vectors:
        raise ValueError("Embedding model returned no vectors.")

    if len(vectors) != len(texts):
        raise ValueError("Embedding count does not match document count.")

    vector_size = len(vectors[0])

    if vector_size == 0:
        raise ValueError("Generated embeddings must not be empty.")

    if any(len(vector) != vector_size for vector in vectors):
        raise ValueError("All document embeddings must have equal dimensions.")

    ensure_collection(
        client,
        vector_size=vector_size,
        collection_name=collection_name,
    )

    ids = [str(uuid4()) for _ in documents]

    upsert_vectors(
        client,
        ids=ids,
        vectors=vectors,
        payloads=payloads,
        collection_name=collection_name,
    )

    return ids


def search_documents(
    client: QdrantClient,
    query: str,
    *,
    embedding_model: Any = None,
    limit: int = 5,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """Find relevant documents using a natural-language query."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must be a non-empty string.")

    if limit <= 0:
        raise ValueError("limit must be greater than zero.")

    model = embedding_model or create_embeddings_model()
    query_vector = model.embed_query(query)

    return search_vectors(
        client,
        query_vector=query_vector,
        limit=limit,
        collection_name=collection_name,
    )
