"""End-to-end retrieval pipeline."""

from typing import Any

from qdrant_client import QdrantClient

from retrieval.hybrid_search import hybrid_search
from retrieval.reranker import rerank_documents


def retrieve_documents(
    client: QdrantClient,
    query: str,
    documents: list[dict[str, Any]],
    *,
    embedding_model: Any = None,
    ranker: Any = None,
    candidate_limit: int = 10,
    final_limit: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve hybrid candidates and rerank the final results."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must be a non-empty string.")

    if candidate_limit <= 0 or final_limit <= 0:
        raise ValueError("Retrieval limits must be greater than zero.")

    if candidate_limit < final_limit:
        raise ValueError("candidate_limit must be greater than or equal to final_limit.")

    if not documents:
        raise ValueError("At least one document is required.")

    candidates = hybrid_search(
        client,
        query,
        documents,
        embedding_model=embedding_model,
        limit=candidate_limit,
    )

    return rerank_documents(
        query,
        candidates,
        ranker=ranker,
        limit=final_limit,
    )
