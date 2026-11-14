"""Hybrid retrieval combining semantic search and BM25."""

from typing import Any

from qdrant_client import QdrantClient

from retrieval.bm25_search import BM25Index
from retrieval.semantic_search import search_documents


def reciprocal_rank_fusion(
    semantic_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    *,
    rrf_k: int = 60,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Merge two ranked document lists using RRF."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero.")

    if limit <= 0:
        raise ValueError("limit must be greater than zero.")

    combined: dict[str, dict[str, Any]] = {}

    def add_result(
        document: dict[str, Any],
        rank: int,
        retrieval_type: str,
    ) -> None:
        key = document["text"]

        if key not in combined:
            combined[key] = {
                "document": document,
                "score": 0.0,
                "semantic_rank": None,
                "bm25_rank": None,
            }

        combined[key]["score"] += 1.0 / (rrf_k + rank)
        combined[key][f"{retrieval_type}_rank"] = rank

    for rank, document in enumerate(semantic_results, start=1):
        add_result(document, rank, "semantic")

    for rank, result in enumerate(keyword_results, start=1):
        add_result(result["document"], rank, "bm25")

    return sorted(
        combined.values(),
        key=lambda result: result["score"],
        reverse=True,
    )[:limit]


def hybrid_search(
    client: QdrantClient,
    query: str,
    documents: list[dict[str, Any]],
    *,
    embedding_model: Any = None,
    limit: int = 5,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Combine semantic and BM25 rankings using Reciprocal Rank Fusion."""

    if not query.strip():
        raise ValueError("Search query must not be empty.")

    if limit <= 0:
        raise ValueError("limit must be greater than zero.")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero.")

    bm25_index = BM25Index(documents)

    raw_semantic_results = search_documents(
        client,
        query,
        embedding_model=embedding_model,
        limit=limit,
    )

    # Convert Qdrant ScoredPoint objects into ordinary dictionaries.
    semantic_results = [dict(result.payload or {}) for result in raw_semantic_results]

    keyword_results = bm25_index.search(
        query,
        limit=limit,
    )

    return reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        rrf_k=rrf_k,
        limit=limit,
    )
