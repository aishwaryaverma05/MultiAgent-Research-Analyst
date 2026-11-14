"""Document reranking using FlashRank."""

import math
from typing import Any

from flashrank import Ranker, RerankRequest

DEFAULT_RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"


def create_reranker() -> Ranker:
    """Initialize the FlashRank reranking model."""
    return Ranker(model_name=DEFAULT_RERANK_MODEL)


def rerank_documents(
    query: str,
    documents: list[dict[str, Any]],
    *,
    ranker: Any = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Rerank hybrid search results according to query relevance."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Reranking query must be a non-empty string.")

    if limit <= 0:
        raise ValueError("limit must be greater than zero.")

    if not documents:
        return []

    passages = []

    for index, result in enumerate(documents):
        document = result.get("document", {})
        text = document.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Every result must contain a document with non-empty text.")

        passages.append(
            {
                "id": str(index),
                "text": text,
                "meta": {"index": index},
            }
        )

    model = ranker if ranker is not None else create_reranker()

    request = RerankRequest(
        query=query,
        passages=passages,
    )

    reranked_results = model.rerank(request)

    if not isinstance(reranked_results, list):
        raise ValueError("Reranker must return a list of results.")

    final_results = []

    for result in reranked_results[:limit]:
        if not isinstance(result, dict):
            raise ValueError("Reranker returned an invalid result.")

        try:
            original_index = int(result["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Reranker returned an invalid document ID.") from exc

        if original_index < 0 or original_index >= len(documents):
            raise ValueError("Reranker returned an invalid document ID.")

        try:
            score = float(result["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Reranker returned a missing or invalid score.") from exc

        if not math.isfinite(score):
            raise ValueError("Reranker returned a non-finite score.")

        original_result = dict(documents[original_index])
        original_result["rerank_score"] = score

        final_results.append(original_result)

    return final_results
