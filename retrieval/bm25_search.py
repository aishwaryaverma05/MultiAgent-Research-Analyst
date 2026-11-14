"""Keyword-based document retrieval using BM25."""

import re
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    """Build and search a BM25 index over documents."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        if not documents:
            raise ValueError("At least one document is required.")

        self.documents = []
        tokenized_corpus = []

        for document in documents:
            text = document.get("text")

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Every document must contain non-empty text.")

            tokens = tokenize(text)

            if not tokens:
                raise ValueError("Document text must contain at least one word.")

            self.documents.append(dict(document))
            tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the highest-ranked documents for a query."""

        if not query.strip():
            raise ValueError("Search query must not be empty.")

        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        query_tokens = tokenize(query)

        if not query_tokens:
            raise ValueError("Search query must contain at least one word.")

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:limit]

        return [
            {
                "document": self.documents[index],
                "score": float(scores[index]),
            }
            for index in ranked_indices
        ]
