
"""Custom retrieval evaluation metrics."""



def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Calculate the proportion of relevant results in the top-k results."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    relevant_retrieved = set(top_k) & set(relevant_ids)

    return len(relevant_retrieved) / len(top_k)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Calculate the proportion of relevant documents retrieved in top-k."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_ids:
        return 0.0

    relevant_retrieved = set(retrieved_ids[:k]) & set(relevant_ids)

    return len(relevant_retrieved) / len(set(relevant_ids))


def mean_reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """Calculate reciprocal rank of the first relevant result."""

    relevant_set = set(relevant_ids)

    if not relevant_set:
        return 0.0

    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in relevant_set:
            return 1.0 / rank

    return 0.0


def hit_rate_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Return 1 if at least one relevant result appears in top-k."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_ids:
        return 0.0

    return float(bool(set(retrieved_ids[:k]) & set(relevant_ids)))


def evaluate_retrieval(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
) -> dict[str, float]:
    """Calculate all custom retrieval metrics."""

    return {
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k),
        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_ids),
        "hit_rate_at_k": hit_rate_at_k(retrieved_ids, relevant_ids, k),
    }
