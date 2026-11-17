
import pytest

from evaluation.retrieval_metrics import (
    evaluate_retrieval,
    hit_rate_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    retrieved = ["A", "B", "C", "D"]
    relevant = ["A", "C"]

    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_recall_at_k():
    retrieved = ["A", "B", "C", "D"]
    relevant = ["A", "C", "E"]

    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_mean_reciprocal_rank():
    retrieved = ["A", "B", "C", "D"]
    relevant = ["C", "D"]

    assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)


def test_hit_rate_at_k():
    retrieved = ["A", "B", "C"]
    relevant = ["C"]

    assert hit_rate_at_k(retrieved, relevant, 3) == 1.0
    assert hit_rate_at_k(retrieved, relevant, 2) == 0.0


def test_evaluate_retrieval():
    result = evaluate_retrieval(
        retrieved_ids=["A", "B", "C"],
        relevant_ids=["A", "C"],
        k=2,
    )

    assert result == {
        "precision_at_k": 0.5,
        "recall_at_k": 0.5,
        "mrr": 1.0,
        "hit_rate_at_k": 1.0,
    }


def test_invalid_k_is_rejected():
    with pytest.raises(ValueError):
        precision_at_k(["A"], ["A"], 0)

    with pytest.raises(ValueError):
        recall_at_k(["A"], ["A"], 0)

    with pytest.raises(ValueError):
        hit_rate_at_k(["A"], ["A"], 0)


def test_empty_relevant_documents():
    assert precision_at_k(["A"], [], 5) == 0.0
    assert recall_at_k(["A"], [], 5) == 0.0
    assert mean_reciprocal_rank(["A"], []) == 0.0
    assert hit_rate_at_k(["A"], [], 5) == 0.0
