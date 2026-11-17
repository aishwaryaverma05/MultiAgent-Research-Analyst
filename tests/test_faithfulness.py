
import pytest

from evaluation.faithfulness import (
    evaluate_claim_support,
    split_claims,
)


def test_split_claims():
    answer = "RAG retrieves documents. It generates grounded answers!"

    assert split_claims(answer) == [
        "RAG retrieves documents.",
        "It generates grounded answers!",
    ]


def test_supported_claim():
    answer = "RAG retrieves relevant documents."
    contexts = [
        "RAG retrieves relevant documents before generating an answer."
    ]

    result = evaluate_claim_support(answer, contexts)

    assert result["faithfulness_score"] == 1.0
    assert len(result["supported_claims"]) == 1
    assert len(result["unsupported_claims"]) == 0


def test_unsupported_claim():
    answer = "Python was invented in 1995."
    contexts = [
        "RAG retrieves relevant documents before generating an answer."
    ]

    result = evaluate_claim_support(answer, contexts)

    assert result["faithfulness_score"] == 0.0
    assert len(result["unsupported_claims"]) == 1


def test_mixed_supported_and_unsupported_claims():
    answer = (
        "RAG retrieves relevant documents. "
        "The system was invented in 1995."
    )
    contexts = [
        "RAG retrieves relevant documents before generating an answer."
    ]

    result = evaluate_claim_support(answer, contexts)

    assert result["total_claims"] == 2
    assert len(result["supported_claims"]) == 1
    assert len(result["unsupported_claims"]) == 1
    assert result["faithfulness_score"] == 0.5


def test_empty_context_is_rejected():
    with pytest.raises(ValueError):
        evaluate_claim_support("Some answer.", [])


def test_non_string_answer_is_rejected():
    with pytest.raises(TypeError):
        split_claims(123)
