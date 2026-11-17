
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from evaluation.ragas_evaluator import (
    create_ragas_metrics,
    evaluate_with_ragas,
)


class FakeMetric:
    def __init__(self, value):
        self.ascore = AsyncMock(
            return_value=SimpleNamespace(value=value)
        )


@pytest.mark.asyncio
async def test_evaluate_with_ragas():
    metrics = {
        "faithfulness": FakeMetric(0.9),
        "answer_relevancy": FakeMetric(0.8),
        "context_precision": FakeMetric(0.7),
        "context_recall": FakeMetric(0.85),
    }

    result = await evaluate_with_ragas(
        metrics,
        user_input="What is RAG?",
        response="RAG combines retrieval and generation.",
        retrieved_contexts=["RAG retrieves external information."],
        reference="RAG combines retrieval with generation.",
    )

    assert result == {
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
        "context_precision": 0.7,
        "context_recall": 0.85,
    }

    for metric in metrics.values():
        metric.ascore.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_user_input_is_rejected():
    with pytest.raises(ValueError, match="User input"):
        await evaluate_with_ragas(
            {},
            user_input=" ",
            response="Some answer",
            retrieved_contexts=["Some context"],
            reference="Some reference",
        )


@pytest.mark.asyncio
async def test_empty_response_is_rejected():
    with pytest.raises(ValueError, match="Response"):
        await evaluate_with_ragas(
            {},
            user_input="Some question",
            response=" ",
            retrieved_contexts=["Some context"],
            reference="Some reference",
        )


@pytest.mark.asyncio
async def test_empty_contexts_are_rejected():
    with pytest.raises(ValueError, match="Retrieved contexts"):
        await evaluate_with_ragas(
            {},
            user_input="Some question",
            response="Some answer",
            retrieved_contexts=[],
            reference="Some reference",
        )


@pytest.mark.asyncio
async def test_empty_reference_is_rejected():
    with pytest.raises(ValueError, match="Reference"):
        await evaluate_with_ragas(
            {},
            user_input="Some question",
            response="Some answer",
            retrieved_contexts=["Some context"],
            reference=" ",
        )


@pytest.mark.asyncio
async def test_unsupported_metric_is_rejected():
    with pytest.raises(ValueError, match="Unsupported Ragas metric"):
        await evaluate_with_ragas(
            {"unknown": FakeMetric(0.5)},
            user_input="Some question",
            response="Some answer",
            retrieved_contexts=["Some context"],
            reference="Some reference",
        )


def test_create_ragas_metrics():
    with (
        patch("ragas.metrics.collections.Faithfulness") as faithfulness,
        patch("ragas.metrics.collections.AnswerRelevancy") as answer_relevancy,
        patch("ragas.metrics.collections.ContextPrecision") as context_precision,
        patch("ragas.metrics.collections.ContextRecall") as context_recall,
    ):
        metrics = create_ragas_metrics(
            llm="fake-llm",
            embeddings="fake-embeddings",
        )

    assert set(metrics) == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    }

    faithfulness.assert_called_once_with(llm="fake-llm")
    answer_relevancy.assert_called_once_with(
        llm="fake-llm",
        embeddings="fake-embeddings",
    )
    context_precision.assert_called_once_with(llm="fake-llm")
    context_recall.assert_called_once_with(llm="fake-llm")
