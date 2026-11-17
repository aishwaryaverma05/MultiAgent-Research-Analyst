
"""Ragas 0.4.4 integration for RAG evaluation."""

from typing import Any

REQUIRED_METRICS = {
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
}


def create_ragas_metrics(
    llm: Any,
    embeddings: Any,
) -> dict[str, Any]:
    """Initialize Ragas evaluation metrics using supplied models."""

    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(
            llm=llm,
            embeddings=embeddings,
        ),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


async def evaluate_with_ragas(
    metrics: dict[str, Any],
    *,
    user_input: str,
    response: str,
    retrieved_contexts: list[str],
    reference: str,
) -> dict[str, float]:
    """Evaluate one RAG response using Ragas metrics."""

    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("User input must not be empty.")

    if not isinstance(response, str) or not response.strip():
        raise ValueError("Response must not be empty.")

    if (
        not isinstance(retrieved_contexts, list)
        or not retrieved_contexts
        or any(
            not isinstance(context, str) or not context.strip()
            for context in retrieved_contexts
        )
    ):
        raise ValueError("Retrieved contexts must be a non-empty list of strings.")

    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Reference answer must not be empty.")

    if not isinstance(metrics, dict):
        raise TypeError("Metrics must be a dictionary.")

    missing_metrics = REQUIRED_METRICS - metrics.keys()
    unsupported_metrics = metrics.keys() - REQUIRED_METRICS

    if unsupported_metrics:
        raise ValueError(
            f"Unsupported Ragas metrics: {sorted(unsupported_metrics)}"
        )

    if missing_metrics:
        raise ValueError(
            f"Missing required Ragas metrics: {sorted(missing_metrics)}"
        )

    metric_inputs = {
        "faithfulness": {
            "user_input": user_input,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
        },
        "answer_relevancy": {
            "user_input": user_input,
            "response": response,
        },
        "context_precision": {
            "user_input": user_input,
            "retrieved_contexts": retrieved_contexts,
            "reference": reference,
        },
        "context_recall": {
            "user_input": user_input,
            "retrieved_contexts": retrieved_contexts,
            "reference": reference,
        },
    }

    results = {}

    for metric_name, metric in metrics.items():
        result = await metric.ascore(**metric_inputs[metric_name])
        score = float(result.value)

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Invalid score returned by {metric_name}: {score}"
            )

        results[metric_name] = score

    return results
