
"""Faithfulness evaluation utilities for grounded research answers."""

import re


def split_claims(answer: str) -> list[str]:
    """Split an answer into individual non-empty sentences."""

    if not isinstance(answer, str):
        raise TypeError("Answer must be a string.")

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", answer.strip())
        if sentence.strip()
    ]


def evaluate_claim_support(
    answer: str,
    contexts: list[str],
) -> dict[str, object]:
    """Estimate claim support using simple token overlap.

    This is a lightweight baseline, not an LLM-based factual verifier.
    """

    if not isinstance(answer, str):
        raise TypeError("Answer must be a string.")

    if not contexts:
        raise ValueError("At least one reference context is required.")

    claims = split_claims(answer)

    context_text = " ".join(contexts).lower()
    context_tokens = set(re.findall(r"\b\w+\b", context_text))

    supported_claims = []
    unsupported_claims = []

    for claim in claims:
        claim_tokens = set(re.findall(r"\b\w+\b", claim.lower()))

        if not claim_tokens:
            continue

        overlap = claim_tokens & context_tokens
        support_ratio = len(overlap) / len(claim_tokens)

        if support_ratio >= 0.5:
            supported_claims.append(claim)
        else:
            unsupported_claims.append(claim)

    total_claims = len(supported_claims) + len(unsupported_claims)

    faithfulness_score = (
        len(supported_claims) / total_claims
        if total_claims
        else 0.0
    )

    return {
        "faithfulness_score": faithfulness_score,
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
    }
