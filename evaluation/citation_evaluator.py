
"""Citation evaluation utilities for research reports."""

import re
from typing import Any


def evaluate_citations(
    report_text: str,
    citations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate citation validity and citation coverage."""

    if not isinstance(report_text, str):
        raise TypeError("Report text must be a string.")

    if not isinstance(citations, list):
        raise TypeError("Citations must be a list.")

    cited_ids = re.findall(r"\[([^\[\]]+)\]", report_text)

    citation_map = {
        str(citation.get("citation_id", "")): citation
        for citation in citations
        if citation.get("citation_id")
    }

    valid_citations = []
    invalid_citations = []

    for citation_id in cited_ids:
        if citation_id in citation_map:
            valid_citations.append(citation_id)
        else:
            invalid_citations.append(citation_id)

    source_ids = {
        citation.get("source_id")
        for citation in citations
        if citation.get("citation_id") and citation.get("source_id")
    }

    unique_cited_ids = set(cited_ids)

    citation_coverage = (
        len(unique_cited_ids & set(citation_map)) / len(unique_cited_ids)
        if unique_cited_ids
        else 0.0
    )

    return {
        "total_citations": len(cited_ids),
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "unique_source_count": len(source_ids),
        "citation_coverage": citation_coverage,
    }
