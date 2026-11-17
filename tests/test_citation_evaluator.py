
import pytest

from evaluation.citation_evaluator import evaluate_citations


def test_all_citations_are_valid():
    report = "RAG combines retrieval and generation [C1]."

    citations = [
        {
            "citation_id": "C1",
            "source_id": "source-001",
            "title": "Introduction to RAG",
            "url": "https://example.com/rag",
        }
    ]

    result = evaluate_citations(report, citations)

    assert result["total_citations"] == 1
    assert result["valid_citations"] == ["C1"]
    assert result["invalid_citations"] == []
    assert result["citation_coverage"] == 1.0


def test_invalid_citation_is_detected():
    report = "RAG improves grounding [C2]."

    citations = [
        {
            "citation_id": "C1",
            "source_id": "source-001",
        }
    ]

    result = evaluate_citations(report, citations)

    assert result["invalid_citations"] == ["C2"]
    assert result["citation_coverage"] == 0.0


def test_multiple_citations():
    report = "RAG retrieves information [C1]. Hybrid search combines methods [C2]."

    citations = [
        {"citation_id": "C1", "source_id": "source-001"},
        {"citation_id": "C2", "source_id": "source-002"},
    ]

    result = evaluate_citations(report, citations)

    assert result["total_citations"] == 2
    assert result["unique_source_count"] == 2
    assert result["citation_coverage"] == 1.0


def test_repeated_citation_ids():
    report = "RAG retrieves information [C1]. RAG improves grounding [C1]."

    citations = [
        {"citation_id": "C1", "source_id": "source-001"}
    ]

    result = evaluate_citations(report, citations)

    assert result["total_citations"] == 2
    assert result["valid_citations"] == ["C1", "C1"]
    assert result["citation_coverage"] == 1.0


def test_report_without_citations():
    result = evaluate_citations("RAG combines retrieval and generation.", [])

    assert result["total_citations"] == 0
    assert result["valid_citations"] == []
    assert result["invalid_citations"] == []
    assert result["citation_coverage"] == 0.0


def test_invalid_input_is_rejected():
    with pytest.raises(TypeError):
        evaluate_citations(123, [])

    with pytest.raises(TypeError):
        evaluate_citations("Some report [C1].", "not-a-list")
