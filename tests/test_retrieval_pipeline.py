from unittest.mock import patch

import pytest

from retrieval.retrieval_pipeline import retrieve_documents


@pytest.fixture
def documents():
    return [
        {"text": "Artificial intelligence uses machine learning."},
        {"text": "Climate change affects global temperatures."},
    ]


@patch("retrieval.retrieval_pipeline.rerank_documents")
@patch("retrieval.retrieval_pipeline.hybrid_search")
def test_retrieval_pipeline_combines_hybrid_and_reranking(
    mock_hybrid,
    mock_rerank,
    documents,
):
    candidates = [
        {"document": documents[0], "score": 0.03},
    ]
    final_results = [
        {"document": documents[0], "rerank_score": 0.99},
    ]

    mock_hybrid.return_value = candidates
    mock_rerank.return_value = final_results

    results = retrieve_documents(
        client=None,
        query="Artificial intelligence",
        documents=documents,
        candidate_limit=10,
        final_limit=5,
    )

    assert results == final_results
    mock_hybrid.assert_called_once()
    mock_rerank.assert_called_once()


@patch("retrieval.retrieval_pipeline.rerank_documents")
@patch("retrieval.retrieval_pipeline.hybrid_search")
def test_pipeline_passes_limits_correctly(
    mock_hybrid,
    mock_rerank,
    documents,
):
    mock_hybrid.return_value = []
    mock_rerank.return_value = []

    retrieve_documents(
        None,
        "Artificial intelligence",
        documents,
        candidate_limit=8,
        final_limit=3,
    )

    assert mock_hybrid.call_args.kwargs["limit"] == 8
    assert mock_rerank.call_args.kwargs["limit"] == 3


def test_pipeline_rejects_empty_query(documents):
    with pytest.raises(ValueError, match="query"):
        retrieve_documents(None, "   ", documents)


@pytest.mark.parametrize("query", [None, 123, "", "   "])
def test_pipeline_rejects_invalid_query_types_and_empty_values(
    documents,
    query,
):
    with pytest.raises(ValueError, match="non-empty string"):
        retrieve_documents(None, query, documents)


def test_pipeline_rejects_empty_documents():
    with pytest.raises(ValueError, match="At least one document"):
        retrieve_documents(None, "Python", [])


def test_pipeline_rejects_invalid_limits(documents):
    with pytest.raises(ValueError, match="greater than zero"):
        retrieve_documents(
            None,
            "Python",
            documents,
            candidate_limit=0,
        )


def test_pipeline_rejects_candidate_limit_smaller_than_final_limit(
    documents,
):
    with pytest.raises(ValueError, match="greater than or equal"):
        retrieve_documents(
            None,
            "Python",
            documents,
            candidate_limit=2,
            final_limit=5,
        )
