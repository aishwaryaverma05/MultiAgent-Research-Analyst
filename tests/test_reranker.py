import pytest

from retrieval.reranker import rerank_documents


class FakeRanker:
    def rerank(self, request):
        return [
            {
                "id": "1",
                "score": 0.95,
                "text": request.passages[1]["text"],
            },
            {
                "id": "0",
                "score": 0.70,
                "text": request.passages[0]["text"],
            },
        ]


def test_reranker_reorders_documents():
    documents = [
        {"document": {"text": "Climate change affects weather."}},
        {"document": {"text": "AI uses machine learning."}},
    ]

    results = rerank_documents(
        "How does AI use machine learning?",
        documents,
        ranker=FakeRanker(),
    )

    assert results[0]["document"]["text"] == "AI uses machine learning."
    assert results[0]["rerank_score"] == 0.95


def test_reranker_preserves_original_metadata():
    documents = [
        {
            "document": {
                "text": "Artificial intelligence research.",
                "source_id": "source-1",
            }
        }
    ]

    class SingleResultRanker:
        def rerank(self, request):
            return [{"id": "0", "score": 0.9}]

    results = rerank_documents(
        "Artificial intelligence",
        documents,
        ranker=SingleResultRanker(),
    )

    assert results[0]["document"]["source_id"] == "source-1"


def test_reranker_respects_result_limit():
    documents = [
        {"document": {"text": "Document one"}},
        {"document": {"text": "Document two"}},
    ]

    results = rerank_documents(
        "Document",
        documents,
        ranker=FakeRanker(),
        limit=1,
    )

    assert len(results) == 1


def test_reranker_returns_empty_list_for_no_documents():
    results = rerank_documents("Artificial intelligence", [])

    assert results == []


def test_reranker_rejects_empty_query():
    with pytest.raises(ValueError, match="query"):
        rerank_documents("   ", [])


def test_reranker_rejects_invalid_limit():
    with pytest.raises(ValueError, match="limit"):
        rerank_documents("Python", [], limit=0)


def test_reranker_rejects_empty_document_text():
    documents = [{"document": {"text": "   "}}]

    with pytest.raises(ValueError, match="non-empty text"):
        rerank_documents("Python", documents, ranker=FakeRanker())


def test_reranker_rejects_non_string_query():
    with pytest.raises(ValueError, match="non-empty string"):
        rerank_documents(None, [])


def test_reranker_rejects_invalid_document_id():
    documents = [{"document": {"text": "Artificial intelligence research."}}]

    class InvalidIDRanker:
        def rerank(self, request):
            return [{"id": "99", "score": 0.9}]

    with pytest.raises(ValueError, match="invalid document ID"):
        rerank_documents(
            "Artificial intelligence",
            documents,
            ranker=InvalidIDRanker(),
        )


def test_reranker_rejects_missing_score():
    documents = [{"document": {"text": "Artificial intelligence research."}}]

    class MissingScoreRanker:
        def rerank(self, request):
            return [{"id": "0"}]

    with pytest.raises(ValueError, match="score"):
        rerank_documents(
            "Artificial intelligence",
            documents,
            ranker=MissingScoreRanker(),
        )


def test_reranker_rejects_none_results():
    documents = [{"document": {"text": "Artificial intelligence research."}}]

    class NoneRanker:
        def rerank(self, request):
            return None

    with pytest.raises(ValueError, match="results"):
        rerank_documents(
            "Artificial intelligence",
            documents,
            ranker=NoneRanker(),
        )
