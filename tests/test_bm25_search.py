import pytest

from retrieval.bm25_search import BM25Index, tokenize


def test_tokenize_converts_text_to_lowercase():
    assert tokenize("Artificial Intelligence") == [
        "artificial",
        "intelligence",
    ]


def test_bm25_returns_relevant_documents_first():
    documents = [
        {
            "text": "Artificial intelligence uses machine learning.",
            "source_id": "ai-source",
        },
        {
            "text": "Climate change affects global temperatures.",
            "source_id": "climate-source",
        },
    ]

    index = BM25Index(documents)
    results = index.search("artificial intelligence machine learning")

    assert results[0]["document"]["source_id"] == "ai-source"


def test_bm25_respects_result_limit():
    documents = [
        {"text": "Python programming language", "source_id": "1"},
        {"text": "Python software development", "source_id": "2"},
        {"text": "Python machine learning", "source_id": "3"},
    ]

    index = BM25Index(documents)
    results = index.search("Python", limit=2)

    assert len(results) == 2


def test_bm25_rejects_empty_documents():
    with pytest.raises(ValueError, match="At least one document"):
        BM25Index([])


def test_bm25_rejects_empty_document_text():
    with pytest.raises(ValueError, match="non-empty text"):
        BM25Index([{"text": "   "}])


def test_bm25_rejects_empty_query():
    index = BM25Index([{"text": "Python programming"}])

    with pytest.raises(ValueError, match="Search query"):
        index.search("   ")


def test_bm25_rejects_invalid_limit():
    index = BM25Index([{"text": "Python programming"}])

    with pytest.raises(ValueError, match="limit"):
        index.search("Python", limit=0)
