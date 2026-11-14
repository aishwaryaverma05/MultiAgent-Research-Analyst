import pytest

from retrieval.hybrid_search import reciprocal_rank_fusion


def test_rrf_combines_semantic_and_bm25_results():
    doc_a = {"text": "Artificial intelligence and machine learning"}
    doc_b = {"text": "Climate change and global warming"}
    doc_c = {"text": "Python programming language"}

    semantic_results = [
        doc_a,
        doc_b,
    ]

    keyword_results = [
        {"document": doc_c},
        {"document": doc_a},
    ]

    results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
    )

    assert len(results) == 3
    assert results[0]["document"] == doc_a


def test_rrf_rewards_documents_found_by_both_methods():
    doc_a = {"text": "Artificial intelligence"}
    doc_b = {"text": "Climate change"}

    semantic_results = [doc_a, doc_b]
    keyword_results = [
        {"document": doc_a},
        {"document": doc_b},
    ]

    results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
    )

    assert results[0]["document"] == doc_a
    assert results[0]["score"] > results[1]["score"]


def test_rrf_preserves_individual_ranks():
    doc = {"text": "Artificial intelligence"}

    results = reciprocal_rank_fusion(
        [doc],
        [{"document": doc}],
    )

    assert results[0]["semantic_rank"] == 1
    assert results[0]["bm25_rank"] == 1


def test_rrf_respects_result_limit():
    documents = [
        {"text": "Document one"},
        {"text": "Document two"},
        {"text": "Document three"},
    ]

    results = reciprocal_rank_fusion(
        documents,
        [{"document": document} for document in documents],
        limit=2,
    )

    assert len(results) == 2


def test_rrf_rejects_invalid_rrf_constant():
    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion([], [], rrf_k=0)


def test_rrf_rejects_invalid_limit():
    with pytest.raises(ValueError, match="limit"):
        reciprocal_rank_fusion([], [], limit=0)
