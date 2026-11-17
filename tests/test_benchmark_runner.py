
"""Tests for the retrieval benchmark runner."""

import json

import pytest

from evaluation.benchmark_runner import (
    load_corpus,
    run_retrieval_benchmark,
)


def test_load_corpus_success(tmp_path):
    corpus_path = tmp_path / "corpus.json"

    corpus_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "source_id": "doc-001",
                        "text": "Retrieval augmented generation.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    documents = load_corpus(corpus_path)

    assert len(documents) == 1
    assert documents[0]["source_id"] == "doc-001"


def test_load_corpus_rejects_empty_documents(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps({"documents": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-empty"):
        load_corpus(corpus_path)


def test_load_corpus_rejects_missing_source_id(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "documents": [
                    {"text": "A document without an ID."}
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_id"):
        load_corpus(corpus_path)


def test_load_corpus_rejects_duplicate_source_ids(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "documents": [
                    {"source_id": "doc-001", "text": "First document."},
                    {"source_id": "doc-001", "text": "Second document."},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique"):
        load_corpus(corpus_path)


def test_benchmark_rejects_invalid_k():
    with pytest.raises(ValueError, match="greater than zero"):
        run_retrieval_benchmark(k=0)


def test_benchmark_rejects_missing_relevant_source(
    tmp_path,
):
    dataset_path = tmp_path / "dataset.json"
    corpus_path = tmp_path / "corpus.json"

    dataset_path.write_text(
        json.dumps(
            {
                "name": "test",
                "samples": [
                    {
                        "question": "What is RAG?",
                        "reference_answer": "RAG retrieves information.",
                        "reference_contexts": [],
                        "relevant_source_ids": ["missing-001"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    corpus_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "source_id": "doc-001",
                        "text": "A sample retrieval document.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing from corpus"):
        run_retrieval_benchmark(
            dataset_path=dataset_path,
            corpus_path=corpus_path,
        )
