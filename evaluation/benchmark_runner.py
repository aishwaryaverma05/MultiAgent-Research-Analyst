
"""Run retrieval quality benchmarks against a separate document corpus."""

import json
from pathlib import Path
from statistics import mean
from typing import Any

from qdrant_client import QdrantClient

from evaluation.dataset import load_dataset
from evaluation.retrieval_metrics import evaluate_retrieval
from retrieval.reranker import create_reranker
from retrieval.retrieval_pipeline import retrieve_documents
from retrieval.semantic_search import index_documents
from retrieval.vector_store import create_qdrant_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "evaluation" / "benchmarks" / "research_benchmark.json"
)

DEFAULT_CORPUS_PATH = (
    PROJECT_ROOT / "evaluation" / "benchmarks" / "research_corpus.json"
)


def load_corpus(path: Path) -> list[dict[str, Any]]:
    """Load and validate the retrieval corpus."""

    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    documents = data.get("documents")

    if not isinstance(documents, list) or not documents:
        raise ValueError("Corpus must contain a non-empty documents list.")

    source_ids = []

    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Every corpus document must be a dictionary.")

        source_id = document.get("source_id")
        text = document.get("text")

        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("Every corpus document must have a source_id.")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Every corpus document must have non-empty text.")

        source_ids.append(source_id)

    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Corpus source_ids must be unique.")

    return documents


def run_retrieval_benchmark(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    client: QdrantClient | None = None,
    embedding_model: Any = None,
    ranker: Any = None,
    k: int = 5,
) -> dict[str, Any]:
    """Evaluate hybrid retrieval and reranking against benchmark samples."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    with dataset_path.open(encoding="utf-8") as file:
        dataset = load_dataset(json.load(file))

    documents = load_corpus(corpus_path)

    corpus_source_ids = {
        document["source_id"] for document in documents
    }

    for sample in dataset.samples:
        missing_ids = (
            set(sample.relevant_source_ids) - corpus_source_ids
        )

        if missing_ids:
            raise ValueError(
                f"Relevant source IDs missing from corpus: {sorted(missing_ids)}"
            )

    qdrant_client = client or create_qdrant_client(in_memory=True)

    try:
        index_documents(
            qdrant_client,
            documents,
            embedding_model=embedding_model,
        )

        reranker = ranker if ranker is not None else create_reranker()

        effective_k = min(k, len(documents))

        sample_results = []

        for sample in dataset.samples:
            retrieved = retrieve_documents(
                qdrant_client,
                sample.question,
                documents,
                embedding_model=embedding_model,
                ranker=reranker,
                candidate_limit=len(documents),
                final_limit=effective_k,
            )

            retrieved_ids = [
                result["document"]["source_id"]
                for result in retrieved
            ]

            metrics = evaluate_retrieval(
                retrieved_ids,
                sample.relevant_source_ids,
                k=effective_k,
            )

            sample_results.append(
                {
                    "question": sample.question,
                    "relevant_source_ids": sample.relevant_source_ids,
                    "retrieved_source_ids": retrieved_ids,
                    "metrics": metrics,
                }
            )

        metric_names = [
            "precision_at_k",
            "recall_at_k",
            "mrr",
            "hit_rate_at_k",
        ]

        average_metrics = {
            metric_name: mean(
                result["metrics"][metric_name]
                for result in sample_results
            )
            for metric_name in metric_names
        }

        return {
            "dataset": dataset.name,
            "total_questions": len(sample_results),
            "k": effective_k,
            "average_metrics": average_metrics,
            "results": sample_results,
        }

    finally:
        if client is None:
            qdrant_client.close()
