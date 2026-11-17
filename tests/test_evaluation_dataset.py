import pytest
from pydantic import ValidationError

from evaluation.dataset import (
    EvaluationDataset,
    EvaluationSample,
    load_dataset,
)


def make_sample():
    return {
        "question": "What is retrieval-augmented generation?",
        "reference_answer": (
            "RAG combines information retrieval with generation."
        ),
        "reference_contexts": [
            "RAG retrieves relevant external information before generating an answer."
        ],
        "relevant_source_ids": ["source-001"],
    }


def test_evaluation_sample_creation():
    sample = EvaluationSample(**make_sample())

    assert sample.question == "What is retrieval-augmented generation?"
    assert sample.relevant_source_ids == ["source-001"]


def test_evaluation_dataset_creation():
    dataset = EvaluationDataset(
        name="research-benchmark",
        description="Basic research evaluation examples",
        samples=[make_sample()],
    )

    assert dataset.name == "research-benchmark"
    assert len(dataset.samples) == 1


def test_load_dataset_from_dictionary():
    data = {
        "name": "research-benchmark",
        "description": "Test benchmark",
        "samples": [make_sample()],
    }

    dataset = load_dataset(data)

    assert isinstance(dataset, EvaluationDataset)
    assert len(dataset.samples) == 1


def test_empty_question_is_rejected():
    data = make_sample()
    data["question"] = ""

    with pytest.raises(ValidationError):
        EvaluationSample(**data)


def test_empty_dataset_is_rejected():
    with pytest.raises(ValidationError):
        EvaluationDataset(name="empty-benchmark", samples=[])