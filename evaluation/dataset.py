"""Dataset models and helpers for research evaluation."""

from typing import Any

from pydantic import BaseModel, Field


class EvaluationSample(BaseModel):
    """A single research evaluation example."""

    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    reference_contexts: list[str] = Field(default_factory=list)
    relevant_source_ids: list[str] = Field(default_factory=list)


class EvaluationDataset(BaseModel):
    """Collection of research evaluation examples."""

    name: str = Field(min_length=1)
    description: str = ""
    samples: list[EvaluationSample] = Field(min_length=1)


def load_dataset(data: dict[str, Any]) -> EvaluationDataset:
    """Validate and load an evaluation dataset from a dictionary."""

    return EvaluationDataset.model_validate(data)