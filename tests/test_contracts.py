
import pytest
from pydantic import ValidationError

from agents.contracts import (
    ResearchPlan,
    ResearchQuestion,
    ResearchRequest,
)


def test_research_request_rejects_short_query():

    with pytest.raises(ValidationError):
        ResearchRequest(query="short")


def test_research_plan_accepts_valid_structure():

    plan = ResearchPlan(
        objective="Assess the impact of retrieval augmented generation.",
        research_questions=[
            ResearchQuestion(
                question="How does RAG reduce hallucinations?",
                rationale="Evaluate evidence grounding.",
            )
        ],
    )

    assert len(plan.research_questions) == 1


def test_research_plan_requires_questions():

    with pytest.raises(ValidationError):
        ResearchPlan(
            objective="A sufficiently long research objective.",
            research_questions=[],
        )
