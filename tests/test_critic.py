
import pytest

from agents.contracts import (
    CriticEvaluation,
    CriticRequest,
    EvidenceDocument,
    EvidenceItem,
    ResearchReport,
)
from agents.critic import CriticAgent


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.received_prompt = None

    async def ainvoke(self, prompt):
        self.received_prompt = prompt
        return self.response


class FakeChatModel:
    def __init__(self, response):
        self.structured_model = FakeStructuredModel(response)

    def with_structured_output(self, schema):
        assert schema is CriticEvaluation
        return self.structured_model


def make_report():
    return ResearchReport(
        title="Renewable Energy Research",
        executive_summary="Solar energy capacity is growing.",
        sections=[
            {
                "heading": "Solar Energy",
                "content": "Solar capacity has increased.",
            }
        ],
        limitations=["Limited source coverage."],
        references=["https://example.com/report"],
        markdown="# Renewable Energy Research\n\nSolar capacity is growing.",
    )


def make_evidence():
    return EvidenceDocument(
        source_id="source-001",
        title="Renewable Energy Report",
        url="https://example.com/report",
        text="Solar energy capacity increased significantly.",
        evidence_items=[
            EvidenceItem(
                claim="Solar energy capacity increased.",
                supporting_quote=(
                    "Solar energy capacity increased significantly."
                ),
            )
        ],
        extraction_status="success",
    )


def make_evaluation():
    return CriticEvaluation(
        overall_score=8.5,
        dimension_scores={
            "factual_accuracy": 9.0,
            "evidence_quality": 8.0,
            "citation_quality": 8.0,
            "completeness": 8.0,
            "clarity": 9.0,
        },
        unsupported_claims=[],
        citation_issues=[],
        missing_information=["More recent data is needed."],
        revision_instructions=[
            "Include more recent renewable energy statistics."
        ],
    )


def make_request():
    return CriticRequest(
        report=make_report(),
        evidence=[make_evidence()],
    )


@pytest.mark.asyncio
async def test_critic_evaluates_research_report():
    model = FakeChatModel(make_evaluation())
    critic = CriticAgent(model=model)

    result = await critic.run(make_request())

    assert isinstance(result, CriticEvaluation)
    assert result.overall_score == 8.5
    assert result.dimension_scores["clarity"] == 9.0
    assert result.revision_instructions


@pytest.mark.asyncio
async def test_critic_accepts_dictionary_output():
    model = FakeChatModel(make_evaluation().model_dump())
    critic = CriticAgent(model=model)

    result = await critic.run(make_request())

    assert isinstance(result, CriticEvaluation)
    assert result.overall_score == 8.5


@pytest.mark.asyncio
async def test_critic_handles_missing_evidence():
    request = CriticRequest(
        report=make_report(),
        evidence=[],
    )

    model = FakeChatModel(make_evaluation())
    critic = CriticAgent(model=model)

    result = await critic.run(request)

    assert isinstance(result, CriticEvaluation)
    assert "VERIFIED RESEARCH EVIDENCE" in (
        model.structured_model.received_prompt
    )
    assert "[]" in model.structured_model.received_prompt


@pytest.mark.asyncio
async def test_critic_includes_report_and_evidence_in_prompt():
    model = FakeChatModel(make_evaluation())
    critic = CriticAgent(model=model)

    await critic.run(make_request())

    prompt = model.structured_model.received_prompt

    assert "Renewable Energy Research" in prompt
    assert "Solar energy capacity increased significantly." in prompt
    assert "source-001" in prompt


@pytest.mark.asyncio
async def test_critic_rejects_invalid_dimension_score():
    evaluation = make_evaluation().model_copy(
        update={
            "dimension_scores": {
                "factual_accuracy": 12.0,
            }
        }
    )

    critic = CriticAgent(model=FakeChatModel(evaluation))

    with pytest.raises(
        ValueError,
        match="dimension scores outside 0-10",
    ):
        await critic.run(make_request())


@pytest.mark.asyncio
async def test_critic_rejects_unexpected_model_output():
    critic = CriticAgent(model=FakeChatModel("invalid response"))

    with pytest.raises(TypeError, match="Critic expected"):
        await critic.run(make_request())
