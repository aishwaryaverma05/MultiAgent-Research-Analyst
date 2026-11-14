
import pytest

from agents.contracts import (
    EvidenceDocument,
    EvidenceItem,
    FactCheckRequest,
    FactCheckResult,
    ResearchReport,
)
from agents.fact_checker import FactCheckerAgent


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
        assert schema is FactCheckResult
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
        references=[
            "https://example.com/report",
            "https://unknown.com/report",
        ],
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


def make_result():
    return FactCheckResult(
        verified_claims=[
            "Solar energy capacity increased."
        ],
        unsupported_claims=[
            "Solar energy will dominate all energy production."
        ],
        contradictory_claims=[],
        unknown_citations=[],
    )


def make_request():
    return FactCheckRequest(
        report=make_report(),
        evidence=[make_evidence()],
    )


@pytest.mark.asyncio
async def test_fact_checker_verifies_report():
    model = FakeChatModel(make_result())
    checker = FactCheckerAgent(model=model)

    result = await checker.run(make_request())

    assert isinstance(result, FactCheckResult)
    assert result.verified_claims
    assert result.unsupported_claims


@pytest.mark.asyncio
async def test_fact_checker_accepts_dictionary_output():
    model = FakeChatModel(make_result().model_dump())
    checker = FactCheckerAgent(model=model)

    result = await checker.run(make_request())

    assert isinstance(result, FactCheckResult)
    assert result.verified_claims


@pytest.mark.asyncio
async def test_fact_checker_identifies_unknown_citations():
    model = FakeChatModel(make_result())
    checker = FactCheckerAgent(model=model)

    result = await checker.run(make_request())

    assert "https://unknown.com/report" in result.unknown_citations
    assert "https://example.com/report" not in result.unknown_citations


@pytest.mark.asyncio
async def test_fact_checker_includes_report_and_evidence_in_prompt():
    model = FakeChatModel(make_result())
    checker = FactCheckerAgent(model=model)

    await checker.run(make_request())

    prompt = model.structured_model.received_prompt

    assert "Renewable Energy Research" in prompt
    assert "Solar energy capacity increased significantly." in prompt
    assert "source-001" in prompt


@pytest.mark.asyncio
async def test_fact_checker_handles_missing_evidence():
    request = FactCheckRequest(
        report=make_report(),
        evidence=[],
    )

    model = FakeChatModel(
        FactCheckResult(
            verified_claims=[],
            unsupported_claims=[
                "Solar energy capacity is growing."
            ],
        )
    )

    checker = FactCheckerAgent(model=model)
    result = await checker.run(request)

    assert result.verified_claims == []
    assert "https://unknown.com/report" in result.unknown_citations
    assert "VERIFIED RESEARCH EVIDENCE" in (
        model.structured_model.received_prompt
    )


@pytest.mark.asyncio
async def test_fact_checker_rejects_unexpected_model_output():
    checker = FactCheckerAgent(
        model=FakeChatModel("invalid response")
    )

    with pytest.raises(TypeError, match="FactChecker expected"):
        await checker.run(make_request())

