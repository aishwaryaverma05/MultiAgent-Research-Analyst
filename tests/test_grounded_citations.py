import pytest

from agents.contracts import (
    Citation,
    EvidenceDocument,
    EvidenceItem,
    FactCheckRequest,
    FactCheckResult,
    ResearchReport,
    WriteRequest,
)
from agents.fact_checker import FactCheckerAgent
from agents.writer import WriterAgent


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.received_prompt = None

    async def ainvoke(self, prompt):
        self.received_prompt = prompt
        return self.response


class FakeWriterChatModel:
    def __init__(self, response):
        self.structured_model = FakeStructuredModel(response)

    def with_structured_output(self, schema, **kwargs):
        assert schema is ResearchReport
        return self.structured_model


class FakeFactCheckerChatModel:
    def __init__(self, response):
        self.structured_model = FakeStructuredModel(response)

    def with_structured_output(self, schema, **kwargs):
        assert schema is FactCheckResult
        return self.structured_model


def make_evidence():
    return EvidenceDocument(
        source_id="source-001",
        title="Renewable Energy Report",
        url="https://example.com/report",
        text="Solar energy capacity increased significantly.",
        evidence_items=[
            EvidenceItem(
                claim="Solar energy capacity increased.",
                supporting_quote=("Solar energy capacity increased significantly."),
            )
        ],
        extraction_status="success",
    )


def make_writer_report():
    return ResearchReport(
        title="Renewable Energy Research",
        executive_summary=("Solar energy capacity is growing. [C1]"),
        sections=[
            {
                "heading": "Solar Energy",
                "content": "Solar capacity has increased. [C1]",
            }
        ],
        limitations=["Limited source coverage."],
        citations=[
            Citation(
                citation_id="C1",
                source_id="source-001",
                title="Renewable Energy Report",
                url="https://example.com/report",
            )
        ],
        markdown=("# Renewable Energy Research\n\nSolar energy capacity is growing. [C1]"),
    )


@pytest.mark.asyncio
async def test_writer_to_fact_checker_grounded_citation_flow():
    evidence = make_evidence()

    # Step 1: Writer generates an evidence-grounded report.
    writer_model = FakeWriterChatModel(make_writer_report())
    writer = WriterAgent(model=writer_model)

    report = await writer.run(
        WriteRequest(
            query="Research global renewable energy development",
            evidence=[evidence],
        )
    )

    assert isinstance(report, ResearchReport)
    assert len(report.citations) == 1
    assert report.citations[0].citation_id == "C1"
    assert report.references == ["https://example.com/report"]
    assert "[C1]" in report.markdown

    # Step 2: Pass the generated report directly to Fact Checker.
    checker_model = FakeFactCheckerChatModel(
        FactCheckResult(
            verified_claims=["Solar energy capacity increased."],
            unsupported_claims=[],
            contradictory_claims=[],
            unknown_citations=[],
        )
    )

    checker = FactCheckerAgent(model=checker_model)

    fact_check = await checker.run(
        FactCheckRequest(
            report=report,
            evidence=[evidence],
        )
    )

    # Step 3: Verify the complete grounded citation flow.
    assert isinstance(fact_check, FactCheckResult)
    assert fact_check.verified_claims
    assert fact_check.unknown_citations == []

    # Confirm that the Fact Checker received the generated citation.
    assert "Renewable Energy Research" in (checker_model.structured_model.received_prompt)
    assert "source-001" in (checker_model.structured_model.received_prompt)
    assert "C1" in (checker_model.structured_model.received_prompt)
