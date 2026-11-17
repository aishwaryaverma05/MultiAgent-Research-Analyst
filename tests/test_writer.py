import pytest

from agents.contracts import (
    Citation,
    EvidenceDocument,
    EvidenceItem,
    ResearchReport,
    WriteRequest,
)
from agents.writer import WriterAgent


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

    def with_structured_output(self, schema, **kwargs):
        assert schema is ResearchReport
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


def make_report():
    return ResearchReport(
        title="Renewable Energy Research",
        executive_summary="Solar energy capacity is growing. [C1]",
        sections=[
            {
                "heading": "Solar Energy",
                "content": "Solar capacity has increased. [C1]",
            }
        ],
        limitations=["Limited source coverage."],
        references=["https://example.com/report"],
        citations=[
            Citation(
                citation_id="C1",
                source_id="source-001",
                title="Renewable Energy Report",
                url="https://example.com/report",
            )
        ],
        markdown=("# Renewable Energy Research\n\nSolar capacity is growing. [C1]"),
    )


def make_request():
    return WriteRequest(
        query="Research global renewable energy development",
        evidence=[make_evidence()],
    )


@pytest.mark.asyncio
async def test_writer_generates_research_report():
    model = FakeChatModel(make_report())
    writer = WriterAgent(model=model)

    result = await writer.run(make_request())

    assert isinstance(result, ResearchReport)
    assert result.title == "Renewable Energy Research"
    assert result.markdown.startswith("#")
    assert len(result.sections) == 1
    assert result.citations[0].citation_id == "C1"
    assert result.references == ["https://example.com/report"]


@pytest.mark.asyncio
async def test_writer_accepts_dictionary_output():
    model = FakeChatModel(make_report().model_dump())
    writer = WriterAgent(model=model)

    result = await writer.run(make_request())

    assert isinstance(result, ResearchReport)
    assert result.references == ["https://example.com/report"]
    assert result.citations[0].source_id == "source-001"


@pytest.mark.asyncio
async def test_writer_rejects_missing_evidence():
    request = WriteRequest(
        query="Research global renewable energy development",
        evidence=[],
    )

    writer = WriterAgent(model=FakeChatModel(make_report()))

    with pytest.raises(ValueError, match="without verified evidence"):
        await writer.run(request)


@pytest.mark.asyncio
async def test_writer_rejects_empty_markdown():
    report = make_report().model_copy(update={"markdown": ""})

    writer = WriterAgent(model=FakeChatModel(report))

    with pytest.raises(ValueError, match="empty Markdown"):
        await writer.run(make_request())


@pytest.mark.asyncio
async def test_writer_includes_revision_feedback():
    model = FakeChatModel(make_report())
    writer = WriterAgent(model=model)

    request = make_request().model_copy(
        update={
            "previous_report": "Previous research report",
            "revision_feedback": ["Improve the executive summary."],
        }
    )

    await writer.run(request)

    assert "Previous research report" in (model.structured_model.received_prompt)
    assert "Improve the executive summary." in (model.structured_model.received_prompt)


@pytest.mark.asyncio
async def test_writer_rejects_unknown_citation_id():
    report = make_report().model_copy(
        update={
            "citations": [
                Citation(
                    citation_id="C99",
                    source_id="source-001",
                    title="Renewable Energy Report",
                    url="https://example.com/report",
                )
            ],
            "markdown": "# Report\n\nSolar capacity increased. [C99]",
        }
    )

    writer = WriterAgent(model=FakeChatModel(report))

    with pytest.raises(ValueError, match="unknown citation ID"):
        await writer.run(make_request())


@pytest.mark.asyncio
async def test_writer_rejects_mismatched_citation_metadata():
    report = make_report().model_copy(
        update={
            "citations": [
                Citation(
                    citation_id="C1",
                    source_id="source-001",
                    title="Fake Report",
                    url="https://fake-example.com/report",
                )
            ]
        }
    )

    writer = WriterAgent(model=FakeChatModel(report))

    with pytest.raises(ValueError, match="metadata does not match"):
        await writer.run(make_request())


@pytest.mark.asyncio
async def test_writer_rejects_markdown_without_citation_marker():
    report = make_report().model_copy(update={"markdown": "# Report\n\nSolar capacity increased."})

    writer = WriterAgent(model=FakeChatModel(report))

    with pytest.raises(ValueError, match="no inline citation markers"):
        await writer.run(make_request())


@pytest.mark.asyncio
async def test_writer_rejects_missing_citation_records():
    report = make_report().model_copy(update={"citations": []})

    writer = WriterAgent(model=FakeChatModel(report))

    with pytest.raises(ValueError, match="generated no citations"):
        await writer.run(make_request())
