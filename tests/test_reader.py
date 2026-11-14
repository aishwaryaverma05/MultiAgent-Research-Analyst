
import pytest

from agents.contracts import (
    EvidenceDocument,
    EvidenceExtraction,
    EvidenceItem,
    ReadRequest,
    SourceRecord,
)
from agents.reader_agent import ReaderAgent

SOURCE_TEXT = (
    "Solar energy capacity increased significantly during the year. "
    "Renewable energy adoption continues to grow."
)


class FakeWebFetcher:
    async def fetch(self, url: str) -> str:
        return "<html><body>Mock webpage</body></html>"


class FakeContentExtractor:
    def extract(self, html: str) -> str:
        return SOURCE_TEXT


class FakeModel:
    def __init__(self, response):
        self.response = response

    def with_structured_output(self, schema):
        assert schema is EvidenceExtraction
        return self

    async def ainvoke(self, prompt: str):
        return self.response


def create_reader(evidence_items):
    model = FakeModel(
        EvidenceExtraction(evidence_items=evidence_items)
    )

    return ReaderAgent(
        model=model,
        web_fetcher=FakeWebFetcher(),
        content_extractor=FakeContentExtractor(),
    )


def create_request():
    return ReadRequest(
        source=SourceRecord(
            source_id="source-001",
            title="Renewable Energy Report",
            url="https://example.com/report",
            snippet="A report about renewable energy.",
        )
    )


@pytest.mark.asyncio
async def test_reader_accepts_supported_evidence():
    evidence = EvidenceItem(
        claim="Solar energy capacity increased.",
        supporting_quote="Solar energy capacity increased significantly",
    )

    reader = create_reader([evidence])
    result = await reader.run(create_request())

    assert isinstance(result, EvidenceDocument)
    assert result.extraction_status == "success"
    assert len(result.evidence_items) == 1
    assert result.evidence_items[0].claim == evidence.claim


@pytest.mark.asyncio
async def test_reader_rejects_unsupported_quotes():
    evidence = EvidenceItem(
        claim="Solar energy production doubled.",
        supporting_quote="Solar production doubled across all countries",
    )

    reader = create_reader([evidence])
    result = await reader.run(create_request())

    assert result.extraction_status == "failed"
    assert result.evidence_items == []


@pytest.mark.asyncio
async def test_reader_marks_mixed_evidence_as_partial():
    supported = EvidenceItem(
        claim="Solar energy capacity increased.",
        supporting_quote="Solar energy capacity increased significantly",
    )

    unsupported = EvidenceItem(
        claim="Wind energy production doubled.",
        supporting_quote="Wind energy production doubled worldwide",
    )

    reader = create_reader([supported, unsupported])
    result = await reader.run(create_request())

    assert result.extraction_status == "partial"
    assert len(result.evidence_items) == 1


@pytest.mark.asyncio
async def test_reader_marks_empty_extraction_as_success():
    reader = create_reader([])
    result = await reader.run(create_request())

    assert result.extraction_status == "success"
    assert result.evidence_items == []
