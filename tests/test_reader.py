import httpx
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

    def with_structured_output(self, schema, **kwargs):
        assert schema is EvidenceExtraction
        assert kwargs.get("method") == "json_schema"
        assert kwargs.get("strict") is True
        return self


    async def ainvoke(self, prompt: str):
        return self.response


def create_reader(evidence_items):
    model = FakeModel(EvidenceExtraction(evidence_items=evidence_items))

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



class ForbiddenWebFetcher:
    async def fetch(self, url: str) -> str:
        request = httpx.Request("GET", url)
        response = httpx.Response(403, request=request)
        raise httpx.HTTPStatusError(
            "403 Forbidden",
            request=request,
            response=response,
        )


class NotFoundWebFetcher:
    async def fetch(self, url: str) -> str:
        request = httpx.Request("GET", url)
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError(
            "404 Not Found",
            request=request,
            response=response,
        )


@pytest.mark.asyncio
async def test_reader_uses_exa_snippet_when_website_returns_403():
    snippet = "India solar capacity reached 100 GW in 2025."

    evidence = EvidenceItem(
        claim="India solar capacity reached 100 GW.",
        supporting_quote="India solar capacity reached 100 GW in 2025.",
    )

    model = FakeModel(
        EvidenceExtraction(evidence_items=[evidence])
    )

    source = SourceRecord(
        source_id="source-403",
        title="Solar Energy Report",
        url="https://example.com/report",
        snippet=snippet,
    )

    reader = ReaderAgent(
        model=model,
        web_fetcher=ForbiddenWebFetcher(),
    )

    result = await reader.run(ReadRequest(source=source))

    assert result.text == snippet
    assert result.extraction_status == "partial"
    assert len(result.evidence_items) == 1


@pytest.mark.asyncio
async def test_reader_does_not_use_fallback_when_snippet_is_empty():
    source = SourceRecord(
        source_id="source-empty",
        title="Unavailable Report",
        url="https://example.com/report",
        snippet="",
    )

    reader = ReaderAgent(
        model=FakeModel(EvidenceExtraction(evidence_items=[])),
        web_fetcher=ForbiddenWebFetcher(),
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await reader.run(ReadRequest(source=source))

    assert exc_info.value.response.status_code == 403


@pytest.mark.asyncio
async def test_reader_does_not_use_fallback_for_404():
    source = SourceRecord(
        source_id="source-404",
        title="Missing Report",
        url="https://example.com/report",
        snippet="Some Exa search highlights.",
    )

    reader = ReaderAgent(
        model=FakeModel(EvidenceExtraction(evidence_items=[])),
        web_fetcher=NotFoundWebFetcher(),
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await reader.run(ReadRequest(source=source))

    assert exc_info.value.response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.integration
async def test_reader_processes_live_exa_source_after_403():
    from agents.contracts import SearchTask
    from search.exa_provider import ExaSearchProvider

    provider = ExaSearchProvider()

    task = SearchTask(
        question="What are India's renewable energy developments?",
        query="India renewable energy capacity developments 2025",
    )

    sources = await provider.search(task)

    assert sources, "Exa returned no sources."

    # Select a source with a usable Exa snippet.
    source = next(
        (item for item in sources if item.snippet.strip()),
        None,
    )

    assert source is not None, "No Exa source contained a snippet."

    # Mock model response using an exact quote from the Exa snippet.
    quote = source.snippet.strip()[:100]

    evidence = EvidenceItem(
        claim="The source contains relevant research information.",
        supporting_quote=quote,
    )

    model = FakeModel(
        EvidenceExtraction(evidence_items=[evidence])
    )

    reader = ReaderAgent(
        model=model,
        web_fetcher=ForbiddenWebFetcher(),
    )

    result = await reader.run(
        ReadRequest(
            source=source,
            query="India renewable energy developments",
        )
    )

    assert result.text == source.snippet.strip()
    assert result.extraction_status == "partial"
    assert len(result.evidence_items) == 1
