
import asyncio
from types import SimpleNamespace

import pytest

from agents.contracts import SearchTask, SourceRecord
from agents.search_agent import SearchAgent
from search.exa_provider import ExaSearchProvider
from search.workers import SearchBatchError, SearchWorkers


def make_task(query: str) -> SearchTask:
    return SearchTask(
        question=f"Research {query} in detail",
        query=query,
    )


class FakeExaClient:
    async def search(self, query, **kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    url="https://www.example.com/article?utm_source=test",
                    title=f"Result for {query}",
                    highlights=["Important research evidence"],
                    score=0.95,
                    published_date="2026-01-01",
                    author="Researcher",
                )
            ]
        )


def make_source(url: str, title: str) -> SourceRecord:
    return SourceRecord(
        source_id=title,
        title=title,
        url=url,
        snippet="Test evidence",
        metadata={"provider": "test"},
    )


def test_normalize_url_removes_tracking_parameters():
    normalized = ExaSearchProvider.normalize_url(
        "https://www.example.com/article/?utm_source=google&fbclid=123"
    )

    assert normalized == "https://example.com/article"


def test_normalize_url_rejects_invalid_scheme():
    with pytest.raises(ValueError):
        ExaSearchProvider.normalize_url("ftp://example.com/file")


def test_normalize_url_rejects_invalid_port():
    with pytest.raises(ValueError, match="invalid URL"):
        ExaSearchProvider.normalize_url("https://example.com:invalid/article")



@pytest.mark.asyncio
async def test_exa_provider_converts_results():
    provider = ExaSearchProvider(client=FakeExaClient())

    results = await provider.search(make_task("artificialintelligence"))

    assert len(results) == 1
    assert results[0].title == "Result for artificialintelligence"
    assert results[0].metadata["provider"] == "exa"
    assert results[0].relevance_score == 0.95
    assert results[0].snippet == "Important research evidence"



@pytest.mark.asyncio
async def test_exa_provider_uses_text_when_highlights_are_empty():
    class TextFallbackClient:
        async def search(self, query, **kwargs):
            return SimpleNamespace(
                results=[
                    SimpleNamespace(
                        url="https://example.com/article",
                        title="Fallback article",
                        highlights=[],
                        text="Fallback research evidence from Exa.",
                        score=0.90,
                    )
                ]
            )

    provider = ExaSearchProvider(client=TextFallbackClient())

    results = await provider.search(make_task("test query"))

    assert len(results) == 1
    assert results[0].snippet == "Fallback research evidence from Exa."


@pytest.mark.asyncio
@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
async def test_exa_provider_rejects_non_finite_scores(score):
    class InvalidScoreClient:
        async def search(self, query, **kwargs):
            return SimpleNamespace(
                results=[
                    SimpleNamespace(
                        url="https://example.com/article",
                        title="Test article",
                        score=score,
                    )
                ]
            )

    provider = ExaSearchProvider(client=InvalidScoreClient())

    results = await provider.search(make_task("test query"))

    assert results == []


class FakeSearchAgent:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query

    async def run(self, task):
        await asyncio.sleep(0.01)

        result = self.results_by_query[task.query]

        if isinstance(result, Exception):
            raise result

        return result


@pytest.mark.asyncio
async def test_workers_deduplicate_sources():
    shared_source = make_source(
        "https://example.com/article",
        "Shared article",
    )

    agent = FakeSearchAgent(
        {
            "query one": [shared_source],
            "query two": [shared_source],
        }
    )

    workers = SearchWorkers(search_agent=agent)

    results = await workers.run([make_task("query one"), make_task("query two")])

    assert len(results) == 1
    assert results[0].title == "Shared article"


@pytest.mark.asyncio
async def test_workers_continue_after_partial_failure():
    valid_source = make_source(
        "https://example.com/valid",
        "Valid result",
    )

    agent = FakeSearchAgent(
        {
            "query one": [valid_source],
            "query two": RuntimeError("Simulated search failure"),
        }
    )

    workers = SearchWorkers(search_agent=agent)

    results = await workers.run([make_task("query one"), make_task("query two")])

    assert len(results) == 1
    assert results[0].title == "Valid result"


@pytest.mark.asyncio
async def test_workers_raise_when_all_tasks_fail():
    agent = FakeSearchAgent(
        {
            "query one": RuntimeError("Failure one"),
            "query two": RuntimeError("Failure two"),
        }
    )

    workers = SearchWorkers(search_agent=agent)

    with pytest.raises(SearchBatchError):
        await workers.run([make_task("query one"), make_task("query two")])



@pytest.mark.asyncio
async def test_search_agent_preserves_source_snippet():
    expected_source = SourceRecord(
        source_id="test-source",
        title="Research article",
        url="https://example.com/article",
        snippet="Important Exa research highlights.",
        metadata={"provider": "exa"},
    )

    class FakeProvider:
        async def search(self, payload):
            return [expected_source]

    agent = SearchAgent(provider=FakeProvider())

    results = await agent.run(make_task("test query"))

    assert len(results) == 1
    assert results[0].snippet == "Important Exa research highlights."
    assert results[0].source_id == "test-source"
