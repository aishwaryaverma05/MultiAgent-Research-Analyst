
import asyncio
from types import SimpleNamespace

import pytest

from agents.contracts import SearchTask, SourceRecord
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


@pytest.mark.asyncio
async def test_exa_provider_converts_results():
    provider = ExaSearchProvider(client=FakeExaClient())

    results = await provider.search(make_task("artificial intelligence"))

    assert len(results) == 1
    assert results[0].title == "Result for artificial intelligence"
    assert results[0].metadata["provider"] == "exa"
    assert results[0].relevance_score == 0.95


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

    results = await workers.run(
        [make_task("query one"), make_task("query two")]
    )

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

    results = await workers.run(
        [make_task("query one"), make_task("query two")]
    )

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
        await workers.run(
            [make_task("query one"), make_task("query two")]
        )
