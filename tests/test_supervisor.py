
import asyncio

import pytest

from agents.contracts import (
    EvidenceDocument,
    EvidenceItem,
    ResearchPlan,
    ResearchQuestion,
    ResearchRequest,
    SearchBatchResult,
    SearchTask,
    SearchTaskFailure,
    SourceRecord,
)
from agents.supervisor import SupervisorAgent


def make_plan() -> ResearchPlan:
    return ResearchPlan(
        objective="Research renewable energy development",
        research_questions=[
            ResearchQuestion(
                question="What is the current renewable energy capacity?"
            )
        ],
        search_tasks=[
            SearchTask(
                question="Investigate renewable energy capacity",
                query="global renewable energy capacity",
            ),
            SearchTask(
                question="Investigate renewable energy growth",
                query="global renewable energy growth",
            ),
        ],
    )


def make_source(index: int) -> SourceRecord:
    return SourceRecord(
        source_id=f"source-{index}",
        title=f"Renewable Energy Source {index}",
        url=f"https://example.com/source-{index}",
        snippet="Renewable energy research information.",
    )


def make_evidence(source: SourceRecord) -> EvidenceDocument:
    return EvidenceDocument(
        source_id=source.source_id,
        url=source.url,
        title=source.title,
        text="Renewable energy capacity continues to grow.",
        evidence_items=[
            EvidenceItem(
                claim="Renewable energy capacity continues to grow.",
                supporting_quote=(
                    "Renewable energy capacity continues to grow."
                ),
            )
        ],
        extraction_status="success",
    )


class FakePlanner:
    def __init__(self, plan: ResearchPlan | None = None):
        self.plan = plan or make_plan()

    async def run(self, payload: ResearchRequest) -> ResearchPlan:
        return self.plan


class FakeSearchWorkers:
    def __init__(self, result: SearchBatchResult):
        self.result = result

    async def run_detailed(
        self,
        tasks: list[SearchTask],
    ) -> SearchBatchResult:
        return self.result


class FakeReader:
    def __init__(
        self,
        failed_urls: set[str] | None = None,
        delay: float = 0,
    ):
        self.failed_urls = failed_urls or set()
        self.delay = delay
        self.active_readers = 0
        self.max_active_readers = 0

    async def run(self, payload):
        source = payload.source

        self.active_readers += 1
        self.max_active_readers = max(
            self.max_active_readers,
            self.active_readers,
        )

        try:
            if self.delay:
                await asyncio.sleep(self.delay)

            if source.url in self.failed_urls:
                raise RuntimeError("Simulated reader failure")

            return make_evidence(source)

        finally:
            self.active_readers -= 1


def make_request() -> ResearchRequest:
    return ResearchRequest(
        query="Research global renewable energy development"
    )


@pytest.mark.asyncio
async def test_supervisor_success():
    sources = [make_source(1), make_source(2)]

    search_result = SearchBatchResult(
        sources=sources,
        total_tasks=2,
        successful_tasks=2,
    )

    reader = FakeReader()

    supervisor = SupervisorAgent(
        planner=FakePlanner(),
        search_workers=FakeSearchWorkers(search_result),
        reader=reader,
    )

    result = await supervisor.run(make_request())

    assert result.status == "success"
    assert result.total_search_tasks == 2
    assert result.successful_search_tasks == 2
    assert result.failed_search_tasks == 0
    assert result.total_sources == 2
    assert result.attempted_readers == 2
    assert result.failed_readers == 0
    assert len(result.evidence) == 2
    assert result.errors == []


@pytest.mark.asyncio
async def test_supervisor_preserves_partial_search_failure():
    sources = [make_source(1)]

    search_result = SearchBatchResult(
        sources=sources,
        failures=[
            SearchTaskFailure(
                task_index=1,
                query="global renewable energy growth",
                error="Simulated search failure",
            )
        ],
        total_tasks=2,
        successful_tasks=1,
    )

    supervisor = SupervisorAgent(
        planner=FakePlanner(),
        search_workers=FakeSearchWorkers(search_result),
        reader=FakeReader(),
    )

    result = await supervisor.run(make_request())

    assert result.status == "partial"
    assert result.successful_search_tasks == 1
    assert result.failed_search_tasks == 1
    assert len(result.evidence) == 1
    assert any("Simulated search failure" in error
               for error in result.errors)


@pytest.mark.asyncio
async def test_supervisor_handles_reader_failure():
    sources = [make_source(1), make_source(2)]

    search_result = SearchBatchResult(
        sources=sources,
        total_tasks=2,
        successful_tasks=2,
    )

    reader = FakeReader(
        failed_urls={"https://example.com/source-2"}
    )

    supervisor = SupervisorAgent(
        planner=FakePlanner(),
        search_workers=FakeSearchWorkers(search_result),
        reader=reader,
    )

    result = await supervisor.run(make_request())

    assert result.status == "partial"
    assert result.attempted_readers == 2
    assert result.failed_readers == 1
    assert len(result.evidence) == 1
    assert len(result.errors) == 1


@pytest.mark.asyncio
async def test_supervisor_returns_failed_when_all_searches_fail():
    search_result = SearchBatchResult(
        sources=[],
        failures=[
            SearchTaskFailure(
                task_index=0,
                query="global renewable energy capacity",
                error="API unavailable",
            ),
            SearchTaskFailure(
                task_index=1,
                query="global renewable energy growth",
                error="API unavailable",
            ),
        ],
        total_tasks=2,
        successful_tasks=0,
    )

    supervisor = SupervisorAgent(
        planner=FakePlanner(),
        search_workers=FakeSearchWorkers(search_result),
        reader=FakeReader(),
    )

    result = await supervisor.run(make_request())

    assert result.status == "failed"
    assert result.successful_search_tasks == 0
    assert result.failed_search_tasks == 2
    assert result.total_sources == 0
    assert result.attempted_readers == 0
    assert result.evidence == []


@pytest.mark.asyncio
async def test_supervisor_respects_reader_concurrency_limit():
    sources = [make_source(index) for index in range(5)]

    search_result = SearchBatchResult(
        sources=sources,
        total_tasks=2,
        successful_tasks=2,
    )

    reader = FakeReader(delay=0.02)

    supervisor = SupervisorAgent(
        planner=FakePlanner(),
        search_workers=FakeSearchWorkers(search_result),
        reader=reader,
        max_reader_concurrency=2,
    )

    result = await supervisor.run(make_request())

    assert result.status == "success"
    assert len(result.evidence) == 5
    assert reader.max_active_readers <= 2
    assert reader.max_active_readers == 2


@pytest.mark.asyncio
async def test_supervisor_respects_maximum_source_limit():
    sources = [make_source(index) for index in range(4)]

    search_result = SearchBatchResult(
        sources=sources,
        total_tasks=2,
        successful_tasks=2,
    )

    supervisor = SupervisorAgent(
        planner=FakePlanner(),
        search_workers=FakeSearchWorkers(search_result),
        reader=FakeReader(),
        max_sources_to_read=2,
    )

    result = await supervisor.run(make_request())

    assert result.status == "partial"
    assert result.total_sources == 4
    assert result.attempted_readers == 2
    assert len(result.evidence) == 2
    assert any("2 sources skipped" in error
               for error in result.errors)
