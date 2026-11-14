
import asyncio
import logging

from agents.base import BaseAgent
from agents.contracts import (
    EvidenceDocument,
    ReadRequest,
    ResearchPlan,
    ResearchRequest,
    SourceRecord,
    SupervisorResult,
)
from agents.planner import PlannerAgent
from agents.reader_agent import ReaderAgent
from search.workers import SearchWorkers

logger = logging.getLogger(__name__)


class SupervisorAgent(
    BaseAgent[ResearchRequest, SupervisorResult]
):
    """Coordinate planning, parallel search and evidence extraction."""

    name = "supervisor"

    def __init__(
        self,
        planner: PlannerAgent | None = None,
        search_workers: SearchWorkers | None = None,
        reader: ReaderAgent | None = None,
        max_reader_concurrency: int = 5,
        max_sources_to_read: int = 10,
    ) -> None:
        if max_reader_concurrency < 1:
            raise ValueError(
                "max_reader_concurrency must be at least 1."
            )

        if max_sources_to_read < 1:
            raise ValueError(
                "max_sources_to_read must be at least 1."
            )

        self.planner = planner or PlannerAgent()
        self.search_workers = search_workers or SearchWorkers()
        self.reader = reader or ReaderAgent()

        self.max_reader_concurrency = max_reader_concurrency
        self.max_sources_to_read = max_sources_to_read

    async def _read_source(
        self,
        source: SourceRecord,
        semaphore: asyncio.Semaphore,
    ) -> EvidenceDocument:
        """Read one source while respecting the concurrency limit."""

        async with semaphore:
            return await self.reader.run(
                ReadRequest(source=source)
            )

    async def _read_sources(
        self,
        sources: list[SourceRecord],
    ) -> tuple[list[EvidenceDocument], int, list[str]]:
        """Read sources concurrently and preserve partial results."""

        semaphore = asyncio.Semaphore(
            self.max_reader_concurrency
        )

        results = await asyncio.gather(
            *(
                self._read_source(source, semaphore)
                for source in sources
            ),
            return_exceptions=True,
        )

        evidence: list[EvidenceDocument] = []
        errors: list[str] = []
        failed_readers = 0

        for source, result in zip(sources, results, strict=True):
            if isinstance(result, asyncio.CancelledError):
                raise result

            if isinstance(result, BaseException):
                if not isinstance(result, Exception):
                    raise result

                failed_readers += 1
                error = (
                    f"Reader failed for {source.url}: {result}"
                )
                errors.append(error)

                logger.error(error)
                continue

            evidence.append(result)

            if result.extraction_status == "failed":
                failed_readers += 1
                errors.append(
                    f"Evidence extraction failed for {source.url}"
                )

            elif result.extraction_status == "partial":
                errors.append(
                    f"Evidence extraction was partial for {source.url}"
                )

        return evidence, failed_readers, errors

    async def run(
        self,
        payload: ResearchRequest,
    ) -> SupervisorResult:
        """Execute the complete supervised research preparation."""

        # Step 1: Generate a research plan.
        plan: ResearchPlan = await self.planner.run(payload)

        if not plan.search_tasks:
            raise ValueError(
                "Research plan contains no executable search tasks."
            )

        # Step 2: Execute search tasks concurrently.
        search_result = await self.search_workers.run_detailed(
            plan.search_tasks
        )

        errors = [
            (
                f"Search task {failure.task_index + 1} failed "
                f"for query '{failure.query}': {failure.error}"
            )
            for failure in search_result.failures
        ]

        sources = search_result.sources

        # Step 3: Select a bounded number of sources for reading.
        selected_sources = sources[:self.max_sources_to_read]

        skipped_sources = len(sources) - len(selected_sources)

        if skipped_sources:
            errors.append(
                f"{skipped_sources} sources skipped because "
                f"the reader limit is {self.max_sources_to_read}."
            )

        # Step 4: Extract evidence from selected sources.
        evidence, failed_readers, reader_errors = (
            await self._read_sources(selected_sources)
        )

        errors.extend(reader_errors)

        # Step 5: Determine the overall execution status.
        successful_readers = (
            len(selected_sources) - failed_readers
        )

        if (
            search_result.successful_tasks == 0
            or not sources
            or (
                selected_sources
                and successful_readers == 0
            )
        ):
            status = "failed"

        elif errors:
            status = "partial"

        else:
            status = "success"

        logger.info(
            "Supervisor completed: status=%s, "
            "searches=%s/%s, sources=%s, "
            "readers=%s/%s",
            status,
            search_result.successful_tasks,
            search_result.total_tasks,
            len(sources),
            successful_readers,
            len(selected_sources),
        )

        return SupervisorResult(
            plan=plan,
            sources=sources,
            evidence=evidence,
            errors=errors,
            status=status,
            total_search_tasks=search_result.total_tasks,
            successful_search_tasks=search_result.successful_tasks,
            failed_search_tasks=len(search_result.failures),
            total_sources=len(sources),
            attempted_readers=len(selected_sources),
            failed_readers=failed_readers,
        )
