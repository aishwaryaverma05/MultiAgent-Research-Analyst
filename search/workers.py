
import asyncio
import logging
from collections.abc import Sequence

from agents.contracts import (
    SearchBatchResult,
    SearchTask,
    SearchTaskFailure,
    SourceRecord,
)
from agents.search_agent import SearchAgent

logger = logging.getLogger(__name__)


class SearchBatchError(RuntimeError):
    """Raised when every search task fails."""


class SearchWorkers:
    """Execute multiple search tasks concurrently."""

    def __init__(
        self,
        search_agent: SearchAgent | None = None,
        max_concurrency: int = 5,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")

        self.search_agent = search_agent or SearchAgent()
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_task(
        self,
        task: SearchTask,
    ) -> list[SourceRecord]:
        async with self.semaphore:
            return await self.search_agent.run(task)

    async def run_detailed(
        self,
        tasks: Sequence[SearchTask],
    ) -> SearchBatchResult:
        """Execute tasks and return sources, failures and execution counts."""

        if not tasks:
            raise ValueError("At least one search task is required.")

        results = await asyncio.gather(
            *(self._run_task(task) for task in tasks),
            return_exceptions=True,
        )

        unique_sources: dict[str, SourceRecord] = {}
        failures: list[SearchTaskFailure] = []
        successful_tasks = 0

        for index, result in enumerate(results):
            if isinstance(result, asyncio.CancelledError):
                raise result

            if isinstance(result, BaseException):
                if not isinstance(result, Exception):
                    raise result

                task = tasks[index]

                failures.append(
                    SearchTaskFailure(
                        task_index=index,
                        query=task.query,
                        error=str(result),
                    )
                )

                logger.error(
                    "Search task %s failed: %s",
                    index + 1,
                    result,
                    exc_info=(
                        type(result),
                        result,
                        result.__traceback__,
                    ),
                )
                continue

            successful_tasks += 1

            for source in result:
                unique_sources.setdefault(source.url, source)

        batch_result = SearchBatchResult(
            sources=list(unique_sources.values()),
            failures=failures,
            total_tasks=len(tasks),
            successful_tasks=successful_tasks,
        )

        logger.info(
            "Search completed: %s tasks, %s successes, "
            "%s failures, %s unique sources",
            batch_result.total_tasks,
            batch_result.successful_tasks,
            len(batch_result.failures),
            len(batch_result.sources),
        )

        return batch_result

    async def run(
        self,
        tasks: Sequence[SearchTask],
    ) -> list[SourceRecord]:
        """Backward-compatible method returning only successful sources."""

        result = await self.run_detailed(tasks)

        if result.successful_tasks == 0:
            raise SearchBatchError(
                f"All {result.total_tasks} search tasks failed."
            )

        return result.sources
