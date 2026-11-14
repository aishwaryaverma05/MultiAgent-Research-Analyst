
from agents.base import BaseAgent
from agents.contracts import SearchTask, SourceRecord
from search.exa_provider import ExaSearchProvider


class SearchAgent(BaseAgent[SearchTask, list[SourceRecord]]):
    """Agent responsible for executing one research search task."""

    name = "search"

    def __init__(
        self,
        provider: ExaSearchProvider | None = None,
    ) -> None:
        self.provider = provider or ExaSearchProvider()

    async def run(
        self,
        payload: SearchTask,
    ) -> list[SourceRecord]:
        return await self.provider.search(payload)
