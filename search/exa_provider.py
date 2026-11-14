
import asyncio
import hashlib
import logging
from typing import Any
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlsplit,
    urlunsplit,
)

from agents.contracts import SearchTask, SourceRecord
from config import get_settings

logger = logging.getLogger(__name__)


class ExaSearchProvider:
    """Search the web using Exa and normalize its results."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    def _get_client(self) -> Any:
        """Create the Exa async client only when required."""
        if self._client is None:
            from exa_py import AsyncExa

            settings = get_settings()
            api_key = settings.require_exa_credentials()

            self._client = AsyncExa(
                api_key=api_key,
                timeout=settings.request_timeout_seconds,
            )

        return self._client

    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize a URL to support reliable duplicate detection."""
        parsed = urlsplit(url.strip())

        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError("Only HTTP and HTTPS URLs are supported.")

        if not parsed.hostname:
            raise ValueError("Search result contains an invalid URL.")

        hostname = parsed.hostname.lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]

        port = parsed.port
        netloc = hostname

        if port and not (
            (parsed.scheme.lower() == "http" and port == 80)
            or (parsed.scheme.lower() == "https" and port == 443)
        ):
            netloc = f"{hostname}:{port}"

        tracking_parameters = {
            "fbclid",
            "gclid",
            "ref",
        }

        query_parameters = [
            (key, value)
            for key, value in parse_qsl(
                parsed.query,
                keep_blank_values=True,
            )
            if not key.lower().startswith("utm_")
            and key.lower() not in tracking_parameters
        ]

        normalized_query = urlencode(sorted(query_parameters))

        return urlunsplit(
            (
                parsed.scheme.lower(),
                netloc,
                parsed.path.rstrip("/") or "/",
                normalized_query,
                "",
            )
        )

    @staticmethod
    def _to_source_record(
        result: Any,
        query: str,
    ) -> SourceRecord:
        """Convert one Exa result into our internal source schema."""
        original_url = getattr(result, "url", None)

        if not isinstance(original_url, str) or not original_url:
            raise ValueError("Exa returned a result without a valid URL.")

        normalized_url = ExaSearchProvider.normalize_url(original_url)

        title = getattr(result, "title", None) or normalized_url

        highlights = getattr(result, "highlights", None) or []
        text = getattr(result, "text", None) or ""

        if highlights:
            snippet = " ".join(str(item) for item in highlights)
        else:
            snippet = str(text)

        score = getattr(result, "score", None)

        metadata = {
            "provider": "exa",
            "search_query": query,
        }

        published_date = getattr(result, "published_date", None)
        author = getattr(result, "author", None)

        if published_date:
            metadata["published_date"] = str(published_date)

        if author:
            metadata["author"] = str(author)

        source_id = hashlib.sha256(
            normalized_url.encode("utf-8")
        ).hexdigest()[:16]

        return SourceRecord(
            source_id=source_id,
            title=str(title),
            url=normalized_url,
            snippet=snippet,
            relevance_score=(
                float(score) if score is not None else None
            ),
            metadata=metadata,
        )

    async def search(
        self,
        task: SearchTask,
        num_results: int = 5,
    ) -> list[SourceRecord]:
        """Execute one Exa search task."""
        if not 1 <= num_results <= 20:
            raise ValueError("num_results must be between 1 and 20.")

        settings = get_settings()
        client = self._get_client()

        response = await asyncio.wait_for(
            client.search(
                task.query,
                num_results=num_results,
                contents={"highlights": True},
                timeout=settings.request_timeout_seconds,
            ),
            timeout=settings.request_timeout_seconds,
        )

        records: list[SourceRecord] = []

        for result in response.results:
            try:
                records.append(
                    self._to_source_record(result, task.query)
                )
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping an invalid Exa result for query: %s",
                    task.query,
                    exc_info=True,
                )

        return records
