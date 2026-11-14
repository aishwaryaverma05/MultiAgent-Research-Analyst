
from urllib.parse import urlsplit

import httpx

from config import get_settings


class WebFetcher:
    """Fetch HTML content from an HTTP or HTTPS webpage."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._client = client
        self.timeout_seconds = timeout_seconds

    def _get_timeout(self) -> float:
        if self.timeout_seconds is not None:
            return self.timeout_seconds

        return get_settings().request_timeout_seconds

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urlsplit(url)

        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError("Only HTTP and HTTPS URLs are supported.")

        if not parsed.hostname:
            raise ValueError("The source URL is invalid.")

        if parsed.username or parsed.password:
            raise ValueError("URLs containing credentials are not supported.")

    async def fetch(self, url: str) -> str:
        """Download a webpage and return its HTML."""

        self._validate_url(url)
        timeout = self._get_timeout()

        if self._client is not None:
            response = await self._client.get(
                url,
                timeout=timeout,
                follow_redirects=True,
            )
            return self._extract_html(response)

        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            max_redirects=5,
            headers={
                "User-Agent": "MultiAgentResearchAnalyst/1.0",
                "Accept": "text/html,application/xhtml+xml",
            },
        ) as client:
            response = await client.get(url)
            return self._extract_html(response)

    @staticmethod
    def _extract_html(response: httpx.Response) -> str:
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()

        if content_type and not any(
            content_type.startswith(media_type)
            for media_type in (
                "text/html",
                "application/xhtml+xml",
            )
        ):
            raise ValueError(
                f"Expected an HTML webpage, received: {content_type}"
            )

        return response.text
