
import asyncio
import html
import io
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Final
from urllib.parse import urlsplit

import httpx
import pymupdf
from pypdf import PdfReader

from config import get_settings


class WebFetcher:
    """Fetch HTML pages and extract text from text-based PDF documents."""

    RETRYABLE_STATUS_CODES: Final = {429, 500, 502, 503, 504}
    MAX_RETRIES: Final = 2

    HEADERS: Final = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36 "
            "MultiAgentResearchAnalyst/1.0"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/pdf;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float | None = None,
        max_retries: int = 2,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        self._client = client
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

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
            raise RuntimeError("Web fetching failed after all retry attempts.")

    async def fetch(self, url: str) -> str:
        """Fetch a webpage or PDF and return extractable HTML content."""

        self._validate_url(url)

        if self._client is not None:
            return await self._fetch_with_client(self._client, url)

        async with httpx.AsyncClient(
            timeout=self._get_timeout(),
            follow_redirects=True,
            max_redirects=5,
        ) as client:
            return await self._fetch_with_client(client, url)

    async def _fetch_with_client(
        self,
        client: httpx.AsyncClient,
        url: str,
    ) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                response = await client.get(
                    url,
                    timeout=self._get_timeout(),
                    follow_redirects=True,
                    headers=self.HEADERS,
                )

                if (
                    response.status_code in self.RETRYABLE_STATUS_CODES
                    and attempt < self.max_retries
                ):
                    await asyncio.sleep(self._get_retry_delay(response, attempt))
                    continue

                return self._extract_html(response)

            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt >= self.max_retries:
                    raise

                await asyncio.sleep(2 ** attempt)

        raise RuntimeError("Web fetching failed after all retry attempts.")

    @staticmethod
    def _get_retry_delay(
        response: httpx.Response,
        attempt: int,
    ) -> float:
        """Respect Retry-After when available, with a bounded delay."""

        retry_after = response.headers.get("Retry-After")

        if retry_after:
            try:
                return min(max(float(retry_after), 0), 10)
            except ValueError:
                try:
                    retry_time = parsedate_to_datetime(retry_after)

                    if retry_time.tzinfo is None:
                        retry_time = retry_time.replace(tzinfo=UTC)

                    delay = (
                        retry_time - datetime.now(UTC)
                    ).total_seconds()

                    return min(max(delay, 0), 10)
                except (TypeError, ValueError, OverflowError):
                    pass

        return float(2 ** attempt)

    @staticmethod
    def _extract_pdf_text(content: bytes) -> str:
        """Extract text from a PDF, using PyMuPDF as a fallback."""

        try:
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )
        except Exception:
            text = ""

        if not text.strip():
            try:
                with pymupdf.open(stream=content, filetype="pdf") as document:
                    text = "\n".join(
                        page.get_text("text")
                        for page in document
                    )
            except Exception as exc:
                raise ValueError(
                    f"Unable to extract text from PDF: {exc}"
                ) from exc

        if not text.strip():
            raise ValueError(
                "PDF contains no extractable text. "
                "Scanned PDFs require OCR."
            )

        return (
            "<html><body><pre>"
            f"{html.escape(text)}"
            "</pre></body></html>"
        )

    @classmethod
    def _extract_html(cls, response: httpx.Response) -> str:
        """Validate the response and handle HTML or PDF content."""

        response.raise_for_status()

        content_type = response.headers.get(
            "content-type", ""
        ).lower()

        content = response.content

        is_pdf = (
            "application/pdf" in content_type
            or content.lstrip().startswith(b"%PDF-")
        )

        if is_pdf:
            return cls._extract_pdf_text(content)

        if content_type and not any(
            content_type.startswith(media_type)
            for media_type in (
                "text/html",
                "application/xhtml+xml",
            )
        ):
            raise ValueError(
                f"Expected HTML or PDF content, received: {content_type}"
            )

        return response.text
