
import httpx
import pytest

from reader.web_fetcher import WebFetcher


@pytest.mark.asyncio
async def test_fetch_returns_html():
    async def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><body>Research content</body></html>",
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        fetcher = WebFetcher(client=client)
        result = await fetcher.fetch("https://example.com")

    assert "Research content" in result


@pytest.mark.asyncio
async def test_fetch_extracts_pdf_text():
    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Renewable energy research report")
    pdf_bytes = document.tobytes()
    document.close()

    async def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "application/pdf"},
            content=pdf_bytes,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        fetcher = WebFetcher(client=client)
        result = await fetcher.fetch("https://example.com/report.pdf")

    assert "Renewable energy research report" in result


@pytest.mark.asyncio
async def test_fetch_retries_server_error():
    attempts = 0

    async def handler(request):
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            return httpx.Response(503)

        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html>Recovered successfully</html>",
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        fetcher = WebFetcher(client=client, max_retries=1)
        result = await fetcher.fetch("https://example.com")

    assert attempts == 2
    assert "Recovered successfully" in result


@pytest.mark.asyncio
async def test_fetch_does_not_retry_forbidden_response():
    attempts = 0

    async def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(403)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        fetcher = WebFetcher(client=client, max_retries=2)

        with pytest.raises(httpx.HTTPStatusError):
            await fetcher.fetch("https://example.com")

    assert attempts == 1
