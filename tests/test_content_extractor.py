
import pytest

from reader.content_extractor import ContentExtractor


def test_extract_article_content():
    html = """
    <html>
        <body>
            <nav>Navigation menu</nav>
            <article>
                <h1>Renewable Energy in India</h1>
                <p>Solar energy capacity continues to increase.</p>
            </article>
            <footer>Footer content</footer>
        </body>
    </html>
    """

    result = ContentExtractor().extract(html)

    assert "Renewable Energy in India" in result
    assert "Solar energy capacity" in result
    assert "Navigation menu" not in result
    assert "Footer content" not in result


def test_extract_falls_back_to_beautifulsoup(monkeypatch):
    import reader.content_extractor as extractor_module

    monkeypatch.setattr(
        extractor_module.trafilatura,
        "extract",
        lambda *args, **kwargs: None,
    )

    html = """
    <html>
        <body>
            <main>
                <p>Government renewable energy initiatives.</p>
            </main>
        </body>
    </html>
    """

    result = ContentExtractor().extract(html)

    assert "Government renewable energy initiatives" in result


def test_extract_respects_character_limit():
    html = """
    <html><body>
        <article>
            <p>Renewable energy research information.</p>
        </article>
    </body></html>
    """

    result = ContentExtractor(max_characters=15).extract(html)

    assert len(result) <= 15


def test_extract_rejects_empty_html():
    with pytest.raises(ValueError, match="empty HTML"):
        ContentExtractor().extract("   ")


def test_extract_rejects_html_without_readable_content():
    html = "<html><body><script>console.log('test')</script></body></html>"

    with pytest.raises(ValueError, match="No readable text"):
        ContentExtractor().extract(html)
