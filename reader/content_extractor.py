
import re

from bs4 import BeautifulSoup


class ContentExtractor:
    """Extract readable text from webpage HTML."""

    REMOVED_TAGS = [
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "noscript",
        "svg",
        "form",
    ]

    def __init__(self, max_characters: int = 100_000) -> None:
        if max_characters < 1:
            raise ValueError("max_characters must be at least 1.")

        self.max_characters = max_characters

    def extract(self, html: str) -> str:
        """Convert HTML into clean, readable text."""

        if not html.strip():
            raise ValueError("Cannot extract content from empty HTML.")

        soup = BeautifulSoup(html, "html.parser")

        for tag_name in self.REMOVED_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        content = (
            soup.find("article")
            or soup.find("main")
            or soup.body
            or soup
        )

        text = content.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            raise ValueError("No readable text found in webpage HTML.")

        return text[: self.max_characters]
