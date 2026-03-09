import html
import os
import re
from typing import Optional

from core.rag.interfaces import DocumentParser
from core.rag.models import ParsedDocument


class HtmlParser(DocumentParser):
    _SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style).*?>.*?</\\1>")
    _TAG_RE = re.compile(r"(?s)<[^>]+>")
    _WS_RE = re.compile(r"\\s+")
    _TITLE_RE = re.compile(r"(?is)<title>(.*?)</title>")

    @classmethod
    def extract_text(cls, html_text: str) -> str:
        cleaned = cls._SCRIPT_STYLE_RE.sub(" ", html_text or "")
        cleaned = cls._TAG_RE.sub(" ", cleaned)
        cleaned = html.unescape(cleaned)
        cleaned = cls._WS_RE.sub(" ", cleaned).strip()
        return cleaned

    @classmethod
    def extract_title(cls, html_text: str) -> Optional[str]:
        m = cls._TITLE_RE.search(html_text or "")
        if not m:
            return None
        title = cls._WS_RE.sub(" ", html.unescape(m.group(1) or "")).strip()
        return title or None

    def parse(self, path: str) -> ParsedDocument:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html_text = f.read()

        text = self.extract_text(html_text)
        title = self.extract_title(html_text)
        return ParsedDocument(
            source=path,
            source_type="html",
            pages=[text],
            metadata={
                "source_type": "html",
                "title": title,
                "extension": os.path.splitext(path)[1].lower(),
            },
        )
