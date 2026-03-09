import json
import os
import urllib.request
from typing import Optional

from core.rag.interfaces import DocumentParser
from core.rag.models import ParsedDocument

from .excel_parser import ExcelParser
from .html_parser import HtmlParser
from .image_ocr_parser import ImageOcrParser
from .text_like_parser import TextLikeParser


class RouterParser(DocumentParser):
    def __init__(self, *, pdf_parser: DocumentParser):
        self.pdf_parser = pdf_parser
        self.text_parser = TextLikeParser()
        self.excel_parser = ExcelParser()
        self.html_parser = HtmlParser()
        self._image_parser: Optional[ImageOcrParser] = None

    @staticmethod
    def _is_url(source: str) -> bool:
        s = (source or "").strip().lower()
        return s.startswith("http://") or s.startswith("https://")

    def _parse_url(self, url: str) -> ParsedDocument:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Synthesizer-RAG/1.0"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
            content_type = (resp.headers.get("Content-Type") or "").lower()

        text = raw.decode("utf-8", errors="ignore")
        source_type = "url"
        page_text = text
        title = None

        if "text/html" in content_type or "<html" in text.lower():
            page_text = self.html_parser.extract_text(text)
            title = self.html_parser.extract_title(text)
            source_type = "url_html"
        elif "application/json" in content_type:
            try:
                parsed = json.loads(text)
                page_text = json.dumps(parsed, indent=2, ensure_ascii=True)
                source_type = "url_json"
            except Exception:
                source_type = "url_text"
        else:
            source_type = "url_text"

        return ParsedDocument(
            source=url,
            source_type=source_type,
            pages=[page_text],
            metadata={"source_type": source_type, "content_type": content_type, "title": title},
        )

    def parse(self, source: str) -> ParsedDocument:
        if self._is_url(source):
            return self._parse_url(source)

        if not os.path.exists(source):
            raise FileNotFoundError(f"Source not found: {source}")

        ext = os.path.splitext(source)[1].lower()
        if ext == ".pdf":
            parsed = self.pdf_parser.parse(source)
            if not parsed.source_type or parsed.source_type == "file":
                parsed.source_type = "pdf"
            parsed.metadata.setdefault("source_type", parsed.source_type)
            return parsed
        if ext in {".xlsx", ".xls"}:
            return self.excel_parser.parse(source)
        if ext in {".html", ".htm"}:
            return self.html_parser.parse(source)
        if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            if self._image_parser is None:
                self._image_parser = ImageOcrParser()
            return self._image_parser.parse(source)
        return self.text_parser.parse(source)
