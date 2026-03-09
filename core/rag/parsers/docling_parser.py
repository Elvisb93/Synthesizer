import os
from typing import Optional

from core.rag.interfaces import DocumentParser
from core.rag.models import ParsedDocument


class DoclingParser(DocumentParser):
    """
    Optional parser backed by Docling.

    This parser is intentionally defensive: if Docling APIs change across
    versions, it attempts a few common export surfaces and otherwise falls back
    to string conversion.
    """

    def __init__(self):
        try:
            from docling.document_converter import DocumentConverter
        except Exception as exc:
            raise RuntimeError(
                "Docling is not installed. Install optional dependencies for parser_mode='docling'."
            ) from exc
        self._converter = DocumentConverter()

    @staticmethod
    def _extract_text(result) -> str:
        # Common Docling shape: result.document.export_to_markdown()
        doc = getattr(result, "document", None)
        if doc is not None:
            if hasattr(doc, "export_to_markdown"):
                try:
                    return str(doc.export_to_markdown() or "")
                except Exception:
                    pass
            if hasattr(doc, "export_to_text"):
                try:
                    return str(doc.export_to_text() or "")
                except Exception:
                    pass

        # Alternate result surfaces.
        if hasattr(result, "export_to_markdown"):
            try:
                return str(result.export_to_markdown() or "")
            except Exception:
                pass
        if hasattr(result, "text"):
            return str(getattr(result, "text") or "")
        return str(result or "")

    def parse(self, path: str) -> ParsedDocument:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Source not found: {path}")

        result = self._converter.convert(path)
        text = self._extract_text(result).strip()
        return ParsedDocument(
            source=path,
            source_type="docling",
            pages=[text],
            metadata={"source_type": "docling", "parser_engine": "docling"},
        )
