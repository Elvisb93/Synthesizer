from typing import List

from core.rag.interfaces import DocumentParser
from core.rag.models import ParsedDocument


class PdfiumParser(DocumentParser):
    def parse(self, path: str) -> ParsedDocument:
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise RuntimeError("pypdfium2 is required for PDF ingestion") from exc

        pdf = pdfium.PdfDocument(path)
        pages: List[str] = []

        try:
            for idx in range(len(pdf)):
                page = pdf[idx]
                text_page = page.get_textpage()
                text = text_page.get_text_range() or ""
                text_page.close()
                page.close()
                pages.append(text.strip())
        finally:
            pdf.close()

        return ParsedDocument(source=path, pages=pages)
