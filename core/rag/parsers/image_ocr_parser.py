import os

from core.rag.interfaces import DocumentParser
from core.rag.ocr.rapidocr_engine import RapidOcrEngine
from core.rag.models import ParsedDocument


class ImageOcrParser(DocumentParser):
    def __init__(self):
        self._engine = RapidOcrEngine()

    def parse(self, path: str) -> ParsedDocument:
        try:
            from PIL import Image
        except Exception as exc:
            raise RuntimeError("Pillow is required for image ingestion (pip install Pillow)") from exc

        image = Image.open(path)
        text, conf = self._engine.extract_text(image)
        text = (text or "").strip()
        return ParsedDocument(
            source=path,
            source_type="image",
            pages=[text],
            metadata={
                "source_type": "image",
                "extension": os.path.splitext(path)[1].lower(),
                "ocr_confidence_avg": conf,
            },
            page_metadata=[
                {
                    "page": 1,
                    "ocr_used": True,
                    "ocr_scope": "full_page",
                    "ocr_confidence_avg": conf,
                }
            ],
        )
