import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.rag.interfaces import DocumentParser, OcrEngine
from core.rag.models import ParsedDocument


Rect = Tuple[float, float, float, float]


class HybridPdfParser(DocumentParser):
    def __init__(
        self,
        *,
        ocr_mode: str = "off",
        ocr_dpi: int = 150,
        ocr_max_pages: int = 20,
        ocr_max_regions_per_page: int = 8,
        ocr_region_padding_px: int = 18,
        ocr_gap_multiplier: float = 2.5,
        ocr_min_extracted_chars: int = 60,
        ocr_timeout_ms_per_page: int = 4000,
        ocr_engine: Optional[OcrEngine] = None,
    ):
        self.ocr_mode = (ocr_mode or "off").lower()
        self.ocr_dpi = max(72, int(ocr_dpi))
        self.ocr_max_pages = max(0, int(ocr_max_pages))
        self.ocr_max_regions_per_page = max(0, int(ocr_max_regions_per_page))
        self.ocr_region_padding_px = max(0, int(ocr_region_padding_px))
        self.ocr_gap_multiplier = max(1.0, float(ocr_gap_multiplier))
        self.ocr_min_extracted_chars = max(0, int(ocr_min_extracted_chars))
        self.ocr_timeout_ms_per_page = max(100, int(ocr_timeout_ms_per_page))
        self._ocr_engine = ocr_engine

    def _ensure_ocr_engine(self) -> Optional[OcrEngine]:
        if self._ocr_engine is not None:
            return self._ocr_engine
        try:
            from core.rag.ocr import RapidOcrEngine

            self._ocr_engine = RapidOcrEngine()
            return self._ocr_engine
        except Exception:
            return None

    def _extract_line_blocks(self, text_page) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        rect_count = int(text_page.count_rects())
        for idx in range(rect_count):
            left, bottom, right, top = text_page.get_rect(idx)
            text = (text_page.get_text_bounded(left=left, bottom=bottom, right=right, top=top) or "").strip()
            if not text:
                continue
            blocks.append(
                {
                    "text": text,
                    "rect": (float(left), float(bottom), float(right), float(top)),
                }
            )

        blocks.sort(key=lambda b: (-b["rect"][3], b["rect"][0]))
        return blocks

    def _line_text(self, blocks: Sequence[Dict[str, Any]]) -> str:
        return "\n".join([b["text"] for b in blocks if b.get("text")]).strip()

    def _detect_gap_regions(self, blocks: Sequence[Dict[str, Any]]) -> List[Rect]:
        if len(blocks) < 3:
            return []

        gaps: List[float] = []
        for i in range(len(blocks) - 1):
            cur = blocks[i]["rect"]
            nxt = blocks[i + 1]["rect"]
            gap = max(0.0, cur[1] - nxt[3])
            gaps.append(gap)

        if not gaps:
            return []
        median_gap = max(1e-6, statistics.median(gaps))

        regions: List[Rect] = []
        for i, gap in enumerate(gaps):
            if gap <= median_gap * self.ocr_gap_multiplier:
                continue

            start_idx = max(0, i - 1)
            end_idx = min(len(blocks) - 1, i + 2)
            subset = blocks[start_idx : end_idx + 1]
            left = min(b["rect"][0] for b in subset)
            bottom = min(b["rect"][1] for b in subset)
            right = max(b["rect"][2] for b in subset)
            top = max(b["rect"][3] for b in subset)
            regions.append((left, bottom, right, top))

        return regions

    @staticmethod
    def _pdf_to_img_rect(rect: Rect, img_w: int, img_h: int, scale: float, pad: int) -> Tuple[int, int, int, int]:
        left, bottom, right, top = rect
        x1 = int(max(0, left * scale - pad))
        x2 = int(min(img_w, right * scale + pad))
        y1 = int(max(0, img_h - top * scale - pad))
        y2 = int(min(img_h, img_h - bottom * scale + pad))
        if x2 <= x1:
            x2 = min(img_w, x1 + 2)
        if y2 <= y1:
            y2 = min(img_h, y1 + 2)
        return x1, y1, x2, y2

    def _ocr_full_page(self, page, engine: OcrEngine) -> Tuple[str, Optional[float]]:
        scale = self.ocr_dpi / 72.0
        pil = page.render(scale=scale).to_pil()
        return engine.extract_text(pil)

    def _ocr_regions(self, page, regions: Sequence[Rect], engine: OcrEngine, deadline: float) -> Tuple[List[str], List[float], int]:
        if not regions:
            return [], [], 0

        scale = self.ocr_dpi / 72.0
        pil = page.render(scale=scale).to_pil()
        img_w, img_h = pil.size

        texts: List[str] = []
        confs: List[float] = []
        used = 0
        for rect in regions[: self.ocr_max_regions_per_page]:
            if time.time() > deadline:
                break
            x1, y1, x2, y2 = self._pdf_to_img_rect(rect, img_w, img_h, scale, self.ocr_region_padding_px)
            crop = pil.crop((x1, y1, x2, y2))
            text, conf = engine.extract_text(crop)
            if text.strip():
                texts.append(text.strip())
                if conf is not None:
                    confs.append(conf)
            used += 1

        return texts, confs, used

    def parse(self, path: str) -> ParsedDocument:
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise RuntimeError("pypdfium2 is required for PDF ingestion") from exc

        pdf = pdfium.PdfDocument(path)
        pages: List[str] = []
        page_metadata: List[Dict[str, Any]] = []
        ocr_pages_used = 0

        try:
            for idx in range(len(pdf)):
                page = pdf[idx]
                text_page = page.get_textpage()

                blocks = self._extract_line_blocks(text_page)
                native_text = self._line_text(blocks)
                final_text = native_text

                mode = self.ocr_mode
                should_consider_ocr = mode in ("auto", "on") and (idx < self.ocr_max_pages)
                ocr_used = False
                ocr_scope = "none"
                ocr_regions_count = 0
                ocr_conf: Optional[float] = None
                ocr_error: Optional[str] = None

                if should_consider_ocr:
                    needs_full = mode == "on" or len(native_text.strip()) < self.ocr_min_extracted_chars
                    regions: List[Rect] = []
                    if mode == "auto" and not needs_full:
                        regions = self._detect_gap_regions(blocks)

                    if needs_full or regions:
                        engine = self._ensure_ocr_engine()
                        if engine is None:
                            if needs_full:
                                ocr_error = "OCR engine unavailable"
                        else:
                            deadline = time.time() + (self.ocr_timeout_ms_per_page / 1000.0)
                            try:
                                if needs_full:
                                    ocr_text, conf = self._ocr_full_page(page, engine)
                                    if ocr_text.strip():
                                        final_text = ocr_text.strip() if mode == "on" else (native_text + "\n" + ocr_text).strip()
                                        ocr_used = True
                                        ocr_scope = "full_page"
                                        ocr_conf = conf
                                        ocr_pages_used += 1
                                elif mode == "auto":
                                    region_texts, confs, used = self._ocr_regions(page, regions, engine, deadline)
                                    ocr_regions_count = used
                                    if region_texts:
                                        final_text = (native_text + "\n" + "\n".join(region_texts)).strip()
                                        ocr_used = True
                                        ocr_scope = "region"
                                        ocr_conf = (sum(confs) / len(confs)) if confs else None
                                        ocr_pages_used += 1
                            except Exception as exc:
                                ocr_error = str(exc)

                pages.append(final_text.strip())
                page_metadata.append(
                    {
                        "page": idx + 1,
                        "ocr_used": ocr_used,
                        "ocr_scope": ocr_scope,
                        "ocr_regions_count": ocr_regions_count,
                        "ocr_confidence_avg": ocr_conf,
                        "ocr_error": ocr_error,
                    }
                )

                text_page.close()
                page.close()
        finally:
            pdf.close()

        return ParsedDocument(source=path, pages=pages, page_metadata=page_metadata)
