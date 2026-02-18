from typing import Optional, Tuple

import numpy as np

from core.rag.interfaces import OcrEngine


class RapidOcrEngine(OcrEngine):
    def __init__(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError("rapidocr-onnxruntime is required for OCR mode auto/on") from exc

        self._ocr = RapidOCR()

    def extract_text(self, image) -> Tuple[str, Optional[float]]:
        arr = np.array(image)
        result, _ = self._ocr(arr)
        if not result:
            return "", None

        lines = []
        scores = []
        for item in result:
            if len(item) < 3:
                continue
            text = str(item[1]).strip()
            conf = float(item[2]) if item[2] is not None else 0.0
            if text:
                lines.append(text)
                scores.append(conf)

        if not lines:
            return "", None

        avg_conf = sum(scores) / len(scores) if scores else None
        return "\n".join(lines), avg_conf
