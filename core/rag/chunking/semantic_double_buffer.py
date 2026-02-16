import re
from typing import List

from core.rag.interfaces import Chunker
from core.rag.models import ChunkRecord, ParsedDocument


class SemanticDoubleBufferChunker(Chunker):
    def __init__(self, window_sentences: int = 6, overlap_sentences: int = 2, buffer_sentences: int = 1):
        self.window_sentences = max(1, window_sentences)
        self.overlap_sentences = max(0, overlap_sentences)
        self.buffer_sentences = max(0, buffer_sentences)

    def _sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p and p.strip()]

    def chunk(self, doc: ParsedDocument) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []

        for page_index, page_text in enumerate(doc.pages, start=1):
            sentences = self._sentences(page_text)
            if not sentences:
                continue

            stride = max(1, self.window_sentences - self.overlap_sentences)
            i = 0
            local_idx = 0

            while i < len(sentences):
                core_start = i
                core_end = min(len(sentences), i + self.window_sentences)

                start = max(0, core_start - self.buffer_sentences)
                end = min(len(sentences), core_end + self.buffer_sentences)
                text = " ".join(sentences[start:end]).strip()

                if text:
                    chunk_id = f"{doc.source}::p{page_index}::c{local_idx}"
                    chunks.append(
                        ChunkRecord(
                            chunk_id=chunk_id,
                            text=text,
                            metadata={
                                "source": doc.source,
                                "page": page_index,
                                "core_start": core_start,
                                "core_end": core_end,
                            },
                        )
                    )

                if core_end >= len(sentences):
                    break

                i += stride
                local_idx += 1

        return chunks
