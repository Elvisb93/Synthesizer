from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ParsedDocument(BaseModel):
    source: str
    pages: List[str]
    page_metadata: List[Dict[str, Any]] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestReport(BaseModel):
    files_processed: int = 0
    files_skipped: int = 0
    chunks_created: int = 0
    vectors_upserted: int = 0
    ocr_pages_total: int = 0
    ocr_pages_full: int = 0
    ocr_regions_total: int = 0
    ocr_failures: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = Field(default_factory=list)


class SearchOptions(BaseModel):
    top_k: int = 5
    min_score: float = 0.25
    max_context_chars: int = 3000
    source_filter: Optional[str] = None
