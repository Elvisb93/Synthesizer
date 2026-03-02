from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentMode(str, Enum):
    HYBRID = "hybrid"
    STRICT_GROUNDED = "strict_grounded"
    PURE = "pure"


class DocumentSection(BaseModel):
    title: str
    purpose: str
    target_words: int


class DocumentOutline(BaseModel):
    topic: str
    audience: str = "General"
    total_target_words: int = 1200
    sections: List[DocumentSection] = Field(default_factory=list)


class FactRegistry(BaseModel):
    named_entities: List[str] = Field(default_factory=list)
    key_stats: List[str] = Field(default_factory=list)
    decisions_made: List[str] = Field(default_factory=list)
    terminology: Dict[str, str] = Field(default_factory=dict)


class StyleSignals(BaseModel):
    detected_tone: str = "professional"
    avg_sentence_length: str = "medium"
    pov: str = "third person"
    special_rules: List[str] = Field(default_factory=list)


class DocumentPosition(BaseModel):
    section_index: int = 0
    section_title: str = ""
    chunk_index: int = 0
    percent_complete: float = 0.0


class DocumentState(BaseModel):
    position: DocumentPosition = Field(default_factory=DocumentPosition)
    rolling_summary: str = ""
    tail_content: str = ""
    fact_registry: FactRegistry = Field(default_factory=FactRegistry)
    style_signals: StyleSignals = Field(default_factory=StyleSignals)
    consistency_patches: List[str] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    chunk_id: str
    section_index: int
    section_title: str
    chunk_index: int
    text: str
    word_count: int
    citations: List[Dict[str, object]] = Field(default_factory=list)


class ValidationResult(BaseModel):
    is_valid: bool
    reasons: List[str] = Field(default_factory=list)


class DocumentCheckpoint(BaseModel):
    job_id: str
    prompt: str
    mode: DocumentMode = DocumentMode.HYBRID
    outline: DocumentOutline
    state: DocumentState
    chunks: List[DocumentChunk] = Field(default_factory=list)
    completed: bool = False


class DocumentGenerationOptions(BaseModel):
    prompt: str
    target_words: int = 1200
    audience: str = "General"
    tone: str = "professional"
    mode: DocumentMode = DocumentMode.HYBRID
    max_chunk_words: int = 500
    min_chunk_words: int = 220
    max_retries: int = 3
    consistency_check_interval: int = 12
    fast_mode: bool = False
    hard_max_words: int = 0
    resume: bool = True
