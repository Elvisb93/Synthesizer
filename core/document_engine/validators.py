import re
from typing import Iterable, Set

from .models import ValidationResult


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def _ngrams(text: str, n: int = 5) -> Set[str]:
    words = [w for w in re.findall(r"\w+", (text or "").lower()) if w]
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def validate_chunk(
    text: str,
    *,
    target_words: int,
    tail_content: str,
    next_section_title: str = "",
    min_ratio: float = 0.8,
    max_ratio: float | None = None,
    repetition_jaccard_threshold: float = 0.30,
) -> ValidationResult:
    reasons = []
    text = (text or "").strip()
    if not text:
        return ValidationResult(is_valid=False, reasons=["Empty output."])

    wc = _word_count(text)
    lower = max(40, int(target_words * min_ratio))
    if wc < lower:
        reasons.append(f"Chunk too short ({wc} words, expected >= {lower}).")
    if max_ratio is not None:
        upper = max(lower + 20, int(target_words * max_ratio))
        if wc > upper:
            reasons.append(f"Chunk too long ({wc} words, expected <= {upper}).")

    if text[-1] not in ".!?\"'":
        reasons.append("Chunk appears truncated (does not end on natural boundary).")

    if tail_content.strip():
        overlap = _jaccard(_ngrams(text, 5), _ngrams(tail_content, 5))
        if overlap > repetition_jaccard_threshold:
            reasons.append(f"Chunk repeats prior content (5-gram Jaccard={overlap:.2f}).")

    next_title = (next_section_title or "").strip()
    if next_title:
        escaped = re.escape(next_title)
        heading_hit = re.search(rf"(?im)^\s*{escaped}\s*$", text)
        mention_hit = re.search(rf"(?i)\b{escaped}\b", text)
        if heading_hit:
            reasons.append(f"Chunk appears to start next section: '{next_title}'.")
        elif mention_hit:
            idx = mention_hit.start()
            if idx >= int(len(text) * 0.6):
                reasons.append(f"Chunk likely overflowed into next section topic: '{next_title}'.")

    return ValidationResult(is_valid=not reasons, reasons=reasons)
