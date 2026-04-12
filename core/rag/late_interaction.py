import re
from collections import Counter
from typing import List, Sequence


class LateInteractionScorer:
    """
    ColBERT-style late interaction approximation.

    This is a lightweight local scorer that models per-token matching behavior.
    It is not a full ColBERT implementation, but it provides a compatible
    scoring hook and improved token-level relevance over plain cosine/BM25 blend.
    """

    TOKEN_RE = re.compile(r"[a-z0-9]+")
    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }

    @classmethod
    def _tokens(cls, text: str) -> List[str]:
        return cls.TOKEN_RE.findall((text or "").lower())

    @staticmethod
    def _char_ngrams(token: str, n: int = 3) -> set[str]:
        if len(token) < n:
            return {token}
        return {token[i : i + n] for i in range(len(token) - n + 1)}

    @classmethod
    def _token_similarity(cls, a: str, b: str) -> float:
        if a == b:
            return 1.0
        if a.startswith(b) or b.startswith(a):
            shorter = min(len(a), len(b))
            longer = max(len(a), len(b))
            if longer > 0:
                return max(0.0, min(0.96, 0.84 + 0.12 * (shorter / longer)))
        ga = cls._char_ngrams(a)
        gb = cls._char_ngrams(b)
        inter = len(ga.intersection(gb))
        union = max(1, len(ga.union(gb)))
        return inter / union

    @classmethod
    def _query_token_weight(cls, token: str) -> float:
        base = 0.35 if token in cls.STOPWORDS else 1.0
        length_bonus = min(0.45, max(0, len(token) - 5) * 0.05)
        return base + length_bonus

    @classmethod
    def _coverage_score(cls, sims: List[float], weights: List[float]) -> float:
        matched = 0.0
        total = max(1e-9, sum(weights))
        for sim, weight in zip(sims, weights):
            if sim >= 0.72:
                matched += weight
            elif sim >= 0.5:
                matched += weight * 0.5
        return min(1.0, matched / total)

    @staticmethod
    def _proximity_signal(best_positions: List[int | None]) -> float:
        positions = [p for p in best_positions if p is not None]
        if len(positions) < 2:
            return 0.0
        span = max(positions) - min(positions) + 1
        return min(1.0, len(positions) / max(1, span))

    @staticmethod
    def _order_signal(query_tokens: List[str], doc_tokens: List[str]) -> float:
        if len(query_tokens) < 2 or len(doc_tokens) < 2:
            return 0.0

        bigram_hits = 0
        trigram_hits = 0
        doc_joined = " ".join(doc_tokens)
        for i in range(len(query_tokens) - 1):
            if " ".join(query_tokens[i : i + 2]) in doc_joined:
                bigram_hits += 1
        for i in range(len(query_tokens) - 2):
            if " ".join(query_tokens[i : i + 3]) in doc_joined:
                trigram_hits += 1

        max_bigrams = max(1, len(query_tokens) - 1)
        max_trigrams = max(1, len(query_tokens) - 2)
        bigram_ratio = bigram_hits / max_bigrams
        trigram_ratio = trigram_hits / max_trigrams if len(query_tokens) >= 3 else 0.0
        return min(1.0, 0.55 * bigram_ratio + 0.45 * trigram_ratio)

    @classmethod
    def score(cls, query: str, text: str) -> float:
        q = cls._tokens(query)
        d = cls._tokens(text)
        if not q or not d:
            return 0.0

        doc_freq = Counter(d)
        sims: List[float] = []
        weights: List[float] = []
        best_positions: List[int | None] = []

        for qt in q:
            best = 0.0
            best_pos = None
            for idx, dt in enumerate(d):
                sim = cls._token_similarity(qt, dt)
                if sim > best:
                    best = sim
                    best_pos = idx
                if best >= 1.0:
                    break
            sims.append(best)
            best_positions.append(best_pos if best >= 0.5 else None)

            weight = cls._query_token_weight(qt)
            freq_penalty = 1.0 / (1.0 + 0.12 * max(0, doc_freq.get(qt, 0) - 1))
            weights.append(weight * freq_penalty)

        weighted_total = max(1e-9, sum(weights))
        weighted_maxsim = sum(sim * weight for sim, weight in zip(sims, weights)) / weighted_total
        coverage = cls._coverage_score(sims, weights)
        proximity_signal = cls._proximity_signal(best_positions)
        order_signal = cls._order_signal(q, d)

        final = (
            0.58 * weighted_maxsim
            + 0.22 * coverage
            + 0.10 * proximity_signal
            + 0.10 * order_signal
        )
        return min(1.0, max(0.0, final))

    @classmethod
    def batch_score(cls, query: str, texts: Sequence[str]) -> List[float]:
        return [cls.score(query, t) for t in texts]
