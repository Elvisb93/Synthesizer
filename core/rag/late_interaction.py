import re
from typing import List, Sequence


class LateInteractionScorer:
    """
    ColBERT-style late interaction approximation.

    This is a lightweight local scorer that models per-token matching behavior.
    It is not a full ColBERT implementation, but it provides a compatible
    scoring hook and improved token-level relevance over plain cosine/BM25 blend.
    """

    TOKEN_RE = re.compile(r"[a-z0-9]+")

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
        ga = cls._char_ngrams(a)
        gb = cls._char_ngrams(b)
        inter = len(ga.intersection(gb))
        union = max(1, len(ga.union(gb)))
        return inter / union

    @classmethod
    def score(cls, query: str, text: str) -> float:
        q = cls._tokens(query)
        d = cls._tokens(text)
        if not q or not d:
            return 0.0

        # MaxSim-like: each query token finds best match in doc tokens.
        sims: List[float] = []
        for qt in q:
            best = 0.0
            for dt in d:
                sim = cls._token_similarity(qt, dt)
                if sim > best:
                    best = sim
                if best >= 1.0:
                    break
            sims.append(best)

        # Phrase bonus if query bigrams appear in order.
        phrase_bonus = 0.0
        if len(q) >= 2:
            q_bigrams = {" ".join(q[i : i + 2]) for i in range(len(q) - 1)}
            text_l = " ".join(d)
            hits = sum(1 for bg in q_bigrams if bg in text_l)
            phrase_bonus = min(0.2, 0.05 * hits)

        return min(1.0, (sum(sims) / len(sims)) + phrase_bonus)

    @classmethod
    def batch_score(cls, query: str, texts: Sequence[str]) -> List[float]:
        return [cls.score(query, t) for t in texts]
