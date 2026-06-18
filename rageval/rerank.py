"""Re-ranking — a CROSS-ENCODER reorders the fused candidate chunks.

WHY a second model after retrieval? Dense retrieval and BM25 are BI-ENCODERS: they embed
the query and each document SEPARATELY, then compare vectors. That's fast (you can
pre-embed millions of docs) but lossy — the query never "sees" the document during
scoring. A CROSS-ENCODER instead feeds (query, document) TOGETHER through a transformer
and outputs one relevance score. It's far more accurate but far too slow to run over a
whole corpus.

The standard pattern — and what this does — is RETRIEVE-THEN-RERANK: use the cheap
bi-encoders (+ BM25) to fetch a small candidate set (say 20), then spend the expensive
cross-encoder only on those 20 to pick the true top-k. Best of both: corpus-scale recall,
pair-scale precision.

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` — a small, fast cross-encoder trained on
the MS MARCO passage-ranking task. Loaded lazily and cached on the instance.
"""

from __future__ import annotations

from .config import SETTINGS, Settings


class CrossEncoderReranker:
    """Wraps a sentence-transformers CrossEncoder. Scores (query, text) pairs."""

    def __init__(self, settings: Settings = SETTINGS):
        from sentence_transformers import CrossEncoder  # lazy: heavy torch import

        self.model = CrossEncoder(settings.rerank_model)

    def rerank(self, query: str, candidates: list, *, top_k: int) -> list:
        """Reorder `candidates` by cross-encoder relevance to `query`, return the top_k.

        `candidates` is a list of objects with a `.text` attribute (our fused hits). We
        return the SAME objects, reordered and truncated, with a `.rerank_score` set."""
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)
        scored = list(zip(candidates, (float(s) for s in scores)))
        scored.sort(key=lambda cs: cs[1], reverse=True)
        out = []
        for c, s in scored[:top_k]:
            # attach the rerank score for transparency/debugging (dataclasses are frozen,
            # so we carry it on a parallel attribute via object.__setattr__-safe copy).
            try:
                object.__setattr__(c, "rerank_score", s)
            except Exception:  # noqa: BLE001
                pass
            out.append(c)
        return out
