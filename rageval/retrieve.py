"""Retrieve: HYBRID (dense + BM25) → RRF FUSION → cross-encoder RE-RANK → top-k.

This is the heart of an enterprise retriever, and it's where most RAG quality actually
comes from. Three ideas, layered:

1. HYBRID retrieval — run TWO retrievers and combine them:
   * DENSE (vectors / Qdrant): matches MEANING. "fruit-like theme" finds "citrus" and
     "lemon" even with no shared words. Weakness: exact terms, rare names, codes.
   * SPARSE (BM25): classic keyword scoring (term frequency × inverse document
     frequency). Matches EXACT words — brand names, IDs, jargon — that embeddings blur.
   Each covers the other's blind spot, so the union beats either alone.

2. RRF (Reciprocal Rank Fusion) — how to combine two ranked lists that have
   INCOMPARABLE scores (a cosine similarity and a BM25 score are not on the same scale).
   RRF ignores the raw scores and uses only RANK POSITION: each item gets
   1 / (k + rank) summed across the lists it appears in (k≈60 damps the top-rank
   dominance). Items ranked high by BOTH retrievers rise to the top. Simple, robust,
   parameter-light — the standard hybrid-fusion method.

3. RE-RANK — hand the fused top candidates to a cross-encoder (rerank.py) for a final,
   precise ordering. See that module for why.

Retrieval quality is the CEILING on answer quality: the generator can only answer from
what you retrieve. That's why this file is the most important one for metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import SAMPLE_COLLECTION_SUFFIX, SETTINGS, Settings
from .embeddings import get_embedder


@dataclass
class Retrieved:
    """One retrieved chunk plus enough metadata to cite it, aggregate on it, and debug."""
    text: str
    project_id: str
    source_set: str
    source: str
    doc_type: str
    chunk_index: int
    score: float                       # fused (RRF) score
    dense_rank: int | None = None      # 1-based rank in the dense list (None if absent)
    sparse_rank: int | None = None     # 1-based rank in the BM25 list (None if absent)
    rerank_score: float = field(default=0.0)


class CollectionMissingError(RuntimeError):
    """The target Qdrant collection doesn't exist yet — raised BEFORE any read so the caller
    gets an actionable message instead of a cryptic deep-stack Qdrant 404.

    A `RuntimeError` subclass so existing `except RuntimeError` paths still catch it, but a named
    type so callers (the API lifespan) can recognise this specific, recoverable startup condition."""


def _preflight_collection_exists(client, settings: Settings) -> None:
    """Guard the FIRST read: confirm the resolved collection exists before `scroll_all` tries it.

    The collection name is CORPUS-SCOPED (config.py): the default corpus is the synthetic sample,
    so it resolves to `..._sample`. If you ingested a REAL corpus (RAGENGINE_CORPUS_ROOT set) but
    then start the API / retriever WITHOUT that env, the name falls back to `..._sample`, which
    doesn't exist — and a raw `scroll()` would die with a cryptic 404 deep inside `scroll_all`.
    We turn that into a clear, ACTIONABLE error naming the collection and the two ways to fix it."""
    name = settings.collection_name
    if client.collection_exists(name):
        return
    raise CollectionMissingError(
        f"Collection '{name}' does not exist. The engine defaults to the SAMPLE corpus "
        f"(suffix '_{SAMPLE_COLLECTION_SUFFIX}'). Ingest it first (python -m rageval.ingest), "
        f"or to serve your real corpus set RAGENGINE_CORPUS_ROOT=<path> (or "
        f"RAGEVAL_COLLECTION=<name>) to match the collection you ingested."
    )


def _tokenize(text: str) -> list[str]:
    """Cheap, dependency-free tokenizer for BM25: lowercase alphanumeric words."""
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def apply_rerank_floor(hits: list[Retrieved], min_score: float | None) -> list[Retrieved]:
    """Drop any hit whose `rerank_score` is below an ABSOLUTE floor, preserving order.

    The "top-k WITH a relevance threshold" pattern: after reranking has produced the final
    ordering, optionally remove weakly-relevant filler so the generator isn't handed junk
    when few things are truly relevant. This REDUCES the count; it never reorders.

    WHY a fixed-k + absolute floor, and NOT a top-p / nucleus cutoff:
      A top-p ("keep the smallest prefix whose CUMULATIVE probability mass ≥ p") only makes
      sense over a CALIBRATED probability distribution that sums to 1 — e.g. a softmax over a
      vocabulary in token sampling. Cross-encoder / cosine relevance scores are NOT that: they
      don't sum to 1, aren't probabilities, and aren't even comparable ACROSS models (a "good"
      ms-marco-MiniLM score is a different number than a "good" bge-reranker score). So
      cumulative-mass thresholding is the wrong tool. The right tool is a fixed top-k (a hard,
      predictable context budget) plus an OPTIONAL absolute-score floor whose value is
      EMPIRICALLY tuned per reranker — which is exactly why the floor ships DISABLED by default
      (a hard-coded universal threshold would be meaningless across models).

    `min_score=None` → no-op (returns the list unchanged), so the default pipeline is
    byte-for-byte the legacy fixed top-k. An EMPTY result is allowed and intentional: if every
    candidate is below the floor, returning nothing lets the generator say "no relevant context"
    rather than answer from filler — so there is deliberately NO min-1 fallback (that would
    defeat the floor's purpose).
    """
    if min_score is None:
        return hits
    return [h for h in hits if h.rerank_score >= min_score]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], *, k: int = 60) -> dict[str, float]:
    """Fuse several ranked id-lists into one {id: score} map via RRF.

    Pure function over rank positions → deterministic and unit-testable (no models). Each
    list contributes 1/(k + rank) for each id, rank being 1-based. Returns the summed
    scores; the caller sorts descending.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class Retriever:
    """Holds the embedder, the Qdrant client, an in-memory BM25 index, and (optionally)
    the cross-encoder. Built once per process so model loads + the BM25 build are paid
    once, then many queries run cheaply.

    The BM25 index is built by reading every chunk's text back out of Qdrant at startup.
    For a demo/medium corpus that's fine and keeps sparse + dense perfectly in sync; a
    huge corpus would use Qdrant's native sparse vectors instead (noted in the README).
    """

    def __init__(self, settings: Settings = SETTINGS, *, dense_only: bool = False):
        """`dense_only=True` isolates the EMBEDDING MODEL's raw contribution: it turns OFF
        BM25 and the cross-encoder rerank, so retrieval is pure dense ANN. This is one axis
        of the A/B — "dense-only vs full hybrid+rerank" — and it's the cleaner signal when
        comparing two embedding models, because hybrid/rerank can mask a weaker embedder."""
        from rank_bm25 import BM25Okapi

        from .index import get_client, scroll_all

        self.settings = settings
        self.dense_only = dense_only
        self.embedder = get_embedder(settings)
        self.client = get_client(settings)

        # PREFLIGHT: confirm the target collection exists before the first `scroll_all` read, so a
        # corpus/collection mismatch yields a clear, actionable error rather than a deep-stack 404.
        _preflight_collection_exists(self.client, settings)

        # Pull all chunks once to build the BM25 corpus AND a payload lookup by chunk_id.
        # In dense-only mode we still need the payload lookup, but skip the BM25 build.
        self._by_id: dict[str, dict] = {}
        corpus_tokens: list[list[str]] = []
        self._bm25_ids: list[str] = []
        for pl in scroll_all(self.client, settings=settings):
            cid = str(pl.get("chunk_id", ""))
            if not cid:
                continue
            self._by_id[cid] = pl
            self._bm25_ids.append(cid)
            corpus_tokens.append(_tokenize(str(pl.get("text", ""))))
        self._bm25 = None if dense_only else (BM25Okapi(corpus_tokens) if corpus_tokens else None)

        self._reranker = None
        if settings.use_rerank and not dense_only:
            from .rerank import CrossEncoderReranker

            self._reranker = CrossEncoderReranker(settings)

    # ---- the two retrievers -----------------------------------------------

    def _dense(self, question: str, *, limit: int, source_set: str | None) -> list[str]:
        """Dense ANN search → ranked list of chunk_ids."""
        from .index import dense_search

        qvec = self.embedder.embed([question])[0]
        hits = dense_search(self.client, qvec, limit=limit,
                            ef_search=self.settings.hnsw_ef_search, source_set=source_set,
                            settings=self.settings)
        return [h.chunk_id for h in hits]

    def _sparse(self, question: str, *, limit: int, source_set: str | None) -> list[str]:
        """BM25 keyword search → ranked list of chunk_ids."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(question))
        ranked = sorted(zip(self._bm25_ids, scores), key=lambda x: x[1], reverse=True)
        out: list[str] = []
        for cid, sc in ranked:
            if sc <= 0:
                continue
            if source_set is not None and self._by_id[cid].get("source_set") != source_set:
                continue
            out.append(cid)
            if len(out) >= limit:
                break
        return out

    # ---- the public retrieve ----------------------------------------------

    def retrieve(self, question: str, *, top_k: int | None = None,
                 source_set: str | None = None) -> list[Retrieved]:
        """Hybrid retrieve → RRF fuse → optional cross-encoder rerank → top_k.

        `source_set` (optional) restricts to one source-set — the metadata-filter path
        used by set-intersection questions.
        """
        k = top_k if top_k is not None else self.settings.top_k
        cand = self.settings.candidate_k

        dense_ids = self._dense(question, limit=cand, source_set=source_set)
        sparse_ids = self._sparse(question, limit=cand, source_set=source_set)

        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=self.settings.rrf_k)
        dense_pos = {cid: i + 1 for i, cid in enumerate(dense_ids)}
        sparse_pos = {cid: i + 1 for i, cid in enumerate(sparse_ids)}

        # Build Retrieved objects for the fused candidates, sorted by fused score.
        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        candidates: list[Retrieved] = []
        for cid, score in ordered[: max(cand, k)]:
            pl = self._by_id.get(cid)
            if not pl:
                continue
            candidates.append(
                Retrieved(
                    text=str(pl.get("text", "")),
                    project_id=str(pl.get("project_id", "")),
                    source_set=str(pl.get("source_set", "")),
                    source=str(pl.get("source", "")),
                    doc_type=str(pl.get("doc_type", "")),
                    chunk_index=int(pl.get("chunk_index", -1)),
                    score=float(score),
                    dense_rank=dense_pos.get(cid),
                    sparse_rank=sparse_pos.get(cid),
                )
            )

        # RE-RANK the fused candidates for final precision, else just truncate.
        if self._reranker is not None and candidates:
            hits = self._reranker.rerank(question, candidates, top_k=k)
        else:
            hits = candidates[:k]

        # (NEW) Optional absolute rerank-score FLOOR — the LAST step, applied AFTER the top-k
        # ordering is fixed. Drops weakly-relevant filler below the threshold without reordering;
        # disabled by default (None) so the legacy behaviour is unchanged. May return [] on
        # purpose — see apply_rerank_floor. The floor only has calibrated scores to act on when
        # the reranker ran; under --dense-only / use_rerank=false rerank_score is 0.0 for every
        # hit, so a positive floor would empty the list — hence we only apply it post-rerank.
        if self._reranker is not None:
            hits = apply_rerank_floor(hits, self.settings.min_rerank_score)
        return hits


def format_context(chunks: list[Retrieved]) -> str:
    """Render retrieved chunks into a single numbered context block for the prompt.

    Each chunk is labelled [n] with its project + source so the generator can cite by
    number and a reader can trace a claim back to a document — attributable RAG."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (project: {c.source_set}/{c.project_id}, source: {c.source})\n{c.text}")
    return "\n\n".join(blocks)


def main() -> None:
    """`python -m rageval.retrieve "your question"` — inspect retrieval in isolation."""
    import sys

    question = " ".join(sys.argv[1:]) or "which projects use a fruit theme?"
    retriever = Retriever(Settings.load())
    hits = retriever.retrieve(question)
    print(f"Query: {question}\n")
    for i, h in enumerate(hits, start=1):
        preview = h.text.replace("\n", " ")[:110]
        ranks = f"dense={h.dense_rank} sparse={h.sparse_rank}"
        print(f"[{i}] rrf={h.score:.4f} rerank={h.rerank_score:.3f} {ranks} "
              f"{h.source_set}/{h.project_id}#{h.source}::{h.chunk_index}\n    {preview}...\n")


if __name__ == "__main__":
    main()
