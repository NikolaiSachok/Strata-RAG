"""Retrieval metrics — Recall@K, Precision@K, MRR, nDCG.

These four numbers are how you know whether retrieval actually works, turned from a vibe
into a measurement. All are computed over a GOLDEN SET (eval/golden.yaml): for each
question, a human labelled which project_ids are relevant. We compare the retriever's
ranked output against those labels.

All functions here are PURE (ranked ids + relevant set → float) so they're trivially
unit-testable on fixtures, with no models or I/O. That's deliberate: metric code you
can't test is metric code you can't trust.

Definitions (let `ranked` = retrieved ids in rank order, `rel` = set of relevant ids):

  Recall@K    — of all relevant items, what fraction appear in the top K?
                "did we FIND the right stuff?" (the ceiling on answer quality).
  Precision@K — of the top K retrieved, what fraction are relevant?
                "how much of what we showed was junk?"
  MRR         — Mean Reciprocal Rank: 1 / (rank of the FIRST relevant item). Rewards
                putting a correct item high; great for "is the best answer near the top?".
  nDCG@K      — Normalized Discounted Cumulative Gain: like recall but RANK-AWARE — a
                relevant item at rank 1 is worth more than at rank 5 (log discount),
                normalized so 1.0 = the ideal ordering. The richest single ranking metric.
"""

from __future__ import annotations

import math


def _dedup_keep_order(ids: list[str]) -> list[str]:
    """Collapse a ranked chunk/id list to distinct items in first-seen order.

    Retrieval returns chunks, but the golden labels are at PROJECT granularity, so we
    dedup to the first occurrence of each project id before scoring."""
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = set(_dedup_keep_order(ranked)[:k])
    return len(topk & relevant) / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = _dedup_keep_order(ranked)[:k]
    if not topk:
        return 0.0
    return sum(1 for i in topk if i in relevant) / len(topk)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for rank, i in enumerate(_dedup_keep_order(ranked), start=1):
        if i in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@K. DCG = sum over top-k of rel_i / log2(rank+1); IDCG is the
    DCG of the best possible ordering (all relevant items first). nDCG = DCG / IDCG."""
    topk = _dedup_keep_order(ranked)[:k]
    dcg = sum((1.0 if i in relevant else 0.0) / math.log2(rank + 1)
              for rank, i in enumerate(topk, start=1))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


def aggregate(per_question: list[dict]) -> dict:
    """Average each metric across questions. `per_question` = list of metric dicts."""
    if not per_question:
        return {}
    keys = per_question[0].keys()
    return {k: sum(q[k] for q in per_question) / len(per_question) for k in keys}
