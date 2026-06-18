"""Tests for RRF fusion and the retrieval metrics — both pure math, both load-bearing.

If RRF or the metric formulas are wrong, every reported number is wrong, so these get
hand-computed fixtures with known answers.
"""

from __future__ import annotations

import math

from rageval.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rageval.retrieve import Retrieved, apply_rerank_floor, reciprocal_rank_fusion


# ---- RRF -------------------------------------------------------------------

def test_rrf_rewards_items_ranked_high_by_both_lists():
    dense = ["a", "b", "c"]
    sparse = ["b", "a", "d"]
    scores = reciprocal_rank_fusion([dense, sparse], k=60)
    # 'a': 1/61 + 1/62 ; 'b': 1/62 + 1/61  → equal and highest.
    assert math.isclose(scores["a"], scores["b"])
    assert scores["a"] > scores["c"]
    assert scores["a"] > scores["d"]


def test_rrf_exact_value():
    scores = reciprocal_rank_fusion([["x"]], k=60)
    assert math.isclose(scores["x"], 1.0 / 61)


def test_rrf_missing_from_one_list_still_scores():
    scores = reciprocal_rank_fusion([["a", "b"], ["c"]], k=60)
    assert set(scores) == {"a", "b", "c"}


# ---- rerank-score floor (top-k WITH a relevance threshold) -----------------

def _hit(score: float) -> Retrieved:
    """A minimal Retrieved carrying a rerank_score; other fields don't matter here."""
    return Retrieved(
        text="x", project_id="p", source_set="s", source="src",
        doc_type="d", chunk_index=0, score=0.0, rerank_score=score,
    )


def test_floor_drops_subthreshold_hits_and_preserves_order():
    # Already in rerank order (descending). Floor at 0.5 keeps the first two, in order.
    hits = [_hit(0.9), _hit(0.6), _hit(0.3), _hit(0.1)]
    out = apply_rerank_floor(hits, 0.5)
    assert [h.rerank_score for h in out] == [0.9, 0.6]  # order preserved, sub-floor dropped


def test_floor_none_is_unchanged():
    hits = [_hit(0.9), _hit(0.1), _hit(-2.0)]
    out = apply_rerank_floor(hits, None)
    assert out is hits or [h.rerank_score for h in out] == [0.9, 0.1, -2.0]


def test_floor_can_empty_the_result():
    # A floor above every score yields an empty set on purpose (no min-1 fallback) — this is
    # the "no relevant context" signal, not a bug.
    hits = [_hit(0.4), _hit(0.2)]
    assert apply_rerank_floor(hits, 0.99) == []


def test_floor_is_inclusive_at_the_boundary():
    hits = [_hit(0.5), _hit(0.4999)]
    out = apply_rerank_floor(hits, 0.5)
    assert [h.rerank_score for h in out] == [0.5]  # >= floor kept, just-below dropped


def test_floor_disabled_by_default_in_settings(monkeypatch):
    # The knob must default to None (disabled) so current behaviour is unchanged when unset.
    from rageval.config import Settings

    monkeypatch.delenv("RAGEVAL_MIN_RERANK_SCORE", raising=False)
    assert Settings.load().min_rerank_score is None
    monkeypatch.setenv("RAGEVAL_MIN_RERANK_SCORE", "0.25")
    assert Settings.load().min_rerank_score == 0.25


# ---- metrics ---------------------------------------------------------------

def test_recall_at_k():
    ranked = ["p1", "p2", "p3", "p4"]
    rel = {"p2", "p4", "p9"}  # p9 is relevant but never retrieved
    assert math.isclose(recall_at_k(ranked, rel, k=4), 2 / 3)
    assert math.isclose(recall_at_k(ranked, rel, k=2), 1 / 3)  # only p2 in top-2


def test_precision_at_k():
    ranked = ["p1", "p2", "p3", "p4"]
    rel = {"p2", "p4"}
    assert math.isclose(precision_at_k(ranked, rel, k=4), 0.5)
    assert math.isclose(precision_at_k(ranked, rel, k=2), 0.5)


def test_mrr_uses_first_relevant_rank():
    assert math.isclose(reciprocal_rank(["x", "y", "rel"], {"rel"}), 1 / 3)
    assert reciprocal_rank(["x", "y"], {"rel"}) == 0.0


def test_ndcg_perfect_and_discounted():
    # All relevant items first → nDCG = 1.0
    assert math.isclose(ndcg_at_k(["a", "b"], {"a", "b"}, k=2), 1.0)
    # One relevant item at rank 2 only: DCG = 1/log2(3); IDCG = 1/log2(2)=1
    expected = (1 / math.log2(3)) / 1.0
    assert math.isclose(ndcg_at_k(["x", "a"], {"a"}, k=2), expected)


def test_metrics_dedup_chunks_to_projects():
    # The retriever returns multiple chunks per project; metrics must dedup to projects.
    ranked = ["p1", "p1", "p1", "p2"]
    assert math.isclose(precision_at_k(ranked, {"p1", "p2"}, k=2), 1.0)
