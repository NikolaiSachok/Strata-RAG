"""Tests for the query router + templated-intent executor + dispatch (issue #4).

Four things are covered, all WITHOUT a live LLM backend (the LLM call is mocked):

  1. Routing classification — the rule pre-filter AND the LLM path, across all four routes.
  2. The templated executor — each intent over a small fixture sidecar returns correct rows.
  3. The validation guard — unknown field/intent is REJECTED (never executed); read-only enforced.
  4. The fallback path — an empty/invalid aggregation falls back to semantic, flagged.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from rageval import aggregate, router
from rageval.dispatch import dispatch
from rageval.generate import Answer
from rageval.sidecar import ProjectRecord, connect, upsert_project


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def fixture_sidecar(tmp_path):
    """A tiny, domain-neutral sidecar: 4 projects across 2 source-sets and 2 publishers."""
    db = tmp_path / "side.sqlite"
    conn = connect(db)
    rows = [
        ProjectRecord(project_id="1", source_set="northwind", publisher="Maple",
                      app_category="game", app_name="Alpha", chunk_count=3),
        ProjectRecord(project_id="2", source_set="northwind", publisher="Maple",
                      app_category="utility", app_name="Beta", chunk_count=5),
        ProjectRecord(project_id="3", source_set="northwind", publisher="Cedar",
                      app_category="game", app_name="Gamma", chunk_count=2),
        ProjectRecord(project_id="4", source_set="atlas", publisher="Cedar",
                      app_category="game", app_name="Delta", chunk_count=1, status="banned"),
    ]
    for r in rows:
        upsert_project(conn, r)
    conn.close()
    return db


class FakeRouterLLM:
    """A scripted LLM backend: returns a canned JSON route verdict. Lets us exercise the
    LLM classification path with no live model."""
    name = "fake"

    def __init__(self, payload: dict):
        self._payload = payload

    def complete(self, system: str, prompt: str, max_tokens: int = 200) -> str:
        return json.dumps(self._payload)


# ===========================================================================
# 1. Routing classification
# ===========================================================================

@pytest.mark.parametrize("question,expected_match", [
    ("How many projects are there?", "how many"),
    ("Count the projects per publisher", "count"),
    ("list all categories", "list all"),
    ("number of banned apps", "number of"),
    ("projects per publisher", "per publisher"),
    ("top 3 publishers by project count", "top 3"),
    ("what is the average chunk_count", "average"),
])
def test_rule_prefilter_catches_aggregation_phrasings(question, expected_match):
    d = router.rule_prefilter(question)
    assert d is not None
    assert d.route == "aggregation"
    assert d.method == "rule"


def test_rule_prefilter_defers_semantic_questions():
    # A meaning question has no aggregation phrasing → the pre-filter declines (returns None),
    # deferring to the LLM classifier.
    assert router.rule_prefilter("which projects have a fruit-like theme?") is None


def test_route_uses_rule_prefilter_without_llm():
    # The rule tier short-circuits BEFORE any LLM call: no backend needed.
    d = router.route("how many projects per publisher?", llm=None)
    assert d.route == "aggregation" and d.method == "rule"


def test_route_defaults_to_semantic_when_no_rule_and_no_llm():
    d = router.route("which apps feel playful?", llm=None)
    assert d.route == "semantic" and d.method == "fallback"


@pytest.mark.parametrize("route_name", ["semantic", "aggregation", "lookup", "hybrid"])
def test_llm_classifier_returns_each_route(route_name):
    # Force the LLM path (use_rules=False) so the classifier — not the pre-filter — decides.
    fake = FakeRouterLLM({"route": route_name, "confidence": 0.8, "reasoning": "x",
                          "intent": "count" if route_name == "aggregation" else None})
    d = router.route("some question", llm=fake, use_rules=False)
    assert d.route == route_name
    assert d.method == "llm"
    assert 0.0 <= d.confidence <= 1.0


def test_llm_classifier_extracts_aggregation_slots():
    fake = FakeRouterLLM({"route": "aggregation", "confidence": 0.9, "reasoning": "group",
                          "intent": "group_by_count", "field": "publisher", "filter": None})
    d = router.route("publishers and their counts", llm=fake, use_rules=False)
    assert d.slots["intent"] == "group_by_count"
    assert d.slots["field"] == "publisher"


def test_llm_garbled_route_defaults_to_semantic():
    fake = FakeRouterLLM({"route": "nonsense", "confidence": 2.0, "reasoning": ""})
    d = router.route("q", llm=fake, use_rules=False)
    assert d.route == "semantic"
    assert d.confidence == 1.0  # clamped from 2.0


def test_llm_backend_exception_is_swallowed_to_semantic():
    class BoomLLM:
        name = "boom"
        def complete(self, *a, **k):
            raise RuntimeError("backend down")
    d = router.route("q", llm=BoomLLM(), use_rules=False)
    assert d.route == "semantic" and d.method == "fallback"


# ===========================================================================
# 2. The templated executor — each intent over the fixture sidecar
# ===========================================================================

def test_count_intent(fixture_sidecar):
    res = aggregate.execute("count", sidecar_path=fixture_sidecar)
    assert res.intent == "count"
    assert res.rows[0]["count"] == 4
    assert "LIMIT" in res.executed_query


def test_count_with_filter(fixture_sidecar):
    res = aggregate.execute("count", filter={"app_category": "game"}, sidecar_path=fixture_sidecar)
    assert res.rows[0]["count"] == 3
    # The filter VALUE is bound as a parameter, never interpolated into the SQL string.
    assert "game" not in res.executed_query
    assert "game" in res.params


def test_list_distinct(fixture_sidecar):
    res = aggregate.execute("list", field="publisher", sidecar_path=fixture_sidecar)
    vals = [r["publisher"] for r in res.rows]
    assert vals == ["Cedar", "Maple"]  # DISTINCT + ORDER BY


def test_group_by_count(fixture_sidecar):
    res = aggregate.execute("group_by_count", field="publisher", sidecar_path=fixture_sidecar)
    counts = {r["publisher"]: r["count"] for r in res.rows}
    assert counts == {"Maple": 2, "Cedar": 2}


def test_top_n_orders_by_count(fixture_sidecar):
    res = aggregate.execute("top_n", field="app_category", limit=1, sidecar_path=fixture_sidecar)
    assert res.row_count == 1
    assert res.rows[0]["app_category"] == "game"  # 3 games is the largest group
    assert res.rows[0]["count"] == 3


def test_lookup_single_row(fixture_sidecar):
    res = aggregate.execute("lookup", filter={"key": "northwind/2"},
                            sidecar_path=fixture_sidecar)
    assert res.row_count == 1
    assert res.rows[0]["app_name"] == "Beta"


def test_lookup_single_field(fixture_sidecar):
    res = aggregate.execute("lookup", field="app_name", filter={"project_id": "3"},
                            sidecar_path=fixture_sidecar)
    assert res.rows[0]["app_name"] == "Gamma"


# ===========================================================================
# 3. The validation guard
# ===========================================================================

def test_unknown_field_rejected(fixture_sidecar):
    with pytest.raises(aggregate.AggregateError):
        aggregate.execute("list", field="evil_column", sidecar_path=fixture_sidecar)


def test_unknown_intent_rejected(fixture_sidecar):
    with pytest.raises(aggregate.AggregateError):
        aggregate.execute("drop_table", sidecar_path=fixture_sidecar)


def test_unknown_filter_field_rejected(fixture_sidecar):
    with pytest.raises(aggregate.AggregateError):
        aggregate.execute("count", filter={"evil; DROP": "x"}, sidecar_path=fixture_sidecar)


def test_injection_value_is_bound_not_executed(fixture_sidecar):
    # A SQL-injection-shaped VALUE is bound as a parameter → it matches nothing, it does NOT
    # alter the query. The table is untouched and the count is 0.
    res = aggregate.execute("count", filter={"publisher": "'; DROP TABLE projects;--"},
                            sidecar_path=fixture_sidecar)
    assert res.rows[0]["count"] == 0
    # Prove the table still exists with all rows (the injection did not execute).
    conn = connect(fixture_sidecar)
    assert conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 4
    conn.close()


def test_connection_is_read_only(fixture_sidecar):
    # The aggregation path opens the sidecar mode=ro; a write must fail at the driver level.
    conn = aggregate._connect_readonly(fixture_sidecar)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM projects")
    conn.close()


def test_allowed_fields_derived_from_record():
    # The whitelist tracks the sidecar's ProjectRecord (single source of truth), so a schema
    # change updates the guard automatically.
    assert "publisher" in aggregate.ALLOWED_FIELDS
    assert "app_category" in aggregate.ALLOWED_FIELDS
    assert "key" in aggregate.ALLOWED_FIELDS
    assert "evil_column" not in aggregate.ALLOWED_FIELDS


def test_lookup_requires_filter(fixture_sidecar):
    with pytest.raises(aggregate.AggregateError):
        aggregate.execute("lookup", sidecar_path=fixture_sidecar)


# ===========================================================================
# 4. Dispatch + the fallback path
# ===========================================================================

class StubPipeline:
    """A minimal RagPipeline stand-in: records that the semantic path was used and returns a
    canned Answer. Lets us assert dispatch's routing/fallback WITHOUT Qdrant or an LLM."""
    def __init__(self, llm=None):
        self.llm = llm
        self.semantic_calls = 0

    def answer(self, question: str) -> Answer:
        self.semantic_calls += 1
        return Answer(question=question, answer="semantic answer", sources=["s/1 (x)"], chunks=[])


def _monkeypatch_sidecar(monkeypatch, db):
    """Point aggregate.execute's default sidecar at the fixture DB (dispatch calls it without an
    explicit path)."""
    monkeypatch.setattr(aggregate, "SIDECAR_PATH", db)


def test_dispatch_aggregation_runs_templated_query(fixture_sidecar, monkeypatch):
    _monkeypatch_sidecar(monkeypatch, fixture_sidecar)
    fake = FakeRouterLLM({"route": "aggregation", "confidence": 0.95, "reasoning": "count",
                          "intent": "count", "field": None, "filter": None})
    pipe = StubPipeline(llm=fake)
    ans = dispatch("how many projects total?", pipe, use_rules=False)
    assert ans.routing["executed_route"] == "aggregation"
    assert ans.routing["fell_back"] is False
    assert "executed_query" in ans.routing
    assert pipe.semantic_calls == 0  # the sidecar answered; semantic was NOT touched


def test_dispatch_semantic_path_attaches_routing(fixture_sidecar, monkeypatch):
    _monkeypatch_sidecar(monkeypatch, fixture_sidecar)
    fake = FakeRouterLLM({"route": "semantic", "confidence": 0.7, "reasoning": "theme"})
    pipe = StubPipeline(llm=fake)
    ans = dispatch("which apps feel playful?", pipe, use_rules=False)
    assert ans.routing["executed_route"] == "semantic"
    assert pipe.semantic_calls == 1


def test_dispatch_empty_aggregation_falls_back_to_semantic(fixture_sidecar, monkeypatch):
    _monkeypatch_sidecar(monkeypatch, fixture_sidecar)
    # A list whose filter matches nothing → zero rows → fall back to semantic, FLAGGED.
    fake = FakeRouterLLM({"route": "aggregation", "confidence": 0.9, "reasoning": "list",
                          "intent": "list", "field": "publisher",
                          "filter": {"app_category": "does-not-exist"}})
    pipe = StubPipeline(llm=fake)
    ans = dispatch("list publishers of nonexistent category", pipe, use_rules=False)
    assert ans.routing["fell_back"] is True
    assert ans.routing["executed_route"] == "semantic"
    assert pipe.semantic_calls == 1
    assert ans.answer == "semantic answer"


def test_dispatch_invalid_slot_falls_back_to_semantic(fixture_sidecar, monkeypatch):
    _monkeypatch_sidecar(monkeypatch, fixture_sidecar)
    # The LLM proposed an unknown field → aggregate.execute rejects it → fall back.
    fake = FakeRouterLLM({"route": "aggregation", "confidence": 0.9, "reasoning": "list",
                          "intent": "list", "field": "evil_column", "filter": None})
    pipe = StubPipeline(llm=fake)
    ans = dispatch("list evil things", pipe, use_rules=False)
    assert ans.routing["fell_back"] is True
    assert pipe.semantic_calls == 1


def test_dispatch_aggregation_without_intent_falls_back(fixture_sidecar, monkeypatch):
    _monkeypatch_sidecar(monkeypatch, fixture_sidecar)
    # Routed aggregation but no intent slot → nothing to execute → semantic fallback.
    fake = FakeRouterLLM({"route": "aggregation", "confidence": 0.5, "reasoning": "?",
                          "intent": None, "field": None, "filter": None})
    pipe = StubPipeline(llm=fake)
    ans = dispatch("something structural but vague", pipe, use_rules=False)
    assert ans.routing["fell_back"] is True
    assert pipe.semantic_calls == 1


def test_dispatch_hybrid_is_labelled(fixture_sidecar, monkeypatch):
    _monkeypatch_sidecar(monkeypatch, fixture_sidecar)
    fake = FakeRouterLLM({"route": "hybrid", "confidence": 0.6, "reasoning": "both",
                          "intent": None, "field": None, "filter": None})
    pipe = StubPipeline(llm=fake)
    ans = dispatch("themes used by a given publisher", pipe, use_rules=False)
    assert ans.routing["executed_route"] == "hybrid"
    assert "note" in ans.routing  # #4 labels hybrid; #5 will decompose
    assert pipe.semantic_calls == 1


def test_routing_block_shape_is_json_serializable(fixture_sidecar, monkeypatch):
    _monkeypatch_sidecar(monkeypatch, fixture_sidecar)
    fake = FakeRouterLLM({"route": "aggregation", "confidence": 0.95, "reasoning": "count",
                          "intent": "count", "field": None, "filter": None})
    ans = dispatch("how many total?", StubPipeline(llm=fake), use_rules=False)
    # The block must round-trip through JSON (it's serialized into the API response).
    blob = json.dumps(ans.routing)
    back = json.loads(blob)
    assert set(["route", "confidence", "reasoning", "method", "executed_route", "fell_back"]) <= set(back)


# ===========================================================================
# 5. Route-aware eval end-to-end (issue #16): a DISPATCHED aggregation answer,
#    graded by the REAL Judge, must PASS — not fail on a bogus faithfulness flag.
# ===========================================================================

def test_dispatched_aggregation_answer_passes_judge(fixture_sidecar, monkeypatch):
    """The bug: the judge scored a sidecar aggregation answer faithfulness:critical (empty
    CONTEXT) and FAILED it. End-to-end: dispatch an aggregation, then run the route-aware Judge
    with a scripted relevance verdict — faithfulness is skipped and the answer passes."""
    from rageval.eval import NOT_APPLICABLE, Judge

    _monkeypatch_sidecar(monkeypatch, fixture_sidecar)
    router_llm = FakeRouterLLM({"route": "aggregation", "confidence": 0.95, "reasoning": "count",
                                "intent": "group_by_count", "field": "publisher", "filter": None})
    pipe = StubPipeline(llm=router_llm)
    ans = dispatch("how many projects per publisher?", pipe, use_rules=False)
    assert ans.routing["executed_route"] == "aggregation"
    assert ans.sources == []  # the sidecar answered — empty sources is CORRECT here

    judge_llm = FakeRouterLLM(
        {"answer_relevance": {"score": 5, "severity": "none", "reason": "Answers it."},
         "findings": []})
    verdict = Judge(llm=judge_llm).evaluate(ans)
    assert verdict.faithfulness.severity == NOT_APPLICABLE
    assert verdict.overall_pass is True
