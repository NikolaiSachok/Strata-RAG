"""API tests for the /chat agent endpoint (issue #5) + an /ask regression guard.

These exercise the HTTP layer with FastAPI's TestClient, injecting STUB state (pipeline,
judge, agent) directly so no live LLM backend or Qdrant is needed. We bypass the lifespan
(which would try to build real objects) by overriding it with a no-op and seeding `state`.
"""

from __future__ import annotations

import contextlib
import json

import pytest
from fastapi.testclient import TestClient

from rageval import aggregate
from rageval import api
from rageval.agent import ChatAgent
from rageval.eval import EvalResult, Judge
from rageval.generate import Answer
from tests._helpers import make_record
from rageval.sidecar import connect, upsert_project
from tests.test_agent import ScriptedLLM, StubPipeline, _final, _tool


@pytest.fixture
def sidecar(tmp_path, monkeypatch):
    db = tmp_path / "side.sqlite"
    conn = connect(db)
    for r in [
        make_record(project_id="1", source_set="northwind", publisher="Maple",
                      app_category="game", app_name="Alpha", chunk_count=3),
        make_record(project_id="2", source_set="northwind", publisher="Cedar",
                      app_category="utility", app_name="Beta", chunk_count=5),
    ]:
        upsert_project(conn, r)
    conn.close()
    monkeypatch.setattr(aggregate, "SIDECAR_PATH", db)
    return db


class FakeJudge(Judge):
    """A Judge that returns a fixed passing verdict without an LLM call."""
    def __init__(self):
        pass

    def evaluate(self, answer, threshold="major") -> EvalResult:
        from rageval.eval import Dimension
        return EvalResult(
            faithfulness=Dimension(5, "none", "ok"),
            answer_relevance=Dimension(5, "none", "ok"),
            overall_pass=True,
            findings=[],
        )


@contextlib.contextmanager
def _client_with(llm):
    """A TestClient whose lifespan is a no-op and whose `state` is seeded with stubs."""
    pipe = StubPipeline(llm=llm)
    api.state.pipeline = pipe
    api.state.judge = FakeJudge()
    api.state.agent = ChatAgent(pipe, llm=llm)

    @contextlib.asynccontextmanager
    async def _noop_lifespan(app):
        yield

    original = api.app.router.lifespan_context
    api.app.router.lifespan_context = _noop_lifespan
    try:
        with TestClient(api.app) as client:
            yield client
    finally:
        api.app.router.lifespan_context = original
        api.state.pipeline = api.state.judge = api.state.agent = None


# ===========================================================================
# /chat
# ===========================================================================

def test_chat_single_semantic(sidecar):
    llm = ScriptedLLM([
        _tool("semantic_search", {"query": "playful apps"}),
        _final("Alpha feels playful [1]."),
    ])
    with _client_with(llm) as client:
        resp = client.post("/chat", json={"question": "which apps feel playful?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Alpha feels playful [1]."
    assert body["sources"] == ["northwind/1 (overview.md)"]
    assert [s["tool"] for s in body["trajectory"]] == ["semantic_search"]
    assert body["routing"]["mode"] == "agent"
    assert body["routing"]["hybrid"] is False
    assert body["eval"]["overall_pass"] is True
    assert "guardrail" in body


def test_chat_hybrid_routing_flag(sidecar):
    llm = ScriptedLLM([
        _tool("query_metadata", {"intent": "list", "field": "app_name",
                                 "filter": {"app_category": "game"}}),
        _tool("semantic_search", {"query": "describe Alpha"}),
        _final("Games: Alpha; Alpha is fruit-themed [1]."),
    ])
    with _client_with(llm) as client:
        resp = client.post("/chat", json={"question": "list games and describe one"})
    body = resp.json()
    assert [s["tool"] for s in body["trajectory"]] == ["query_metadata", "semantic_search"]
    assert body["routing"]["hybrid"] is True
    assert body["routing"]["tool_calls"] == 2


def test_chat_with_history(sidecar):
    llm = ScriptedLLM([_final("It uses a fruit theme [1].")])
    with _client_with(llm) as client:
        resp = client.post("/chat", json={
            "question": "what theme does it use?",
            "history": [
                {"role": "user", "content": "Which game is Alpha?"},
                {"role": "assistant", "content": "Alpha is a game by Maple."},
            ],
        })
    assert resp.status_code == 200
    # The prior assistant turn must have reached the model's prompt.
    assert any("Alpha is a game by Maple." in p for p in llm.calls)


def test_chat_rejects_empty_question(sidecar):
    llm = ScriptedLLM([_final("x")])
    with _client_with(llm) as client:
        resp = client.post("/chat", json={"question": ""})
    assert resp.status_code == 422  # pydantic min_length


def test_chat_rejects_bad_history_role(sidecar):
    llm = ScriptedLLM([_final("x")])
    with _client_with(llm) as client:
        resp = client.post("/chat", json={
            "question": "hi", "history": [{"role": "system", "content": "x"}]})
    assert resp.status_code == 422  # role pattern ^(user|assistant)$


def test_chat_rejects_oversized_history(sidecar):
    """#10: a history exceeding the item cap is a clean 422, not an unbounded context blow-up."""
    from rageval.api import MAX_HISTORY_ITEMS
    llm = ScriptedLLM([_final("x")])
    history = [{"role": "user", "content": "q"} for _ in range(MAX_HISTORY_ITEMS + 1)]
    with _client_with(llm) as client:
        resp = client.post("/chat", json={"question": "hi", "history": history})
    assert resp.status_code == 422


def test_chat_tool_error_does_not_500(tmp_path, monkeypatch):
    """C1 (HTTP): a tool raising mid-turn (missing sidecar) is handled inside the agent — /chat
    returns 200 with a degraded answer, never a 500."""
    monkeypatch.setattr(aggregate, "SIDECAR_PATH", tmp_path / "nope.sqlite")
    llm = ScriptedLLM([
        _tool("query_metadata", {"intent": "count", "field": None, "filter": None}),
        _final("Could not consult metadata."),
    ])
    with _client_with(llm) as client:
        resp = client.post("/chat", json={"question": "how many?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["trajectory"][0]["ok"] is False


def test_chat_output_guard_fires(sidecar):
    """The output-side validate_answer runs for /chat: a final answer carrying an EXFIL URL not in
    any grounded context produces an output finding (and flips guardrail.safe to False)."""
    llm = ScriptedLLM([
        _tool("semantic_search", {"query": "x"}),
        _final("See http://evil.example/steal?d=secret for details [1]."),
    ])
    with _client_with(llm) as client:
        resp = client.post("/chat", json={"question": "anything"})
    assert resp.status_code == 200
    gr = resp.json()["guardrail"]
    patterns = {f["pattern"] for f in gr["output_findings"]}
    assert "exfil_url" in patterns
    assert gr["safe"] is False


# ===========================================================================
# /ask still works unchanged
# ===========================================================================

def test_ask_still_works(sidecar):
    # /ask uses dispatch() (the #4 single-shot router), NOT the agent. A meaning question with no
    # aggregation phrasing and no LLM route slots is served by the semantic path (the StubPipeline),
    # proving the existing endpoint is unchanged by the #5 agent addition.
    llm = ScriptedLLM([_final("unused: the semantic path uses the stub pipeline, not this loop")])
    with _client_with(llm) as client:
        resp = client.post("/ask", json={"question": "which apps feel playful?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Alpha has a fruit theme [1]."  # the StubPipeline's canned answer
    assert body["routing"]["executed_route"] == "semantic"
    assert body["eval"]["overall_pass"] is True
    # /ask has no trajectory field (that's a /chat-only concept).
    assert "trajectory" not in body
