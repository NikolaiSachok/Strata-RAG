"""Unit tests for the thin Streamlit UI's PURE helpers (rageval/ui.py, issue #6).

These exercise the request builder, the HTTP-call wrappers, the response→view-model shaping, and
history handling — with a MOCKED transport. There is NO running server, NO browser, and NO live
backend; we never drive the Streamlit runtime (that glue is `main()`, excluded from coverage). The
point is that "the UI surfaces the engine's signals" is asserted against the view-model dicts the
render layer consumes, so the contract is testable without any of Streamlit's machinery.
"""

from __future__ import annotations

import pytest

from rageval import ui


# ---------------------------------------------------------------------------
# A tiny mock transport with the `requests` response/raise_for_status surface.
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, json_body=None, status_ok=True):
        self._json = json_body if json_body is not None else {}
        self._ok = status_ok

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("HTTP 500")

    def json(self):
        return self._json


class RecordingPost:
    """Captures the args of the last call so we can assert the request shape."""
    def __init__(self, response: FakeResponse):
        self._response = response
        self.url = None
        self.json = None
        self.timeout = None
        self.calls = 0

    def __call__(self, url, json=None, timeout=None):  # mimics requests.post signature
        self.calls += 1
        self.url = url
        self.json = json
        self.timeout = timeout
        return self._response


# A representative full /chat response body (the ChatResponse contract from api.py).
CHAT_BODY = {
    "question": "list games and describe one",
    "answer": "Games: Alpha; Alpha is fruit-themed [1].",
    "sources": ["northwind/1 (overview.md)"],
    "trajectory": [
        {"tool": "query_metadata", "args": {"intent": "list"}, "thought": "list the games",
         "result_summary": "2 rows", "ok": True},
        {"tool": "semantic_search", "args": {"query": "describe Alpha"}, "thought": "now describe",
         "result_summary": "1 chunk", "ok": True},
    ],
    "routing": {"mode": "agent", "tool_calls": 2,
                "tools_used": ["query_metadata", "semantic_search"], "hybrid": True},
    "eval": {
        "faithfulness": {"score": 5, "severity": "none", "reason": "grounded"},
        "answer_relevance": {"score": 4, "severity": "minor", "reason": "on topic"},
        "overall_pass": True,
        "findings": [],
    },
    "guardrail": {
        "sentinel_used": True, "safe": True,
        "input_max_severity": "none", "output_max_severity": "none",
        "input_findings": [], "output_findings": [],
        "quarantined_chunks": [], "layers": {"input_scan": True, "output_scan": True},
    },
}


# ===========================================================================
# Base-URL resolution
# ===========================================================================

def test_api_url_default_when_unset():
    assert ui.api_url_from_env({}) == "http://localhost:8000"


def test_api_url_from_env_overrides_and_strips_slash():
    assert ui.api_url_from_env({"RAGEVAL_API_URL": "http://host:9000/"}) == "http://host:9000"


# ===========================================================================
# History handling + request building
# ===========================================================================

def test_append_turn_is_non_mutating():
    h0: list[dict] = []
    h1 = ui.append_turn(h0, "user", "hi")
    h2 = ui.append_turn(h1, "assistant", "hello")
    assert h0 == []                      # original untouched
    assert h1 == [{"role": "user", "content": "hi"}]
    assert h2[-1] == {"role": "assistant", "content": "hello"}
    assert len(h2) == 2


def test_build_chat_request_carries_history():
    history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    body = ui.build_chat_request("q2", history)
    assert body["question"] == "q2"
    assert body["history"] == history
    # A copy, not the same list object, so later mutation of session state can't alias the request.
    assert body["history"] is not history


# ===========================================================================
# call_chat — the HTTP boundary, with a mocked transport
# ===========================================================================

def test_call_chat_posts_to_chat_with_body():
    post = RecordingPost(FakeResponse(CHAT_BODY))
    history = [{"role": "user", "content": "earlier"}]
    out = ui.call_chat(post, "http://localhost:8000", "now", history)
    assert post.url == "http://localhost:8000/chat"
    assert post.json == {"question": "now", "history": history}
    assert post.timeout == 120.0
    assert out["answer"].startswith("Games:")


def test_call_chat_wraps_transport_error_as_apierror():
    def boom(*a, **k):
        raise ConnectionError("refused")
    with pytest.raises(ui.APIError) as ei:
        ui.call_chat(boom, "http://localhost:8000", "q", [])
    assert "failed" in str(ei.value)


def test_call_chat_wraps_non_2xx_as_apierror():
    post = RecordingPost(FakeResponse({}, status_ok=False))
    with pytest.raises(ui.APIError):
        ui.call_chat(post, "http://localhost:8000", "q", [])


# ===========================================================================
# call_health — best-effort, never raises
# ===========================================================================

def test_call_health_returns_body():
    health = {"status": "ok", "chunks_indexed": 42, "pipeline_ready": True,
              "llm": {"backend": "anthropic"}}
    get = RecordingPost(FakeResponse(health))
    assert ui.call_health(get, "http://localhost:8000") == health
    assert get.url == "http://localhost:8000/health"


def test_call_health_returns_none_on_failure():
    def boom(*a, **k):
        raise ConnectionError("down")
    assert ui.call_health(boom, "http://localhost:8000") is None


# ===========================================================================
# Response → view-model shaping (the heart of "surface the signals")
# ===========================================================================

def test_shape_response_surfaces_all_signals():
    v = ui.shape_response(CHAT_BODY)
    assert v["answer"].startswith("Games:")
    assert v["sources"] == ["northwind/1 (overview.md)"]
    # routing
    assert v["routing"]["mode"] == "agent"
    assert v["routing"]["tools_used"] == ["query_metadata", "semantic_search"]
    assert v["routing"]["hybrid"] is True
    assert v["routing"]["tool_calls"] == 2
    # trajectory — ordered, numbered, show-your-work
    assert [s["n"] for s in v["trajectory"]] == [1, 2]
    assert [s["tool"] for s in v["trajectory"]] == ["query_metadata", "semantic_search"]
    assert v["trajectory"][0]["thought"] == "list the games"
    # eval
    assert v["eval"]["overall_pass"] is True
    assert "PASS" in v["eval"]["badge"]
    assert v["eval"]["faithfulness"]["score"] == 5
    assert v["eval"]["answer_relevance"]["severity"] == "minor"
    # guardrail
    assert v["guardrail"]["safe"] is True
    assert v["guardrail"]["layers_fired"] == ["input_scan", "output_scan"]


def test_shape_eval_fail_badge_is_red():
    body = {"eval": {"faithfulness": {"score": 1, "severity": "major", "reason": "hallucinated"},
                     "answer_relevance": {"score": 3, "severity": "minor", "reason": "ok"},
                     "overall_pass": False, "findings": ["unsupported claim"]}}
    ev = ui.shape_eval(body)
    assert ev["overall_pass"] is False
    assert "FAIL" in ev["badge"]
    assert ev["findings"] == ["unsupported claim"]


def test_shape_guardrail_collects_findings_and_unsafe():
    body = {"guardrail": {"safe": False, "layers": {"input_scan": True, "output_scan": False},
                          "input_findings": [{"pattern": "jailbreak", "severity": "high",
                                              "snippet": "ignore previous", "where": "input"}],
                          "output_findings": [{"pattern": "exfil_url", "severity": "high",
                                               "snippet": "http://x", "where": "output"}],
                          "quarantined_chunks": ["c3"]}}
    g = ui.shape_guardrail(body)
    assert g["safe"] is False
    assert g["layers_fired"] == ["input_scan"]
    assert {f["pattern"] for f in g["findings"]} == {"jailbreak", "exfil_url"}
    assert g["quarantined_chunks"] == ["c3"]


def test_shape_routing_handles_ask_shape():
    """/ask carries route/executed_route, not a tools list — still renders a uniform routing view."""
    body = {"routing": {"route": "semantic", "executed_route": "semantic", "method": "rules",
                        "confidence": 0.9, "reasoning": "meaning question", "fell_back": False}}
    r = ui.shape_routing(body)
    assert r["tools_used"] == ["semantic"]
    assert r["hybrid"] is False
    assert r["mode"] in ("rules", "semantic", "—")  # falls back to method when no mode


def test_shape_response_tolerates_missing_blocks():
    """A degraded/partial body must not KeyError the UI (e.g. an /ask response with no trajectory)."""
    v = ui.shape_response({"answer": "hi", "sources": []})
    assert v["answer"] == "hi"
    assert v["trajectory"] == []
    assert v["eval"]["overall_pass"] is False
    assert v["routing"]["tools_used"] == []
    assert v["guardrail"]["safe"] is True


@pytest.mark.parametrize("empty_answer", [None, ""])
def test_empty_answer_does_not_wedge_history(empty_answer):
    """Regression (review M1): a turn whose backend answer is empty/None must not poison the
    replayed transcript. The persisted assistant `content` must be non-empty so the NEXT turn's
    build_chat_request produces a body the API's ChatMessage.content (min_length=1) would accept —
    otherwise every subsequent /chat 422s and the conversation is wedged until "Clear conversation".
    """
    view = ui.shape_response({"answer": empty_answer, "sources": []})
    # This mirrors the persistence in main(): coalesce the assistant turn before appending.
    history = ui.append_turn([], "user", "q")
    history = ui.append_turn(history, "assistant", view["answer"] or "(no answer)")
    assistant_turn = history[-1]
    assert assistant_turn["role"] == "assistant"
    assert assistant_turn["content"]                       # non-empty placeholder, not ""/None
    assert len(assistant_turn["content"]) >= 1             # satisfies API min_length=1
    # And the replayed transcript is a valid request body (non-empty content on every turn).
    body = ui.build_chat_request("next", history)
    assert all(m["content"] for m in body["history"])


@pytest.mark.parametrize("bad_step", [None, "not-a-dict", 42, ["list"]])
def test_shape_trajectory_skips_malformed_step(bad_step):
    """Regression (review M2): a trajectory with a null/non-dict element must degrade gracefully
    (like the empty/absent case) instead of raising AttributeError out of shape_response — that
    exception would escape the APIError handler and error the whole Streamlit run."""
    body = {
        "answer": "ok",
        "trajectory": [
            {"tool": "semantic_search", "args": {}, "thought": "t", "result_summary": "1 chunk",
             "ok": True},
            bad_step,
        ],
    }
    # No exception, and a safe view model: the malformed step becomes a default ("?") row.
    traj = ui.shape_trajectory(body)
    assert [s["n"] for s in traj] == [1, 2]
    assert traj[0]["tool"] == "semantic_search"
    assert traj[1]["tool"] == "?"
    assert traj[1]["ok"] is True
    # And the full shape_response path is likewise exception-free.
    v = ui.shape_response(body)
    assert len(v["trajectory"]) == 2


# ===========================================================================
# Health status line
# ===========================================================================

def test_shape_health_formats_line():
    line = ui.shape_health({"chunks_indexed": 128, "pipeline_ready": True,
                            "llm": {"backend": "claude-cli"}})
    assert "claude-cli" in line
    assert "128" in line
    assert "ready" in line


def test_shape_health_marks_unreachable_index():
    line = ui.shape_health({"chunks_indexed": -1, "pipeline_ready": False, "llm": {"backend": "x"}})
    assert "unreachable" in line
    assert "not ready" in line


def test_shape_health_none_when_absent():
    assert ui.shape_health(None) is None
    assert ui.shape_health({}) is None
