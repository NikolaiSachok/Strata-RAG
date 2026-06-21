"""Optional Streamlit UI **over the running API** — a thin front end so you can *see* the engine.

WHY this exists (and why it's deliberately thin):
  The engine's distinctive value is not "it answers questions" — it's the *signals* it exposes
  around each answer: which route/tools served it, the agent's step-by-step trajectory, the
  LLM-as-judge eval verdict, and the guardrail report. Those are invisible on the command line.
  This UI makes them visible and expandable.

  It is a UI *over the API*, NOT a re-implementation of RAG. It holds **no** retrieval/eval/agent
  logic. It POSTs to ``/chat`` on the running FastAPI service and renders whatever comes back.
  That separation (UI talks to API, API owns the logic) is the same shape you'd use in production.

DESIGN: the HTTP call and the response→view-model shaping are factored into **plain functions**
(below) that take/return dicts and a tiny injected ``post`` callable. They are unit-tested with a
mocked transport — NO running server, NO browser. The Streamlit glue at the bottom is then just
"call helper, render dict", and is skipped entirely unless Streamlit is installed and this module
is run as a Streamlit app.

Run it with the API already serving on :8000::

    pip install -e ".[ui]"
    uvicorn rageval.api:app             # terminal 1
    streamlit run rageval/ui.py         # terminal 2  (or: python -m streamlit run rageval/ui.py)

Point it at a non-default backend with the ``RAGEVAL_API_URL`` env var or the sidebar field.
"""

from __future__ import annotations

import os
from typing import Any, Callable

# Default backend. Overridable via env (read at import) or the sidebar field (read per-run).
DEFAULT_API_URL = "http://localhost:8000"


def api_url_from_env(environ: dict[str, str] | None = None) -> str:
    """Resolve the backend base URL from the environment, falling back to the default.

    Pure + injectable so it's testable without touching the real process environment. A trailing
    slash is stripped so callers can safely append ``/chat`` etc."""
    env = environ if environ is not None else os.environ
    return env.get("RAGEVAL_API_URL", DEFAULT_API_URL).rstrip("/")


# ---------------------------------------------------------------------------
# History handling — the multi-turn transcript the client owns and replays.
# ---------------------------------------------------------------------------
# The server is stateless: it never stores conversation state, so the UI keeps the running
# transcript and replays it on every /chat call (matching the API's ChatRequest.history contract).
# These helpers are pure list transforms so the session-state glue stays trivial and testable.


def append_turn(history: list[dict[str, str]], role: str, content: str) -> list[dict[str, str]]:
    """Return a NEW history with one ``{role, content}`` turn appended (oldest first).

    Non-mutating so Streamlit's session_state assignment stays explicit and the function is a pure,
    testable transform."""
    return [*history, {"role": role, "content": content}]


def build_chat_request(question: str, history: list[dict[str, str]]) -> dict[str, Any]:
    """Shape the POST /chat request body from the new question + prior transcript.

    Mirrors the API's ``ChatRequest`` schema ({question, history}). Kept separate from the HTTP call
    so the request-building is unit-testable on its own."""
    return {"question": question, "history": list(history)}


# ---------------------------------------------------------------------------
# The HTTP call — the ONLY place the UI talks to the network.
# ---------------------------------------------------------------------------
# `post` / `get` are injected (a callable with the `requests` signature) so tests pass a mock and
# never open a socket. In the live app we bind them to `requests.post` / `requests.get`.


class APIError(RuntimeError):
    """A backend call failed (transport error or non-2xx). Carries a human-readable message that
    the UI surfaces verbatim — the UI never needs to know *how* it failed, only what to show."""


def call_chat(
    post: Callable[..., Any],
    base_url: str,
    question: str,
    history: list[dict[str, str]],
    timeout: float = 120.0,
) -> dict[str, Any]:
    """POST one turn to ``{base_url}/chat`` and return the parsed JSON body.

    ``post`` is the injected transport (``requests.post`` in the app, a mock in tests). Any
    transport failure or non-2xx response is normalised into an ``APIError`` with a readable
    message, so the render layer has a single, simple failure mode to handle."""
    body = build_chat_request(question, history)
    try:
        resp = post(f"{base_url}/chat", json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except APIError:
        raise
    except Exception as e:  # noqa: BLE001 — collapse any transport/parse error into one UI error.
        raise APIError(f"Request to {base_url}/chat failed: {e}") from e


def call_health(
    get: Callable[..., Any],
    base_url: str,
    timeout: float = 5.0,
) -> dict[str, Any] | None:
    """GET ``{base_url}/health`` and return the parsed body, or ``None`` if it's unreachable.

    Health is best-effort status decoration (backend + chunks_indexed), so a failure returns None
    rather than raising — the UI just omits the status line instead of erroring the whole page."""
    try:
        resp = get(f"{base_url}/health", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:  # noqa: BLE001 — health is optional; never let it break the page.
        return None


# ---------------------------------------------------------------------------
# Response → view-model shaping (pure). The render layer consumes these dicts.
# ---------------------------------------------------------------------------
# These translate the API's ChatResponse into small, render-ready view models. Keeping them pure
# (dict in, dict out) is what lets us unit-test "the UI surfaces the right signals" with zero
# Streamlit/browser involvement: assert on the view model, not on pixels.


def shape_eval(data: dict[str, Any]) -> dict[str, Any]:
    """View model for the eval verdict badge + per-dimension metrics.

    ``overall_pass`` drives a green/red badge; faithfulness + answer_relevance give score/severity/
    reason. Tolerant of a missing ``eval`` block (older/degraded responses) so the UI never KeyErrors."""
    ev = data.get("eval") or {}
    faith = ev.get("faithfulness") or {}
    rel = ev.get("answer_relevance") or {}
    overall = bool(ev.get("overall_pass", False))
    return {
        "overall_pass": overall,
        "badge": ("PASS — grounded & relevant" if overall else "FAIL — flagged by the judge"),
        "faithfulness": {
            "score": faith.get("score"),
            "severity": faith.get("severity", ""),
            "reason": faith.get("reason", ""),
        },
        "answer_relevance": {
            "score": rel.get("score"),
            "severity": rel.get("severity", ""),
            "reason": rel.get("reason", ""),
        },
        "findings": list(ev.get("findings") or []),
    }


def shape_routing(data: dict[str, Any]) -> dict[str, Any]:
    """View model for the routing block — *which path answered*.

    Handles BOTH shapes: /chat's agent summary ({mode, tools_used, tool_calls, hybrid}) and /ask's
    single decision ({route, executed_route, ...}). Returns a normalised, render-ready dict and
    keeps the raw block for the expandable detail view."""
    r = data.get("routing") or {}
    mode = r.get("mode") or r.get("method") or "—"
    tools_used = list(r.get("tools_used") or [])
    # /ask carries route/executed_route instead of a tools list; fold that in for a uniform display.
    if not tools_used and r.get("executed_route"):
        tools_used = [r["executed_route"]]
    return {
        "mode": mode,
        "tools_used": tools_used,
        "tool_calls": r.get("tool_calls", len(tools_used)),
        "hybrid": bool(r.get("hybrid", False)),
        "raw": r,
    }


def shape_trajectory(data: dict[str, Any]) -> list[dict[str, Any]]:
    """View model for the agent's 'show your work' timeline.

    One render-ready row per tool call — tool, args, thought, result_summary, ok — in execution
    order. Empty when the response carries no trajectory (e.g. an /ask response)."""
    steps = data.get("trajectory") or []
    out: list[dict[str, Any]] = []
    for i, s in enumerate(steps, start=1):
        out.append(
            {
                "n": i,
                "tool": s.get("tool", "?"),
                "args": s.get("args", {}),
                "thought": s.get("thought", ""),
                "result_summary": s.get("result_summary", ""),
                "ok": bool(s.get("ok", True)),
            }
        )
    return out


def shape_guardrail(data: dict[str, Any]) -> dict[str, Any]:
    """View model for the guardrail report — which layers fired, whether the turn is safe, findings.

    Surfaces ``safe`` (the headline), the per-layer on/off map, and the input/output findings so a
    user can SEE the injection defenses ran and what (if anything) they caught."""
    g = data.get("guardrail") or {}
    layers = g.get("layers") or {}
    findings = list(g.get("input_findings") or []) + list(g.get("output_findings") or [])
    return {
        "safe": bool(g.get("safe", True)),
        "layers": layers,
        "layers_fired": [name for name, fired in layers.items() if fired],
        "input_max_severity": g.get("input_max_severity", ""),
        "output_max_severity": g.get("output_max_severity", ""),
        "findings": findings,
        "quarantined_chunks": list(g.get("quarantined_chunks") or []),
    }


def shape_response(data: dict[str, Any]) -> dict[str, Any]:
    """Shape a full ChatResponse into the complete view model the render layer consumes.

    One pure function the renderer calls once; everything below ``shape_response`` in the UI is then
    just 'read this dict and draw it', and this single function is what the tests assert against."""
    return {
        "answer": data.get("answer", ""),
        "sources": list(data.get("sources") or []),
        "routing": shape_routing(data),
        "trajectory": shape_trajectory(data),
        "eval": shape_eval(data),
        "guardrail": shape_guardrail(data),
    }


def shape_health(health: dict[str, Any] | None) -> str | None:
    """Render-ready one-line health string, or None when health is unavailable.

    Pure so the (tiny) status-line formatting is tested without a backend."""
    if not health:
        return None
    backend = (health.get("llm") or {}).get("backend") or health.get("llm") or "?"
    chunks = health.get("chunks_indexed", "?")
    ready = health.get("pipeline_ready")
    ready_str = "ready" if ready else "not ready"
    chunks_str = "unreachable" if chunks == -1 else chunks
    return f"backend: {backend} · pipeline {ready_str} · chunks_indexed: {chunks_str}"


# ---------------------------------------------------------------------------
# Streamlit render layer — thin glue, skipped unless run as a Streamlit app.
# ---------------------------------------------------------------------------
# Everything above is import-safe with NO third-party deps beyond the stdlib, so the tests import
# this module and exercise the helpers even when `streamlit`/`requests` aren't installed. The render
# code below imports them lazily inside `main()` so a plain `import rageval.ui` never requires them.


def _render_routing(st, routing: dict[str, Any]) -> None:
    tools = ", ".join(routing["tools_used"]) or "—"
    flag = " · hybrid (chained engines)" if routing["hybrid"] else ""
    st.markdown(f"**Routing** — mode: `{routing['mode']}` · tools: {tools}"
                f" · {routing['tool_calls']} call(s){flag}")
    with st.expander("Routing detail"):
        st.json(routing["raw"])


def _render_trajectory(st, trajectory: list[dict[str, Any]]) -> None:
    if not trajectory:
        return
    st.markdown("**Trajectory** — the agent's tool calls, in order (show your work):")
    for step in trajectory:
        status = "✅" if step["ok"] else "⚠️"
        with st.expander(f"{status} step {step['n']}: `{step['tool']}`"):
            if step["thought"]:
                st.markdown(f"_thought:_ {step['thought']}")
            st.markdown("_args:_")
            st.json(step["args"])
            st.markdown(f"_result:_ {step['result_summary']}")


def _render_eval(st, ev: dict[str, Any]) -> None:
    if ev["overall_pass"]:
        st.success(ev["badge"])
    else:
        st.error(ev["badge"])
    c1, c2 = st.columns(2)
    f, r = ev["faithfulness"], ev["answer_relevance"]
    c1.metric("Faithfulness", f"{f['score']}/5" if f["score"] is not None else "—", f["severity"])
    c2.metric("Answer relevance", f"{r['score']}/5" if r["score"] is not None else "—", r["severity"])
    with st.expander("Why (judge reasoning + findings)"):
        st.markdown(f"**Faithfulness:** {f['reason']}")
        st.markdown(f"**Answer relevance:** {r['reason']}")
        for item in ev["findings"]:
            st.markdown(f"- {item}")


def _render_guardrail(st, g: dict[str, Any]) -> None:
    if g["safe"]:
        st.markdown("**Guardrail** — :green[safe]")
    else:
        st.markdown("**Guardrail** — :red[NOT safe — findings below]")
    with st.expander("Guardrail detail (layers fired + findings)"):
        st.markdown("_layers:_")
        st.json(g["layers"])
        if g["findings"]:
            st.markdown("_findings:_")
            for f in g["findings"]:
                st.json(f)
        else:
            st.markdown("_no findings._")
        if g["quarantined_chunks"]:
            st.markdown(f"_quarantined chunks:_ {', '.join(g['quarantined_chunks'])}")


def _render_turn(st, view: dict[str, Any]) -> None:
    """Render one assistant turn's full view model (answer + every engine signal)."""
    st.markdown(view["answer"] or "_(no answer)_")
    st.markdown("**Sources:** " + (", ".join(view["sources"]) or "—"))
    _render_routing(st, view["routing"])
    _render_trajectory(st, view["trajectory"])
    _render_eval(st, view["eval"])
    _render_guardrail(st, view["guardrail"])


def main() -> None:  # pragma: no cover — Streamlit glue, exercised manually, not in unit tests.
    """The Streamlit entry point. Lazy-imports streamlit/requests so importing this module for its
    pure helpers never requires the ``[ui]`` extra."""
    import requests
    import streamlit as st

    st.set_page_config(page_title="Strata-RAG", page_icon=":books:", layout="centered")
    st.title("Strata-RAG — see your RAG")
    st.caption("A thin UI over the engine's `/chat` API. Ask about the bundled sample corpus; "
               "every answer carries its routing, the agent's trajectory, the eval verdict, and "
               "the guardrail report — all expandable below.")

    # --- Sidebar: configurable backend + health status line. ---
    with st.sidebar:
        st.header("Backend")
        base_url = st.text_input("API base URL", value=api_url_from_env()).rstrip("/")
        health = call_health(requests.get, base_url)
        line = shape_health(health)
        if line:
            st.success(line)
        else:
            st.warning(f"No /health from {base_url} — is `uvicorn rageval.api:app` running?")
        if st.button("Clear conversation"):
            st.session_state.pop("history", None)
            st.session_state.pop("turns", None)
            st.rerun()

    # --- Multi-turn state: the client owns the transcript and replays it each /chat call. ---
    history: list[dict[str, str]] = st.session_state.setdefault("history", [])
    turns: list[dict[str, Any]] = st.session_state.setdefault("turns", [])

    # Replay the conversation so far (user bubbles + the full signal panel per assistant turn).
    for msg in turns:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                _render_turn(st, msg["view"])

    question = st.chat_input("Ask about the sample corpus…")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Routing, retrieving, reasoning, and evaluating…"):
                try:
                    data = call_chat(requests.post, base_url, question, history)
                except APIError as e:
                    st.error(str(e))
                    st.stop()
                view = shape_response(data)
                _render_turn(st, view)

        # Persist BOTH the API-facing transcript (for replay to the server) and the rich turn
        # records (for re-rendering the page). The assistant's API turn is the plain answer text.
        new_history = append_turn(history, "user", question)
        new_history = append_turn(new_history, "assistant", view["answer"])
        st.session_state["history"] = new_history
        st.session_state["turns"] = [
            *turns,
            {"role": "user", "content": question},
            {"role": "assistant", "view": view},
        ]


if __name__ == "__main__":  # pragma: no cover
    main()
