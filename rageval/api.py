"""API: wrap the RAG + eval pipeline in a FastAPI service.

WHY FastAPI for an LLM service:
  * Pydantic models give you typed, validated request/response schemas for free —
    the request body is parsed and checked before your code runs, and the response
    shape is documented automatically.
  * It auto-generates interactive API docs at /docs (try it in a browser).
  * It's async-friendly, which matters once you have many concurrent LLM calls.

The endpoint contract:
  GET  /health  → backend + index status (no LLM call)
  POST /ask     → {question} → {answer, sources, eval}  (single-shot router; issue #4)
  POST /chat    → {question, history?} → {answer, sources, trajectory, routing, eval,
                  guardrail}  (multi-turn agent over the router; issue #5)

The whole RAG + eval flow is assembled once at startup (loading the embedding model
and opening the vector store is expensive) and reused across requests via FastAPI's
lifespan handler.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .agent import ChatAgent, Turn
from .config import Settings
from .dispatch import dispatch
from .eval import EvalResult, Judge
from .generate import RagPipeline
from .llm import LLMError, backend_status
from .retrieve import CollectionMissingError

# ---------------------------------------------------------------------------
# Pydantic schemas — the typed contract clients can rely on.
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """The POST /ask body. FastAPI validates this before our handler runs, so a
    missing/empty question yields a clean 422 automatically."""
    question: str = Field(..., min_length=1, description="Natural-language question about the docs.")


class DimensionModel(BaseModel):
    score: int
    severity: str
    reason: str


class EvalModel(BaseModel):
    faithfulness: DimensionModel
    answer_relevance: DimensionModel
    overall_pass: bool
    findings: list[str]


class FindingModel(BaseModel):
    pattern: str
    severity: str
    snippet: str
    where: str


class GuardrailModel(BaseModel):
    """The prompt-injection guardrail report. Surfaced so a client can SEE the defenses
    ran and decide whether to trust the answer (via `safe`)."""
    sentinel_used: bool
    safe: bool
    input_max_severity: str
    output_max_severity: str
    input_findings: list[FindingModel]
    output_findings: list[FindingModel]
    quarantined_chunks: list[str]
    layers: dict[str, bool]


class RoutingModel(BaseModel):
    """The query-router's transparent decision (see router.py / dispatch.py). Surfaced so a
    client can SEE which engine answered (semantic vs the sidecar), how confident the router
    was, and — for an aggregation — the EXACT templated query + bound params that ran.

    Extra keys (executed_query, params, intent, row_count, fell_back, hybrid_source_set, note)
    appear only on the paths that produce them, so the model allows them rather than 5 optional
    fields the semantic path would always leave null."""
    model_config = {"extra": "allow"}

    route: str
    confidence: float
    reasoning: str
    method: str
    executed_route: str
    fell_back: bool = False


class AskResponse(BaseModel):
    """What POST /ask returns: the grounded answer, its cited sources, the judge's verdict,
    the guardrail report, AND the routing decision — RAG with a quality gate, an injection-defense
    audit, AND a transparent record of which engine served the question."""
    question: str
    answer: str
    sources: list[str]
    eval: EvalModel
    guardrail: GuardrailModel
    routing: RoutingModel | None = None


# ---- /chat (issue #5): the multi-turn agent over the router ----------------

class ChatMessage(BaseModel):
    """One prior conversation turn the client passes back so follow-ups resolve in context."""
    role: str = Field(..., pattern="^(user|assistant)$", description="'user' or 'assistant'.")
    content: str = Field(..., min_length=1)


# Cost/abuse bounds on the replayed transcript (#10). The server is stateless and the client
# replays history every turn, so an unbounded `history` is an easy way to blow the LLM context
# (and the bill). Enforced as clean 422 validation, not a silent truncation.
MAX_HISTORY_ITEMS = 50
MAX_HISTORY_CHARS = 50_000


class ChatRequest(BaseModel):
    """The POST /chat body: the new user question plus the running conversation history.

    `history` is optional (an empty/absent list starts a fresh conversation); the client owns
    the transcript and replays it each turn (stateless server — horizontally scalable). It is
    BOUNDED (item count + total chars) so a replayed transcript can't blow context/cost."""
    question: str = Field(..., min_length=1, description="The new user turn.")
    history: list[ChatMessage] = Field(default_factory=list,
                                       description="Prior turns, oldest first.")

    @field_validator("history")
    @classmethod
    def _bound_history(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if len(v) > MAX_HISTORY_ITEMS:
            raise ValueError(
                f"history too long: {len(v)} turns exceeds the {MAX_HISTORY_ITEMS}-turn cap.")
        total = sum(len(m.content) for m in v)
        if total > MAX_HISTORY_CHARS:
            raise ValueError(
                f"history too large: {total} chars exceeds the {MAX_HISTORY_CHARS}-char cap.")
        return v


class TrajectoryStep(BaseModel):
    """One step the agent took — the TRANSPARENCY record of how the answer was derived.

    `extra=allow` keeps it forward-compatible if a tool adds a field; the core shape is fixed."""
    model_config = {"extra": "allow"}

    tool: str
    args: dict
    result_summary: str
    thought: str = ""
    ok: bool = True


class ChatResponse(BaseModel):
    """What POST /chat returns: the composed grounded answer, its sources, the eval verdict, the
    merged guardrail report, AND the agent's trajectory (the ordered tool calls) so the client
    can see exactly how the answer was derived. `routing` summarises the agent's plan (the tools
    it used) — the multi-step analogue of /ask's single routing block."""
    question: str
    answer: str
    sources: list[str]
    trajectory: list[TrajectoryStep]
    routing: dict
    eval: EvalModel
    guardrail: GuardrailModel


# ---------------------------------------------------------------------------
# App + lifespan (build the heavy objects once).
# ---------------------------------------------------------------------------

# A tiny holder for the pipeline/judge so the request handlers can reach them.
class _State:
    pipeline: RagPipeline | None = None
    judge: Judge | None = None
    agent: ChatAgent | None = None


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs at startup/shutdown. We build the pipeline + judge here so the embedding
    model and vector store load ONCE, not per request. If the LLM backend isn't
    configured we still start (so /health works and explains the problem) but defer
    the error to /ask."""
    settings = Settings.load()
    try:
        state.pipeline = RagPipeline(settings)
        # Reuse the pipeline's LLM backend for the judge + agent to avoid a second init.
        state.judge = Judge(settings, llm=state.pipeline.llm)
        state.agent = ChatAgent(state.pipeline)
    except LLMError:
        # No backend yet — leave them None; /ask and /chat return a clear 503.
        state.pipeline = None
        state.judge = None
        state.agent = None
    except CollectionMissingError as e:
        # The target collection doesn't exist (corpus/collection mismatch — e.g. ingested the real
        # corpus but started without RAGENGINE_CORPUS_ROOT, so the name fell back to `..._sample`).
        # We CAN'T serve without an index, so abort startup — but show the clear, actionable reason
        # on its own (the message from the preflight), NOT a 50-line Qdrant traceback.
        raise RuntimeError(f"Cannot start rag-eval-demo: {e}") from None
    yield
    # (nothing to tear down — Chroma persists itself)


app = FastAPI(
    title="rag-eval-demo",
    summary="A tiny RAG service with an LLM-as-judge eval gate.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Liveness + configuration check. Makes NO LLM call, so it's cheap and safe to
    poll. Tells you which backend is active and how many chunks are in the index."""
    indexed = 0
    collection = ""
    try:
        from .index import count, get_client

        s = Settings.load()
        collection = s.collection_name
        indexed = count(get_client(s), s)
    except Exception:  # noqa: BLE001 — Qdrant may be down; health must still answer
        indexed = -1
    return {
        "status": "ok",
        "collection": collection,
        "chunks_indexed": indexed,  # -1 = Qdrant unreachable
        "llm": backend_status(),
        "pipeline_ready": state.pipeline is not None,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """The core endpoint: retrieve → generate → evaluate, in one call.

    Returns the answer with its citations AND the eval verdict, so a client can show
    the answer and decide (via overall_pass) whether to trust/surface it."""
    if state.pipeline is None or state.judge is None:
        # The backend wasn't available at startup. Re-resolve to produce the precise
        # reason (e.g. "set ANTHROPIC_API_KEY or install the claude CLI").
        try:
            settings = Settings.load()
            state.pipeline = RagPipeline(settings)
            state.judge = Judge(settings, llm=state.pipeline.llm)
        except LLMError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

    try:
        # ROUTE → run the chosen engine (semantic / aggregation / lookup / hybrid). The router
        # reuses the pipeline's own LLM backend for classification; the answer carries a
        # transparent `routing` block. Semantic-routed questions behave exactly as before.
        answer = dispatch(req.question, state.pipeline)
        verdict: EvalResult = state.judge.evaluate(answer)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"LLM backend error: {e}") from e

    # to_dict() emits the exact, tested schema; Pydantic re-validates it on the way out.
    return AskResponse(
        question=answer.question,
        answer=answer.answer,
        sources=answer.sources,
        eval=verdict.to_dict(),  # type: ignore[arg-type]  (validated by EvalModel)
        guardrail=answer.guardrail.to_dict(),  # type: ignore[arg-type]  (validated by GuardrailModel)
        routing=answer.routing,  # type: ignore[arg-type]  (validated by RoutingModel; None ok)
    )


def _agent_routing_summary(result) -> dict:
    """Summarise the agent's PLAN for the routing block — the multi-step analogue of /ask's
    single routing decision. Records which tools ran and how many hops, so a client gets the
    same 'how was this served' transparency at the conversation level."""
    tools_used = [t.tool for t in result.trajectory]
    return {
        "mode": "agent",
        "tool_calls": len(result.trajectory),
        "tools_used": tools_used,
        # 'hybrid' iff the agent chained BOTH engines in one turn (the multi-hop decomposition).
        "hybrid": ("semantic_search" in tools_used and "query_metadata" in tools_used),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """The MULTI-TURN agentic endpoint (issue #5): the agent decomposes the question into one
    or more tool calls (semantic search and/or the metadata sidecar), chains them as needed,
    and composes ONE grounded, cited answer — carrying prior `history` so follow-ups resolve in
    context. Returns the answer, sources, the eval verdict, the merged guardrail report, AND the
    agent's trajectory (the ordered tool calls) so the derivation is fully auditable."""
    if state.agent is None or state.judge is None:
        # Backend wasn't available at startup — re-resolve to produce the precise reason.
        try:
            settings = Settings.load()
            state.pipeline = RagPipeline(settings)
            state.judge = Judge(settings, llm=state.pipeline.llm)
            state.agent = ChatAgent(state.pipeline)
        except LLMError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

    try:
        history = [Turn(role=m.role, content=m.content) for m in req.history]
        result = state.agent.chat(req.question, history=history)
        # The agent hands us an Answer (eval_answer) whose context is the gathered evidence, so
        # the EXISTING judge grades the multi-step answer unchanged.
        verdict: EvalResult = state.judge.evaluate(result.eval_answer)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"LLM backend error: {e}") from e

    return ChatResponse(
        question=req.question,
        answer=result.answer,
        sources=result.sources,
        trajectory=[t.to_dict() for t in result.trajectory],  # type: ignore[arg-type]
        routing=_agent_routing_summary(result),
        eval=verdict.to_dict(),  # type: ignore[arg-type]  (validated by EvalModel)
        guardrail=result.guardrail.to_dict(),  # type: ignore[arg-type]  (validated by GuardrailModel)
    )
