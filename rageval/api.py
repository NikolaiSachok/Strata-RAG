"""API: wrap the RAG + eval pipeline in a FastAPI service.

WHY FastAPI for an LLM service:
  * Pydantic models give you typed, validated request/response schemas for free —
    the request body is parsed and checked before your code runs, and the response
    shape is documented automatically.
  * It auto-generates interactive API docs at /docs (try it in a browser).
  * It's async-friendly, which matters once you have many concurrent LLM calls.

The endpoint contract:
  GET  /health  → backend + index status (no LLM call)
  POST /ask     → {question} → {answer, sources, eval}

The whole RAG + eval flow is assembled once at startup (loading the embedding model
and opening the vector store is expensive) and reused across requests via FastAPI's
lifespan handler.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings
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


class AskResponse(BaseModel):
    """What POST /ask returns: the grounded answer, its cited sources, the judge's verdict,
    and the guardrail report — RAG with both a quality gate AND an injection-defense audit."""
    question: str
    answer: str
    sources: list[str]
    eval: EvalModel
    guardrail: GuardrailModel


# ---------------------------------------------------------------------------
# App + lifespan (build the heavy objects once).
# ---------------------------------------------------------------------------

# A tiny holder for the pipeline/judge so the request handlers can reach them.
class _State:
    pipeline: RagPipeline | None = None
    judge: Judge | None = None


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
        # Reuse the pipeline's LLM backend for the judge to avoid a second init.
        state.judge = Judge(settings, llm=state.pipeline.llm)
    except LLMError:
        # No backend yet — leave them None; /ask will return a clear 503.
        state.pipeline = None
        state.judge = None
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
        answer = state.pipeline.answer(req.question)
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
    )
