"""Router: classify a question, then dispatch it across the engine's two query classes.

THE PROBLEM this solves (the core architectural insight, see sidecar.py): user questions
split into two families, and the WRONG engine fails each one:

  * "which projects have a fruit-like theme?"   → SEMANTIC (embed → top-k). A SQL GROUP BY
                                                   can't match meaning across synonyms.
  * "how many projects per publisher?"          → AGGREGATION (SQL over the sidecar). A vector
                                                   top-k literally cannot count or group.

So before answering we ROUTE. The router returns its decision TRANSPARENTLY (route +
confidence + reasoning) so the choice is auditable, never a black box.

THE CLASSIFIER — two tiers, cheapest first:

  1. A DETERMINISTIC RULE PRE-FILTER short-circuits the obvious aggregation phrasings
     ("how many", "count", "list all", "per <field>", "total/average") with NO LLM call.
     Cheap, instant, and free — most structural questions are caught here.
  2. For everything else, an LLM CLASSIFIER (structured JSON output via the existing llm
     backend) picks the route. The call is isolated behind one method so tests inject a fake
     backend — no live model is needed to unit-test routing.

The four routes:
  * semantic     — answer from the vector index (RagPipeline.answer).
  * aggregation  — answer from the sidecar via a templated intent (aggregate.py).
  * lookup       — a degenerate aggregation: fetch one row's fields by id (also aggregate.py).
  * hybrid       — needs BOTH (e.g. metadata-filter THEN semantic). #4 handles this MINIMALLY
                   (a basic two-step) and labels it; the richer agentic decomposition is #5.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .llm import LLMBackend

VALID_ROUTES = ("semantic", "aggregation", "lookup", "hybrid")


@dataclass
class RouteDecision:
    """The router's transparent verdict. Carried onto the answer so a client can SEE why a
    given engine was chosen (and with what confidence)."""
    route: str               # one of VALID_ROUTES
    confidence: float        # 0.0–1.0
    reasoning: str
    method: str = "llm"      # "rule" (pre-filter) | "llm" (classifier) | "fallback"
    # Optional structured slots when the LLM proposed an aggregation intent (passed to
    # aggregate.execute). None for pure semantic routing.
    slots: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Tier 1 — the deterministic rule pre-filter (no LLM call).
# ---------------------------------------------------------------------------

# Phrases that almost always mean AGGREGATION. Word-boundary matched, case-insensitive.
# Kept deliberately conservative: a false "aggregation" that the sidecar can't serve falls
# back to semantic anyway (see dispatch.py), so the cost of a miss is bounded.
_AGG_PATTERNS = [
    r"\bhow many\b",
    r"\bcount(?:\s+of)?\b",
    r"\bnumber of\b",
    r"\bhow much\b",
    r"\btotal\b",
    r"\baverage\b",
    r"\bmean\b",
    r"\blist (?:all|every|the)\b",
    r"\b(?:per|by|grouped by|group by|for each)\s+\w+",
    r"\btop\s+\d+\b",
    r"\bmost common\b",
    r"\bdistinct\b",
    r"\bhow many .* per\b",
]
_AGG_RE = re.compile("|".join(_AGG_PATTERNS), re.IGNORECASE)


def rule_prefilter(question: str) -> RouteDecision | None:
    """Cheap deterministic short-circuit. Returns an aggregation RouteDecision for an obvious
    structural phrasing, else None (defer to the LLM classifier).

    This is the 'save an LLM call on the easy cases' tier. It only ever ASSERTS aggregation
    (the unambiguous direction); it never asserts semantic — a non-match just means 'not
    obviously aggregation, ask the model'."""
    q = question.strip()
    m = _AGG_RE.search(q)
    if not m:
        return None
    return RouteDecision(
        route="aggregation",
        confidence=0.9,
        reasoning=f"Rule pre-filter matched aggregation phrasing: {m.group(0)!r}.",
        method="rule",
    )


# ---------------------------------------------------------------------------
# Tier 2 — the LLM classifier (structured output).
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = (
    "You are a query router for a retrieval system that has TWO engines:\n"
    "  (A) a SEMANTIC vector index over document text (good at meaning/theme questions), and\n"
    "  (B) a structured SQL sidecar of per-project metadata (good at counting, grouping, "
    "filtering, and exact lookups).\n\n"
    "Classify the user QUESTION into exactly one route and return ONLY a JSON object "
    "(no prose, no code fences) of this shape:\n"
    "{\n"
    '  "route": "semantic|aggregation|lookup|hybrid",\n'
    '  "confidence": <0.0-1.0>,\n'
    '  "reasoning": "<one short sentence>",\n'
    '  "intent": "count|list|group_by_count|top_n|lookup|null",\n'
    '  "field": "<sidecar column or null>",\n'
    '  "filter": {<column>: <value>} or null\n'
    "}\n\n"
    "Route definitions:\n"
    "- semantic: answerable from document MEANING (themes, descriptions, 'which ones are about X').\n"
    "- aggregation: COUNT / LIST / GROUP-BY / TOP-N over the metadata (uses intent+field+filter).\n"
    "- lookup: fetch one specific project's fields by its id (intent='lookup', filter names the id).\n"
    "- hybrid: needs BOTH a metadata filter AND semantic meaning (e.g. 'themes used by publisher X').\n"
    "Set intent/field/filter only for aggregation or lookup; use null otherwise. Keep reasoning short."
)


def _parse_router_json(raw: str) -> dict:
    """Extract the JSON verdict, tolerating fences/prose (same robustness as eval/enrich)."""
    raw = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start, end = raw.find("{"), raw.rfind("}")
        candidate = raw[start : end + 1] if start != -1 and end > start else "{}"
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _decision_from_llm(data: dict) -> RouteDecision:
    """Normalize the model's raw JSON into a validated RouteDecision (defensive defaults so a
    garbled field can never crash dispatch)."""
    route = str(data.get("route", "")).lower().strip()
    if route not in VALID_ROUTES:
        # Unknown/garbled route → safest default is semantic (it can answer anything, even if
        # imperfectly), so the user always gets a response.
        route = "semantic"
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    reasoning = str(data.get("reasoning", ""))[:300] or "LLM classifier decision."

    slots: dict = {}
    if route in ("aggregation", "lookup"):
        intent = data.get("intent")
        intent = None if intent in (None, "null", "") else str(intent)
        slots = {
            "intent": intent,
            "field": (None if data.get("field") in (None, "null", "") else str(data["field"])),
            "filter": data.get("filter") if isinstance(data.get("filter"), dict) else None,
        }
    return RouteDecision(route=route, confidence=confidence, reasoning=reasoning,
                         method="llm", slots=slots)


def llm_classify(question: str, llm: LLMBackend) -> RouteDecision:
    """Ask the LLM for a structured route decision. `llm` is any object with
    `.complete(system, prompt) -> str`; tests pass a fake. On any backend/parse failure we
    return a low-confidence SEMANTIC decision so routing never hard-fails the request."""
    prompt = f"QUESTION:\n{question}\n\nReturn the JSON route verdict now."
    try:
        raw = llm.complete(ROUTER_SYSTEM_PROMPT, prompt, max_tokens=200)
    except Exception as e:  # noqa: BLE001 — never let routing crash the pipeline
        return RouteDecision(route="semantic", confidence=0.3,
                             reasoning=f"LLM classifier unavailable ({e}); defaulting to semantic.",
                             method="fallback")
    return _decision_from_llm(_parse_router_json(raw))


# ---------------------------------------------------------------------------
# The public entry point.
# ---------------------------------------------------------------------------

def route(question: str, llm: LLMBackend | None = None, *, use_rules: bool = True) -> RouteDecision:
    """Classify `question` → RouteDecision.

    Two-tier: the deterministic rule pre-filter runs first (free, instant) and short-circuits
    obvious aggregation phrasings; otherwise the LLM classifier decides. With no `llm` and no
    rule match, we default to SEMANTIC (the universal fallback) at low confidence.

    `use_rules=False` forces the LLM path (used to test the classifier in isolation)."""
    if use_rules:
        pre = rule_prefilter(question)
        if pre is not None:
            return pre
    if llm is None:
        return RouteDecision(route="semantic", confidence=0.3,
                             reasoning="No rule match and no LLM backend; defaulting to semantic.",
                             method="fallback")
    return llm_classify(question, llm)
