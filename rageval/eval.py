"""Eval: LLM-AS-JUDGE — score each answer's faithfulness and relevance.

This is the stage most RAG tutorials skip, and it's what separates a demo from
something you'd trust in production.

THE PROBLEM: a RAG system can retrieve the right context and *still* produce a bad
answer — it can hallucinate a detail not in the context, or answer a different
question than the one asked. You can't catch that by checking the pipeline ran; you
have to judge the *content* of the answer.

THE TECHNIQUE — LLM-as-judge: make a SECOND, independent LLM call whose only job is
to grade the first one against a strict rubric. The judge sees three things — the
question, the exact context the generator was given, and the answer — and returns a
structured verdict. Because it's structured (scores + severities), you can:
  * gate a single response (block answers that fail), and
  * aggregate over a question set into pass-rates → a real eval harness / regression test.

TWO DIMENSIONS we score (the classic RAG eval pair):
  * faithfulness / groundedness — is every claim in the answer supported by the
    context? (catches hallucination)
  * answer_relevance — does the answer actually address the question? (catches
    "technically true but off-topic" answers)

SEVERITY SCALE: borrowed from code/content review rubric practice — none / minor /
major / critical. Scores tell you "how good"; severity tells you "how much it matters",
which is what you actually gate on. The default gate fails the response if either
dimension is `major` or worse.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .config import SETTINGS, Settings
from .generate import Answer
from .llm import LLMBackend, get_llm

# Severities ordered from harmless to blocking. The gate compares against this order.
SEVERITY_ORDER = ["none", "minor", "major", "critical"]
DEFAULT_GATE_SEVERITY = "major"  # response fails if any dimension is this severe or worse

# The judge's instructions. We demand strict JSON so the result is machine-checkable.
# Note we ask the judge to reason in the "reason" fields but still emit only JSON —
# a small, robust contract that's easy to parse and test.
JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluation judge for a retrieval-augmented question-answering "
    "system. You will be given a QUESTION, the CONTEXT passages that were retrieved, "
    "and the ANSWER that was generated from them. Grade the answer on two dimensions and "
    "return ONLY a JSON object (no prose, no code fences) with this exact shape:\n"
    "{\n"
    '  "faithfulness": {"score": <1-5>, "severity": "none|minor|major|critical", "reason": "<short>"},\n'
    '  "answer_relevance": {"score": <1-5>, "severity": "none|minor|major|critical", "reason": "<short>"},\n'
    '  "findings": ["<short notes about any problems>"]\n'
    "}\n\n"
    "Definitions:\n"
    "- faithfulness: is EVERY claim in the answer supported by the CONTEXT? A claim not "
    "in the context is unfaithful (a hallucination). If the answer correctly says it "
    "lacks information AND the context indeed lacks it, that is fully faithful (score 5).\n"
    "- answer_relevance: does the answer address the QUESTION that was actually asked?\n"
    "Scoring: 5 = perfect, 4 = minor issue, 3 = noticeable issue, 2 = serious issue, "
    "1 = fails. Set severity to reflect impact: none (5), minor (4), major (2-3), "
    "critical (1). Keep reasons to one sentence."
)


@dataclass
class Dimension:
    """One graded dimension of the answer."""
    score: int                 # 1-5
    severity: str              # one of SEVERITY_ORDER
    reason: str

    def normalized(self) -> "Dimension":
        """Clamp/repair model output so downstream code can rely on the schema even if
        the judge returns something slightly off (e.g. score 7, or an unknown severity)."""
        score = max(1, min(5, int(self.score)))
        severity = self.severity if self.severity in SEVERITY_ORDER else _severity_from_score(score)
        return Dimension(score=score, severity=severity, reason=str(self.reason)[:400])


@dataclass
class EvalResult:
    """The complete verdict for one answer. `overall_pass` is the gate decision."""
    faithfulness: Dimension
    answer_relevance: Dimension
    overall_pass: bool
    findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Stable JSON schema — this is exactly what the API returns and what
        tests/test_eval_schema.py asserts against."""
        return {
            "faithfulness": vars(self.faithfulness),
            "answer_relevance": vars(self.answer_relevance),
            "overall_pass": self.overall_pass,
            "findings": list(self.findings),
        }


def _severity_from_score(score: int) -> str:
    """Map a 1-5 score to a default severity when the judge omits/garbles it."""
    return {5: "none", 4: "minor", 3: "major", 2: "major", 1: "critical"}.get(score, "major")


def _severity_at_least(severity: str, threshold: str) -> bool:
    """True if `severity` is as bad as or worse than `threshold`."""
    return SEVERITY_ORDER.index(severity) >= SEVERITY_ORDER.index(threshold)


def compute_gate(faithfulness: Dimension, relevance: Dimension,
                 threshold: str = DEFAULT_GATE_SEVERITY) -> bool:
    """Pure gate logic, separated so it can be tested without any LLM call.

    Returns True (pass) only if NEITHER dimension reaches the failing severity."""
    return not (
        _severity_at_least(faithfulness.severity, threshold)
        or _severity_at_least(relevance.severity, threshold)
    )


def _parse_judge_json(raw: str) -> dict:
    """Extract the JSON verdict from the judge's reply, tolerating code fences or
    stray prose around the object (LLMs sometimes add both despite instructions)."""
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start, depth = raw.find("{"), 0
        if start != -1:
            for i in range(start, len(raw)):
                if raw[i] == "{":
                    depth += 1
                elif raw[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = raw[start : i + 1]
                        break
    if candidate is None:
        raise ValueError(f"no JSON object found in judge output: {raw[:200]}")
    return json.loads(candidate)


def _dimension_from(raw: dict) -> Dimension:
    """Build a Dimension from a raw sub-dict, with safe fallbacks."""
    return Dimension(
        score=int(raw.get("score", 3)),
        severity=str(raw.get("severity", "")).lower(),
        reason=str(raw.get("reason", "")),
    ).normalized()


def parse_eval(raw: str, threshold: str = DEFAULT_GATE_SEVERITY) -> EvalResult:
    """Turn the judge's raw text into a validated EvalResult. Separated from the LLM
    call so the parsing/gating logic is unit-testable with canned strings."""
    data = _parse_judge_json(raw)
    faithfulness = _dimension_from(data.get("faithfulness", {}))
    relevance = _dimension_from(data.get("answer_relevance", {}))
    findings = data.get("findings") or []
    if not isinstance(findings, list):
        findings = [str(findings)]
    return EvalResult(
        faithfulness=faithfulness,
        answer_relevance=relevance,
        overall_pass=compute_gate(faithfulness, relevance, threshold),
        findings=[str(f) for f in findings],
    )


class Judge:
    """Wraps the LLM backend to evaluate answers. Reuses the same backend as the
    generator by default — in a stricter setup you might use a *different* (stronger)
    model as judge, which this design allows by passing a separate llm."""

    def __init__(self, settings: Settings = SETTINGS, llm: LLMBackend | None = None):
        self.settings = settings
        self.llm = llm if llm is not None else get_llm(settings)

    def evaluate(self, answer: Answer, threshold: str = DEFAULT_GATE_SEVERITY) -> EvalResult:
        prompt = (
            f"QUESTION:\n{answer.question}\n\n"
            f"CONTEXT:\n{answer.context_text()}\n\n"
            f"ANSWER:\n{answer.answer}\n\n"
            "Return the JSON verdict now."
        )
        raw = self.llm.complete(JUDGE_SYSTEM_PROMPT, prompt, max_tokens=600)
        return parse_eval(raw, threshold)


# ===========================================================================
# RETRIEVAL EVAL HARNESS — Recall@K / Precision@K / MRR / nDCG over a golden set.
# ===========================================================================
#
# The LLM-judge above grades GENERATION (is the answer faithful?). This second harness
# grades RETRIEVAL (did we fetch the right projects?), which is the upstream thing that
# caps everything. It needs NO LLM — just the retriever + the hand-labelled golden set —
# so it runs cheaply and deterministically as a regression gate.

from pathlib import Path  # noqa: E402

import yaml  # noqa: E402  (kept here so the judge section above stays import-light)

from .config import GOLDEN_PATH  # noqa: E402
from . import metrics  # noqa: E402


def load_golden(path=GOLDEN_PATH, *, kind: str = "all") -> list[dict]:
    """Load a golden YAML → list of {id, question, relevant:set, kind}.

    Each question MAY carry a `kind:` field; a MISSING kind defaults to "retrieval" (the
    original sample set has no kinds, so it's all retrieval). `kind` arg filters:
      - "retrieval" → only theme/semantic questions (the EMBEDDING A/B ground truth).
      - "metadata"  → only publisher/structural questions (answered by the sidecar, NOT
                      semantic retrieval — the publisher lives in the roster, not the docs).
      - "all"       → everything.
    Keeping metadata questions OUT of the A/B is the point: scoring a publisher lookup against a
    vector index measures the wrong thing — that's a SQL query, not an embedding test."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = []
    for q in data.get("questions", []):
        q_kind = (q.get("kind") or "retrieval").lower()
        if kind != "all" and q_kind != kind:
            continue
        out.append({
            "id": q["id"],
            "question": q["question"],
            "relevant": set(q.get("relevant", [])),
            "kind": q_kind,
        })
    return out


def evaluate_retrieval(settings: Settings = SETTINGS, *, k: int | None = None,
                       golden_path=GOLDEN_PATH, kind: str = "retrieval",
                       dense_only: bool = False) -> dict:
    """Run the retriever over the (kind-filtered) golden questions and compute the metrics.

    Returns a report dict incl. the model/collection/mode so two A/B runs are self-labelling.
    The retriever import is local so importing this module needs no Qdrant.

    For the embedding A/B you want `kind="retrieval"` (the default): metadata/publisher questions
    are answered by the sidecar, not the embedder, so including them would pollute the signal.
    `dense_only=True` turns off BM25 + rerank to isolate the embedder's raw contribution.
    """
    from .retrieve import Retriever

    k = k if k is not None else settings.top_k
    golden = load_golden(golden_path, kind=kind)
    retriever = Retriever(settings, dense_only=dense_only)

    per_q = []
    for item in golden:
        hits = retriever.retrieve(item["question"], top_k=k)
        # Map retrieved chunks → their project ids, preserving rank order.
        ranked_projects = [f"{h.source_set}/{h.project_id}" for h in hits]
        rel = item["relevant"]
        per_q.append({
            "id": item["id"],
            "recall": metrics.recall_at_k(ranked_projects, rel, k),
            "precision": metrics.precision_at_k(ranked_projects, rel, k),
            "mrr": metrics.reciprocal_rank(ranked_projects, rel),
            "ndcg": metrics.ndcg_at_k(ranked_projects, rel, k),
        })

    metric_only = [{kk: q[kk] for kk in ("recall", "precision", "mrr", "ndcg")} for q in per_q]
    return {
        "per_question": per_q,
        "aggregate": metrics.aggregate(metric_only),
        "k": k,
        "n_questions": len(per_q),
        "kind": kind,
        "model": settings.embed_model,
        "collection": settings.collection_name,
        "mode": "dense-only" if dense_only else "hybrid+rerank",
    }


def render_metric_table(report: dict) -> str:
    """Pretty-print the per-question + aggregate retrieval metrics, TAGGED with the run's
    model / collection / mode so two A/B runs are trivially comparable side by side."""
    k = report["k"]
    lines = [
        "RETRIEVAL METRICS",
        f"  model      : {report.get('model', '?')}",
        f"  collection : {report.get('collection', '?')}",
        f"  mode       : {report.get('mode', '?')}   kind={report.get('kind', '?')}   "
        f"K={k}   n={report.get('n_questions', len(report['per_question']))}",
        "-" * 64,
    ]
    lines.append(f"{'question':<16}{'Recall@K':>10}{'Prec@K':>10}{'MRR':>8}{'nDCG':>8}")
    for q in report["per_question"]:
        lines.append(f"{q['id']:<16}{q['recall']:>10.3f}{q['precision']:>10.3f}"
                     f"{q['mrr']:>8.3f}{q['ndcg']:>8.3f}")
    agg = report["aggregate"]
    lines.append("-" * 64)
    lines.append(f"{'MEAN':<16}{agg.get('recall', 0):>10.3f}{agg.get('precision', 0):>10.3f}"
                 f"{agg.get('mrr', 0):>8.3f}{agg.get('ndcg', 0):>8.3f}")
    return "\n".join(lines)


# ===========================================================================
# INJECTION-DEFENSE EVAL — treat prompt-injection like a retrieval metric.
# ===========================================================================
#
# The point of measuring (not just asserting) defenses: you can toggle a guard_* layer off
# and watch this number move, which proves each layer earns its place. The "attack success
# rate" is the fraction of known attacks the INPUT scanner failed to flag — lower is better
# (0.0 = every attack detected). It's deterministic (no LLM), so it runs anywhere.

def evaluate_injection_defense() -> dict:
    """Run the input scanner over the adversarial fixture set + the in-corpus injected
    docs; report detection rate, attack-success-rate, and false positives on clean text."""
    from . import guardrails as g
    # The fixtures live under tests/ but are pure data; import defensively so a packaged
    # install without the tests dir still runs the rest of eval.
    try:
        from tests.attack_fixtures import CLEAN_SAMPLES, INPUT_ATTACKS
    except Exception:  # noqa: BLE001
        return {"available": False}

    detected = 0
    rows = []
    for atk in INPUT_ATTACKS:
        findings = g.scan_for_injection(atk.payload)
        patterns = {f.pattern for f in findings}
        hit = bool(findings)
        if hit:
            detected += 1
        rows.append({"id": atk.id, "detected": hit,
                     "expected_pattern_present": atk.expect_pattern in patterns,
                     "max_severity": g.max_severity(findings)})

    false_positives = sum(1 for s in CLEAN_SAMPLES if g.scan_for_injection(s))

    total = len(INPUT_ATTACKS)
    return {
        "available": True,
        "rows": rows,
        "total_attacks": total,
        "detected": detected,
        "attack_success_rate": (total - detected) / total if total else 0.0,
        "false_positives": false_positives,
        "clean_total": len(CLEAN_SAMPLES),
    }


def render_injection_table(report: dict) -> str:
    if not report.get("available"):
        return "INJECTION DEFENSE: attack fixtures not importable (run from repo root)."
    lines = ["INJECTION-DEFENSE EVAL (input scanner over the adversarial fixture set)", "-" * 64]
    lines.append(f"{'attack':<24}{'detected':>10}{'expected-hit':>14}{'severity':>10}")
    for r in report["rows"]:
        lines.append(f"{r['id']:<24}{str(r['detected']):>10}"
                     f"{str(r['expected_pattern_present']):>14}{r['max_severity']:>10}")
    lines.append("-" * 64)
    lines.append(f"detected {report['detected']}/{report['total_attacks']}  |  "
                 f"ATTACK-SUCCESS-RATE = {report['attack_success_rate']:.1%}  |  "
                 f"false positives on clean text: {report['false_positives']}/{report['clean_total']}")
    return "\n".join(lines)


def main() -> None:
    """`python -m rageval.eval` — print the retrieval metric table over a golden set.

    A/B flags (compare two embedding models / two retrieval modes):
      --golden <path>      golden YAML to use (default eval/golden.yaml).
      --collection <name>  eval against a specific Qdrant collection (else model-derived).
      --kind retrieval|metadata|all  which golden questions to score (default retrieval —
                           the embedding A/B must use ONLY theme/semantic questions; publisher/
                           structural ones are metadata, answered by the sidecar).
      --dense-only         turn off BM25 + rerank to isolate the embedder's raw contribution.
      --k <int>            K for Recall@K/Precision@K/nDCG.
    Other:
      --faithfulness       also run the LLM-as-judge faithfulness check (needs an LLM backend).
      --injection          also run the deterministic injection-defense eval (no LLM needed).
    """
    import argparse
    import dataclasses

    parser = argparse.ArgumentParser(description="Retrieval (+ A/B) + faithfulness + injection eval.")
    parser.add_argument("--k", type=int, default=None, help="K for Recall@K/Precision@K/nDCG.")
    parser.add_argument("--golden", default=None, help="Path to a golden YAML (default eval/golden.yaml).")
    parser.add_argument("--collection", default=None,
                        help="Eval against this Qdrant collection (overrides the model-derived name).")
    parser.add_argument("--kind", choices=["retrieval", "metadata", "all"], default="retrieval",
                        help="Filter golden questions by kind (default retrieval — the embedding A/B set).")
    parser.add_argument("--dense-only", action="store_true",
                        help="Disable BM25 + cross-encoder rerank (isolate the embedding model).")
    parser.add_argument("--faithfulness", action="store_true",
                        help="Also run the LLM-as-judge faithfulness check (needs an LLM backend).")
    parser.add_argument("--injection", action="store_true",
                        help="Also run the deterministic prompt-injection-defense eval.")
    args = parser.parse_args()

    settings = Settings.load()
    if args.collection:
        # Pin the exact collection (e.g. to eval a specific A/B index) without changing model.
        settings = dataclasses.replace(settings, collection_override=args.collection)
    golden_path = Path(args.golden) if args.golden else GOLDEN_PATH

    report = evaluate_retrieval(settings, k=args.k, golden_path=golden_path,
                                kind=args.kind, dense_only=args.dense_only)
    print(render_metric_table(report))

    if args.injection:
        print()
        print(render_injection_table(evaluate_injection_defense()))

    if args.faithfulness:
        from .generate import RagPipeline

        print("\nFAITHFULNESS (LLM-as-judge) per question:")
        pipeline = RagPipeline(settings)
        judge = Judge(settings, llm=pipeline.llm)
        for item in load_golden(golden_path, kind=args.kind):
            ans = pipeline.answer(item["question"])
            verdict = judge.evaluate(ans)
            print(f"  {item['id']:<16} pass={verdict.overall_pass} "
                  f"faithfulness={verdict.faithfulness.score}/5 "
                  f"relevance={verdict.answer_relevance.score}/5")


if __name__ == "__main__":
    main()
