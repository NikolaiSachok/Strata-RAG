"""The Runner — orchestrate generate → render → execute → scan → judge → record.

This is where the two numbers are computed. For every attack case the runner:

  1. RENDER   — the case already carries its obfuscated payload (the Strategist did this).
  2. SCAN     — run the engine's REAL `scan_for_injection` on the rendered payload. Did the
                deterministic gate flag it? → the SCANNER-EVASION number (independent of the model).
  3. EXECUTE  — fire at the target (direct / multi-turn replay / indirect doc delivery).
  4. JUDGE    — run the oracle on the FINAL answer. Did the model comply? → the END-TO-END ASR.
  5. RECORD   — one RunRecord per case, tagged by family × encoder × delivery for the breakdown.

The two numbers are deliberately separate: a payload can evade the scanner yet fail to hijack the
model, or slip past both. Keeping them apart is the whole point — it tells you WHICH layer earns
its place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..guardrails import max_severity, scan_for_injection
from . import agent as strategist
from .agent import AttackCase
from .oracle import adjudicate
from .target import Target


@dataclass
class RunRecord:
    """The outcome of firing one attack case. Carries enough to compute both rates and to promote
    a successful bypass into the fixture corpus."""

    case_id: str
    intent_id: str
    family: str
    encoder: str
    delivery: str
    rendered_payload: str
    # Scanner side (evasion):
    scanner_flagged: bool
    scanner_severity: str
    scanner_pattern: str = ""
    # Model side (ASR):
    success: bool = False
    success_kind: str = ""
    evidence: str = ""
    answer_excerpt: str = ""
    error: str = ""
    # Back-reference so the adaptive strategist can read the intent off a failed record.
    intent: object = None


def _scan_target_text(case: AttackCase) -> tuple[bool, str, str]:
    """Run the deterministic scanner over the text the attack actually delivers. For indirect, that
    is the planted document; otherwise the rendered user turn. Returns (flagged, severity, pattern)."""
    d = case.deliveries
    text = d.doc if d.kind == "indirect" else d.rendered_payload
    findings = scan_for_injection(text)
    sev = max_severity(findings)
    flagged = sev != "none"
    pattern = findings[0].pattern if findings else ""
    return flagged, sev, pattern


def _execute(case: AttackCase, target: Target) -> dict:
    """Fire one case at the target per its delivery channel.

    * direct     — one /send with the rendered payload.
    * multi_turn — replay the benign turns as history, send the LAST turn, judge its answer.
    * indirect   — deliver the planted doc. The MockTarget reads the doc directly; for a live HTTP
                   target v1 cannot inject a corpus doc, so we deliver the doc as a QUOTED
                   observation inside the user turn (a documented limitation, flagged on the record).
    """
    d = case.deliveries
    if d.kind == "multi_turn":
        history = []
        for t in d.turns[:-1]:
            history.append({"role": "user", "content": t})
            history.append({"role": "assistant", "content": "Okay."})
        return target.send(d.turns[-1], history=history)

    if d.kind == "indirect":
        if getattr(target, "name", "") == "mock":
            # MockTarget consumes the planted doc text directly (full indirect support).
            return target.send(d.doc)
        # Live/HTTP v1: no corpus injection — quote the doc as a retrieved observation.
        framed = (
            f"{d.question}\n\nFor reference, here is the retrieved document:\n\"\"\"\n{d.doc}\n\"\"\""
        )
        return target.send(framed)

    return target.send(d.question)


def run(
    target: Target,
    cases: list[AttackCase],
    *,
    max_cases: int | None = None,
    llm=None,
) -> list[RunRecord]:
    """Execute up to `max_cases` cases against `target`, returning a RunRecord each.

    `llm` (optional) is passed to the oracle for the ambiguous persona-residue judge ONLY — the
    deterministic oracle runs regardless, so a run with no LLM is fully meaningful."""
    selected = cases[:max_cases] if max_cases else cases
    records: list[RunRecord] = []
    for case in selected:
        flagged, sev, pattern = _scan_target_text(case)
        resp = _execute(case, target)
        answer = resp.get("answer", "")
        err = resp.get("error", "")
        verdict = adjudicate(answer, case.intent, llm=llm)
        records.append(RunRecord(
            case_id=case.id,
            intent_id=case.intent.id,
            family=case.intent.family,
            encoder=case.encoder,
            delivery=case.delivery,
            rendered_payload=case.rendered_payload,
            scanner_flagged=flagged,
            scanner_severity=sev,
            scanner_pattern=pattern,
            success=verdict.success,
            success_kind=verdict.kind,
            evidence=verdict.evidence,
            answer_excerpt=(answer or "")[:280],
            error=err,
            intent=case.intent,
        ))
    return records


@dataclass
class RunSummary:
    """Aggregate the two numbers overall and by breakdown. `evasion_rate` = fraction of cases the
    scanner FAILED to flag; `asr` = fraction where the model COMPLIED."""

    total: int
    scanner_evaded: int
    succeeded: int
    by_breakdown: dict = field(default_factory=dict)  # (family,encoder,delivery) → counts

    @property
    def evasion_rate(self) -> float:
        return self.scanner_evaded / self.total if self.total else 0.0

    @property
    def asr(self) -> float:
        return self.succeeded / self.total if self.total else 0.0


def summarize(records: list[RunRecord]) -> RunSummary:
    """Compute the overall + per-(family,encoder,delivery) rates from the records."""
    by: dict[tuple, dict] = {}
    evaded = succeeded = 0
    for r in records:
        if not r.scanner_flagged:
            evaded += 1
        if r.success:
            succeeded += 1
        key = (r.family, r.encoder, r.delivery)
        cell = by.setdefault(key, {"n": 0, "evaded": 0, "succeeded": 0})
        cell["n"] += 1
        cell["evaded"] += 0 if r.scanner_flagged else 1
        cell["succeeded"] += 1 if r.success else 0
    return RunSummary(total=len(records), scanner_evaded=evaded, succeeded=succeeded,
                      by_breakdown=by)


def run_campaign(
    target: Target,
    *,
    families: list[str] | None = None,
    encoders: list[str] | None = None,
    max_cases: int | None = None,
    llm=None,
    adapt: bool = False,
) -> tuple[list[RunRecord], RunSummary]:
    """Convenience: generate the matrix, run it, optionally do ONE adaptive round, summarize.

    The adaptive round (only if `adapt` and an `llm` is present) mutates the failed cases and runs
    the new ones too — the 'loop until dry' shape, bounded to one extra pass in v1."""
    cases = strategist.generate(families=families, encoders=encoders)
    records = run(target, cases, max_cases=max_cases, llm=llm)
    if adapt and llm is not None:
        extra_cases = strategist.adapt(llm, records)
        if extra_cases:
            records += run(target, extra_cases, llm=llm)
    return records, summarize(records)
