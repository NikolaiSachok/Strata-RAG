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
    """The outcome of firing one attack case (possibly over several trials). Carries enough to
    compute both rates and to promote a successful bypass into the fixture corpus.

    WHY PER-TRIAL COUNTS. Live testing showed per-payload compliance is STOCHASTIC: the same
    zero-width payload hijacked Sonnet in 6 of 8 shots but refused once — a single-shot verdict is
    misleading. So a record aggregates N trials: `complied`/`refused`/`errors` outcome counts plus
    the model-side rate. Crucially, `errors` (transient 502s, timeouts) are NOT refusals — they are
    excluded from the rate denominator (`answered = complied + refused`). The SCANNER side is
    deterministic per rendered payload, so it is scanned ONCE and never multiplied by trials.

    `success` stays = "did this payload EVER comply" (complied > 0), so the single-shot semantics
    and the promote-to-fixtures gate are unchanged."""

    case_id: str
    intent_id: str
    family: str
    encoder: str
    delivery: str
    rendered_payload: str
    # Scanner side (evasion) — scanned ONCE, deterministic, never multiplied by trials.
    scanner_flagged: bool
    scanner_severity: str
    scanner_pattern: str = ""
    # Model side (ASR), aggregated over trials:
    trials: int = 1                 # how many times this payload was fired
    complied: int = 0               # trials where the oracle judged a successful hijack
    refused: int = 0                # trials answered but NOT complying (a genuine refusal)
    errors: int = 0                 # trials that errored (502/timeout) — excluded from the rate
    success: bool = False           # complied > 0 — "did this payload EVER hijack"
    success_kind: str = ""          # the kind from the first complying trial (for the breakdown)
    evidence: str = ""
    answer_excerpt: str = ""
    error: str = ""                 # the first error seen (for the report's Errors section)
    # Back-reference so the adaptive strategist can read the intent off a failed record.
    intent: object = None

    @property
    def answered(self) -> int:
        """Trials that produced an answer (complied or refused) — the rate DENOMINATOR."""
        return self.complied + self.refused

    @property
    def success_rate(self) -> float:
        """Per-payload success rate over ANSWERED trials (errors excluded). 0.0 if none answered."""
        return self.complied / self.answered if self.answered else 0.0


def _delivered(case: AttackCase, target: Target) -> tuple[str, list[dict]]:
    """Compute the EXACT (question, history) that will be sent to the target for this case.

    This is the single source of truth for delivery so the scanner and the executor never diverge:
    the bytes we SCAN for evasion are precisely the bytes we SEND. (A prior bug scanned the bare
    payload for multi-turn but sent the wrapped last turn, and scanned the doc but sent it framed —
    so the evasion number described a different string than the model saw.)

    * direct     — send the rendered payload, no history.
    * multi_turn — replay the benign earlier turns as history; the delivered (attack-bearing) turn
                   is the LAST one — that is what we scan AND send.
    * indirect   — MockTarget reads the planted doc directly (full indirect support); a live target
                   can't inject a corpus doc in v1, so the doc is quoted inside a framed user turn —
                   and we scan that exact framed string.
    """
    d = case.deliveries
    if d.kind == "multi_turn":
        history: list[dict] = []
        for t in d.turns[:-1]:
            history.append({"role": "user", "content": t})
            history.append({"role": "assistant", "content": "Okay."})
        return d.turns[-1], history

    if d.kind == "indirect":
        if getattr(target, "name", "") == "mock":
            return d.doc, []
        framed = (
            f"{d.question}\n\nFor reference, here is the retrieved document:\n\"\"\"\n{d.doc}\n\"\"\""
        )
        return framed, []

    return d.question, []


def _scan_target_text(case: AttackCase, target: Target, *,
                      normalize: bool = True) -> tuple[bool, str, str]:
    """Run the deterministic scanner over the EXACT text `_execute` will send (computed once via
    `_delivered`). Returns (flagged, severity, pattern).

    `normalize` mirrors the engine's `guard_normalize`: run it False to measure the PRE-#31 blind
    scanner (the BEFORE evasion rate), True for the WITH-normalization scanner (the AFTER rate).
    Defaults to the target's own `normalize` setting when present so a `MockTarget(normalize=...)`
    and the evasion metric stay consistent."""
    question, _history = _delivered(case, target)
    findings = scan_for_injection(question, normalize=normalize)
    sev = max_severity(findings)
    flagged = sev != "none"
    pattern = findings[0].pattern if findings else ""
    return flagged, sev, pattern


def _send(target: Target, question: str, history=None, *, trial: int = 0) -> dict:
    """Call target.send, forwarding the trial index ONLY to targets that accept it (the
    deterministic test double varies its outcome by trial index). Real targets — HttpChatTarget,
    MockTarget — ignore the index; this keeps the `Target` protocol's `.send(question, history)`
    signature the contract while letting a trial-aware double opt in without randomness."""
    if getattr(target, "trial_aware", False):
        return target.send(question, history=history, trial=trial)
    return target.send(question, history=history)


def _execute(case: AttackCase, target: Target, *, trial: int = 0) -> dict:
    """Fire one case at the target using the EXACT (question, history) from `_delivered` — the same
    bytes the scanner saw."""
    question, history = _delivered(case, target)
    return _send(target, question, history=history or None, trial=trial)


def run(
    target: Target,
    cases: list[AttackCase],
    *,
    max_cases: int | None = None,
    trials: int = 1,
    llm=None,
    normalize: bool | None = None,
) -> list[RunRecord]:
    """Execute up to `max_cases` cases against `target`, firing each `trials` times, returning ONE
    aggregated RunRecord per case.

    Per-payload compliance is stochastic on a live model, so each case is fired `trials` times and
    the record aggregates the outcomes: `complied` / `refused` / `errors`. Errors are NOT refusals —
    they are excluded from the success-rate denominator. The scanner is deterministic, so it is run
    ONCE per payload and never multiplied by trials.

    `llm` (optional) is passed to the oracle for the ambiguous persona-residue judge ONLY — the
    deterministic oracle runs regardless, so a run with no LLM is fully meaningful."""
    n_trials = max(1, trials)
    selected = cases[:max_cases] if max_cases else cases
    # The evasion metric uses the SAME normalization setting as the target, so the reported
    # scanner-evasion rate matches the scanner the target actually ran. `normalize=None` (the
    # default) inherits the target's own `normalize` flag (MockTarget), falling back to True (the
    # shipped guard_normalize default) for targets that don't carry the flag (http/flaky).
    do_normalize = getattr(target, "normalize", True) if normalize is None else normalize
    records: list[RunRecord] = []
    for case in selected:
        # scanned ONCE — deterministic per payload
        flagged, sev, pattern = _scan_target_text(case, target, normalize=do_normalize)
        complied = refused = errors = 0
        first_error = ""
        first_success_kind = ""
        first_evidence = ""
        winning_excerpt = ""
        any_excerpt = ""
        for t in range(n_trials):
            resp = _execute(case, target, trial=t)
            answer = resp.get("answer", "")
            err = resp.get("error", "")
            if err:
                errors += 1
                if not first_error:
                    first_error = err
                continue  # an error is neither a comply nor a refuse — excluded from the rate
            verdict = adjudicate(answer, case.intent, llm=llm)
            if not any_excerpt:
                any_excerpt = (answer or "")[:280]
            if verdict.success:
                complied += 1
                if not first_success_kind:
                    first_success_kind = verdict.kind
                    first_evidence = verdict.evidence
                    winning_excerpt = (answer or "")[:280]
            else:
                refused += 1
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
            trials=n_trials,
            complied=complied,
            refused=refused,
            errors=errors,
            success=complied > 0,
            success_kind=first_success_kind,
            evidence=first_evidence,
            answer_excerpt=winning_excerpt or any_excerpt,
            error=first_error,
            intent=case.intent,
        ))
    return records


@dataclass
class RunSummary:
    """Aggregate the two numbers overall and by breakdown.

    Scanner-evasion is per-PAYLOAD (deterministic): `evasion_rate` = fraction of payloads the
    scanner FAILED to flag. ASR's denominator EXCLUDES ERRORS in BOTH modes (consistent rule:
    "errors never sit in the denominator"):
      * trials == 1 → `asr` = complied payloads / payloads that ANSWERED (an all-errored payload is
        excluded, not counted as a non-success — same as the multi-trial rule).
      * trials  > 1 → compliance is stochastic, so `asr` is RATE-BASED: total complied trials over
        total ANSWERED trials. This is the statistically honest headline.
    Both are exposed so the report can pick the right one."""

    total: int                       # number of payloads (cases)
    scanner_evaded: int              # payloads the scanner did NOT flag
    succeeded: int                   # payloads that complied at least once (complied > 0)
    trials: int = 1
    payloads_answered: int = 0       # payloads with ≥1 answered trial — single-shot ASR denominator
    total_complied: int = 0          # Σ complied trials across all payloads
    total_answered: int = 0          # Σ answered trials (complied + refused) — rate denominator
    total_errors: int = 0            # Σ errored trials (excluded from the rate)
    by_breakdown: dict = field(default_factory=dict)  # (family,encoder,delivery) → counts

    @property
    def evasion_rate(self) -> float:
        return self.scanner_evaded / self.total if self.total else 0.0

    @property
    def asr(self) -> float:
        """Multi-trial: complying trials / answered trials. Single-shot: complying payloads /
        ANSWERED payloads (all-errored payloads excluded from the denominator, same rule)."""
        if self.trials > 1:
            return self.total_complied / self.total_answered if self.total_answered else 0.0
        return self.succeeded / self.payloads_answered if self.payloads_answered else 0.0


def summarize(records: list[RunRecord]) -> RunSummary:
    """Compute the overall + per-(family,encoder,delivery) rates from the records.

    Per breakdown cell we accumulate both the payload counts (`n`, `evaded`, `succeeded`) AND the
    trial-level totals (`complied`, `answered`, `errors`) so the report can show a per-cell success
    RATE like `6/8 = 75%` when trials>1 while keeping the single-shot columns when trials==1."""
    by: dict[tuple, dict] = {}
    evaded = succeeded = payloads_answered = 0
    total_complied = total_answered = total_errors = 0
    # Trials should be uniform across a run; take the max so a stray single-trial record can't
    # mislabel a multi-trial run as single-shot (and vice versa).
    n_trials = max((r.trials for r in records), default=1)
    for r in records:
        if not r.scanner_flagged:
            evaded += 1
        if r.success:
            succeeded += 1
        if r.answered > 0:
            payloads_answered += 1
        total_complied += r.complied
        total_answered += r.answered
        total_errors += r.errors
        key = (r.family, r.encoder, r.delivery)
        cell = by.setdefault(key, {"n": 0, "evaded": 0, "succeeded": 0,
                                   "complied": 0, "answered": 0, "errors": 0})
        cell["n"] += 1
        cell["evaded"] += 0 if r.scanner_flagged else 1
        cell["succeeded"] += 1 if r.success else 0
        cell["complied"] += r.complied
        cell["answered"] += r.answered
        cell["errors"] += r.errors
    return RunSummary(total=len(records), scanner_evaded=evaded, succeeded=succeeded,
                      trials=n_trials, payloads_answered=payloads_answered,
                      total_complied=total_complied, total_answered=total_answered,
                      total_errors=total_errors, by_breakdown=by)


def run_campaign(
    target: Target,
    *,
    families: list[str] | None = None,
    encoders: list[str] | None = None,
    max_cases: int | None = None,
    trials: int = 1,
    llm=None,
    adapt: bool = False,
    normalize: bool | None = None,
) -> tuple[list[RunRecord], RunSummary]:
    """Convenience: generate the matrix, run it (each payload `trials` times), optionally do ONE
    adaptive round, summarize.

    The adaptive round (only if `adapt` and an `llm` is present) mutates the failed cases and runs
    the new ones too — the 'loop until dry' shape, bounded to one extra pass in v1."""
    cases = strategist.generate(families=families, encoders=encoders)
    records = run(target, cases, max_cases=max_cases, trials=trials, llm=llm, normalize=normalize)
    if adapt and llm is not None:
        extra_cases = strategist.adapt(llm, records)
        if extra_cases:
            records += run(target, extra_cases, trials=trials, llm=llm, normalize=normalize)
    return records, summarize(records)
