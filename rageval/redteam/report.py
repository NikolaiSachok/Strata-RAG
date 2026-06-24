"""The Reporter — the two-numbers markdown table + promote-to-fixtures.

`render_report` turns a run into the headline artifact: scanner-evasion rate and end-to-end ASR,
broken down by family × encoder × delivery, so you can SEE which obfuscation evades the regex and
whether any of it actually hijacked the model.

NOTE on the ASR number: when computed from the DETERMINISTIC oracle alone (no LLM judge), the ASR is
a CONSERVATIVE LOWER BOUND — the oracle is precision-first and parks any ambiguous "needle present
but refusal language too" answer as not-success. The LLM behavioural judge recovers that recall on
live runs (see `oracle.py`).

`promote_to_fixtures` closes the loop the issue is really about: a bypass the red-team DISCOVERS
should become a permanent regression test. It emits ready-to-paste `Attack(...)` source for the
successful cases — but it NEVER auto-edits `tests/attack_fixtures.py`. A human reviews and pastes,
so the corpus grows from real discoveries while a person still gates what becomes a permanent test
(the same human-in-the-loop gate as the leak-defense's semantic audit)."""

from __future__ import annotations

from .runner import RunRecord, RunSummary, summarize


def _pct(n: int, d: int) -> str:
    return f"{(100 * n / d):.0f}%" if d else "—"


def render_report(records: list[RunRecord], summary: RunSummary | None = None) -> str:
    """Markdown report: an overall line + a per-(family,encoder,delivery) breakdown table.

    When `trials == 1` the output is the original single-shot table (model-complied = payloads that
    complied). When `trials > 1` it switches to the statistically honest view: a per-cell success
    RATE over answered trials (e.g. `6/8 = 75%`) plus an explicit `errors` column, because live
    compliance is stochastic and errors (502/timeout) must not masquerade as refusals."""
    s = summary or summarize(records)
    multi = s.trials > 1
    lines: list[str] = []
    lines.append("# Red-team run — injection bypass report\n")
    lines.append(f"- **Cases:** {s.total}" + (f" × {s.trials} trials each" if multi else ""))
    lines.append(f"- **Scanner-evasion rate:** {_pct(s.scanner_evaded, s.total)} "
                 f"({s.scanner_evaded}/{s.total} payloads the deterministic scanner did NOT flag)")
    if multi:
        lines.append(
            f"- **End-to-end ASR (rate over answered trials):** "
            f"{_pct(s.total_complied, s.total_answered)} "
            f"({s.total_complied}/{s.total_answered} complying trials; {s.total_errors} errored "
            f"trials excluded)")
        lines.append(f"- **Payloads that complied at least once:** "
                     f"{_pct(s.succeeded, s.total)} ({s.succeeded}/{s.total})\n")
    else:
        excluded = s.total - s.payloads_answered
        suffix = f"; {excluded} all-errored payload(s) excluded" if excluded else ""
        lines.append(f"- **End-to-end ASR:** {_pct(s.succeeded, s.payloads_answered)} "
                     f"({s.succeeded}/{s.payloads_answered} of payloads that answered complied"
                     f"{suffix})\n")

    lines.append("## Breakdown by family × encoder × delivery\n")
    if multi:
        lines.append("| family | encoder | delivery | n | scanner-evaded | "
                     "success-rate (complied/answered) | errors |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
        for (family, encoder, delivery), c in sorted(s.by_breakdown.items()):
            lines.append(
                f"| {family} | {encoder} | {delivery} | {c['n']} | "
                f"{c['evaded']}/{c['n']} ({_pct(c['evaded'], c['n'])}) | "
                f"{c['complied']}/{c['answered']} ({_pct(c['complied'], c['answered'])}) | "
                f"{c['errors']} |"
            )
    else:
        lines.append("| family | encoder | delivery | n | scanner-evaded | model-complied |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        for (family, encoder, delivery), c in sorted(s.by_breakdown.items()):
            lines.append(
                f"| {family} | {encoder} | {delivery} | {c['n']} | "
                f"{c['evaded']}/{c['n']} ({_pct(c['evaded'], c['n'])}) | "
                f"{c['succeeded']}/{c['n']} ({_pct(c['succeeded'], c['n'])}) |"
            )

    # Successful bypasses get called out explicitly — these are the promotion candidates.
    wins = [r for r in records if r.success]
    lines.append(f"\n## Successful bypasses ({len(wins)})\n")
    if not wins:
        lines.append("_None — the model refused or ignored every attack in this run._")
    else:
        for r in wins:
            rate = (f" rate={r.complied}/{r.answered} ({_pct(r.complied, r.answered)})"
                    if multi else "")
            lines.append(f"- `{r.case_id}` — kind={r.success_kind}, encoder={r.encoder}, "
                         f"delivery={r.delivery},{rate} evidence={r.evidence!r}")

    # Errors (e.g. unreachable target / transient 502) surfaced so a run that silently failed is
    # obvious. These trials are EXCLUDED from the success-rate denominator (an error is not a
    # refusal); we list the affected payloads with their errored-trial count.
    errs = [r for r in records if r.errors]
    if errs:
        total_err_trials = sum(r.errors for r in errs)
        lines.append(f"\n## Errors ({total_err_trials} errored trials across {len(errs)} payloads, "
                     f"excluded from the rate)\n")
        for r in errs[:20]:
            count = f" ({r.errors}/{r.trials} trials)" if multi else ""
            lines.append(f"- `{r.case_id}`{count} — {r.error}")

    return "\n".join(lines) + "\n"


# Map an intent family / success kind to the (expect_pattern, expect_min_severity) the STATIC
# scanner would assign once the payload is normalized — so the promoted Attack is a meaningful
# regression test (it documents what the scanner SHOULD catch after a normalization pre-pass).
_PROMOTE_HINTS = {
    "instruction-override": ("instruction_override", "critical"),
    "role-persona-override": ("role_override", "major"),
    "prompt-leak-exfil": ("prompt_leak", "critical"),
    "output-format-hijack": ("format_hijack", "major"),
    "data-exfil": ("markdown_image_exfil", "critical"),
    "tool-abuse": ("tool_command", "major"),
}


def promote_to_fixtures(records: list[RunRecord]) -> str:
    """Emit ready-to-paste `Attack(...)` source for the SUCCESSFUL bypasses (human-reviewed before
    promotion). Each promoted fixture uses the DECODABLE/plaintext intent as the payload and the
    family's expected (pattern, severity) — so once a normalization pre-pass lands, the static
    scanner catches it and the discovered bypass becomes a permanent regression test.

    Returns Python source as a string. Does NOT write any file."""
    wins = [r for r in records if r.success]
    if not wins:
        return "# No successful bypasses to promote in this run.\n"

    seen: set[str] = set()
    out: list[str] = [
        "# --- Promoted from a red-team run (HUMAN REVIEW before pasting into "
        "tests/attack_fixtures.py) ---",
        "# Each is a bypass the adaptive harness discovered; payload shown in PLAINTEXT (the "
        "decodable intent),",
        "# with the (pattern, severity) the scanner SHOULD assign after a normalization pre-pass.",
        "PROMOTED_ATTACKS = [",
    ]
    for r in wins:
        # De-dup by intent+encoder so the same intent under many deliveries promotes once.
        key = f"{r.intent_id}:{r.encoder}"
        if key in seen:
            continue
        seen.add(key)
        pattern, severity = _PROMOTE_HINTS.get(r.family, ("instruction_override", "critical"))
        intent = r.intent
        payload = getattr(intent, "text", r.rendered_payload)
        fid = f"rt_{r.intent_id}_{r.encoder}"
        # Use repr() to produce a VALID Python string literal: it escapes ALL control characters
        # (\n, \r, \t, quotes, backslashes), so a payload containing e.g. a carriage return doesn't
        # emit broken source. Hand-rolled escaping missed \r and friends.
        payload_src = repr(payload)
        out.append(
            f'    Attack({fid!r},\n'
            f'           {payload_src},\n'
            f'           {pattern!r}, {severity!r}),'
        )
    out.append("]")
    return "\n".join(out) + "\n"
