"""rageval.redteam — an adaptive prompt-injection red-team harness for the engine's own chat surface.

Authorized DEFENSIVE security tooling: it attacks ONLY the local engine to DISCOVER injection
bypasses the static fixture corpus can't, MEASURE them (scanner-evasion rate + end-to-end ASR), and
GROW the regression corpus from what it finds. See docs/red-team.md for the design + the "two
numbers" framing, and GitHub issue #30 for the spec.

The package is a small team of roles (the brief's architecture):
  encoders  — deterministic obfuscation transforms (+ inverse / normalize)   [the portfolio piece]
  payloads  — domain-neutral malicious intents, each with a canary observable
  scenarios — multi-turn split sequences + indirect (poisoned-retrieval) doc templates
  oracle    — deterministic canary detection first, optional LLM behavioural judge on the residue
  target    — Target protocol: HttpChatTarget (live), InProcessTarget (v1 stub), MockTarget (CI)
  agent     — the Strategist: deterministic catalog builder (+ an optional LLM adapt() hook)
  runner    — generate → scan → execute → judge → record; computes the two numbers by breakdown
  report    — markdown table + promote_to_fixtures() (human-reviewed before promotion)

Everything runs with NO LLM (the MockTarget + deterministic oracle make the suite fast/offline).
"""

from __future__ import annotations

from . import agent, encoders, oracle, payloads, report, runner, scenarios, target
from .agent import AttackCase, generate
from .oracle import Verdict, adjudicate, judge
from .payloads import BASE_INTENTS, Intent
from .report import promote_to_fixtures, render_report
from .runner import RunRecord, RunSummary, run, run_campaign, summarize
from .target import HttpChatTarget, InProcessTarget, MockTarget, Target

__all__ = [
    "encoders", "payloads", "scenarios", "oracle", "target", "agent", "runner", "report",
    "Intent", "BASE_INTENTS",
    "AttackCase", "generate",
    "Verdict", "judge", "adjudicate",
    "Target", "HttpChatTarget", "InProcessTarget", "MockTarget",
    "RunRecord", "RunSummary", "run", "summarize", "run_campaign",
    "render_report", "promote_to_fixtures",
]
