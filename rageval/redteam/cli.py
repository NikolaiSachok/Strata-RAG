"""`python -m rageval.redteam` — run a bounded red-team campaign and print/write the report.

Defaults to the offline MockTarget so it runs in CI with no model or network. Point it at a live
engine with `--target http --base-url http://localhost:8000` for the honest test (slow: each /chat
call is a real model generation). The optional LLM (for the oracle's persona-residue judge and the
adaptive strategist) is wired through the engine's own get_llm() and is OFF unless --adapt is set.
"""

from __future__ import annotations

import argparse
import sys

from . import encoders as enc
from .payloads import FAMILIES
from .report import promote_to_fixtures, render_report
from .runner import run_campaign
from .target import HttpChatTarget, MockTarget


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m rageval.redteam",
        description="Adaptive prompt-injection red-team harness (authorized self-test of the local "
                    "engine). Computes scanner-evasion rate + end-to-end ASR.",
    )
    p.add_argument("--target", choices=["mock", "http"], default="mock",
                   help="Where to fire attacks. 'mock' (default) runs offline/in CI; 'http' hits a "
                        "live /chat (slow).")
    p.add_argument("--base-url", default="http://localhost:8000",
                   help="Base URL for --target http (default: http://localhost:8000).")
    p.add_argument("--families", default=None,
                   help=f"Comma-separated attack families to include. Available: {', '.join(FAMILIES)}.")
    p.add_argument("--encoders", default=None,
                   help=f"Comma-separated encoders. Available: {', '.join(enc.ENCODERS)}.")
    p.add_argument("--max-cases", type=int, default=None,
                   help="Cap the number of cases fired (use a small bound for live runs).")
    p.add_argument("--trials", type=int, default=1,
                   help="Fire each payload N times and report a per-payload success RATE "
                        "(compliance is stochastic on a live model). Errors are excluded from the "
                        "rate denominator. Default 1 (single-shot).")
    p.add_argument("--report-path", default=None,
                   help="Write the markdown report here (also printed to stdout).")
    p.add_argument("--promote-path", default=None,
                   help="Write promote_to_fixtures() Attack snippets here (human review before use).")
    p.add_argument("--adapt", action="store_true",
                   help="Enable one adaptive LLM round (needs an LLM backend; off by default).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.target == "http":
        target = HttpChatTarget(args.base_url)
        print(f"[redteam] target=http {args.base_url} — live model calls are slow (~60-90s each).",
              file=sys.stderr)
    else:
        target = MockTarget()

    # The oracle's persona-residue judge + the adaptive strategist need an LLM; only resolve one if
    # --adapt is requested (otherwise stay fully offline-capable).
    llm = None
    if args.adapt:
        try:
            from ..llm import get_llm

            llm = get_llm()
        except Exception as e:  # noqa: BLE001 — no backend → run deterministically anyway
            print(f"[redteam] --adapt requested but no LLM backend: {e}", file=sys.stderr)

    records, summary = run_campaign(
        target,
        families=_csv(args.families),
        encoders=_csv(args.encoders),
        max_cases=args.max_cases,
        trials=args.trials,
        llm=llm,
        adapt=args.adapt,
    )

    report = render_report(records, summary)
    print(report)

    if args.report_path:
        with open(args.report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[redteam] report written to {args.report_path}", file=sys.stderr)

    if args.promote_path:
        with open(args.promote_path, "w", encoding="utf-8") as f:
            f.write(promote_to_fixtures(records))
        print(f"[redteam] promotion snippets written to {args.promote_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
