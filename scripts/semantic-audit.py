#!/usr/bin/env python3
"""semantic-audit.py — generic SEMANTIC leak gate (the soft, judgement layer).

A hostile-reviewer LLM reads the changed files (or a path) against the generic policy
pack (policies/generic.pack.yaml) and asks: "would publishing this leak secrets, PII, or
private context, or look bad?" It prints severity-tiered findings and an overall verdict.
Critical/High findings -> exit non-zero (fail CI). This is the GENERIC layer only; the
deterministic grep gate (scripts/leak-check.sh) is the separate hard gate — they compose.

DUAL BACKEND (mirrors the engine's rageval/llm.py):
  * CLI backend — the local `claude` CLI (Claude Code) signed in with a Claude.ai
    subscription. We shell out to `claude -p --output-format json`. Spends NO API
    credits. This is what the maintainer runs LOCALLY for free.
  * API backend — the official `anthropic` SDK over HTTPS, using ANTHROPIC_API_KEY.
    Costs credits. This is what CI uses (CI has no `claude` CLI / subscription).
Selection priority (override with SEMANTIC_AUDIT_BACKEND=cli|api):
  1. SEMANTIC_AUDIT_BACKEND explicit override
  2. `claude` CLI on PATH      → cli   (free local subscription — preferred)
  3. ANTHROPIC_API_KEY present → api
  4. otherwise                 → graceful skip (exit 0) so CI is green before the key is set

Usage:
  python scripts/semantic-audit.py                       # scan staged diff (local CLI if present)
  python scripts/semantic-audit.py --base origin/main    # scan vs a ref
  python scripts/semantic-audit.py PATH...               # scan files/dirs
  ANTHROPIC_API_KEY=... python scripts/semantic-audit.py # force the API path in CI

Exit codes: 0 = PASS (or graceful skip), 1 = FAIL (Critical/High found), 2 = setup/transport error.
Dependencies: PyYAML; the `anthropic` SDK only for the API backend (already a project dep).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = REPO_ROOT / "policies" / "generic.pack.yaml"

MODEL = os.environ.get("SEMANTIC_AUDIT_MODEL", "claude-sonnet-4-5")
MAX_BYTES_PER_FILE = 60_000          # truncate huge files to keep the prompt bounded
MAX_TOTAL_BYTES = 400_000            # overall budget across all files

# Paths that legitimately hold synthetic fixture secrets/PII (mirror leak-check.sh).
EXCLUDE_PREFIXES = ("data/sample/", "tests/", ".venv/", ".git/", "__pycache__/")
EXCLUDE_SUFFIXES = (".lock", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip",
                    ".rar", ".aab", ".apk", ".jks", ".sqlite", ".sqlite3")


def sh(*args: str) -> str:
    return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True).stdout


def is_excluded(rel: str) -> bool:
    rel = rel.lstrip("./")
    if rel == "scripts/semantic-audit.py":
        return True
    return rel.startswith(EXCLUDE_PREFIXES) or rel.endswith(EXCLUDE_SUFFIXES)


def collect_targets(args: argparse.Namespace) -> list[Path]:
    files: list[str] = []
    if args.paths:
        for p in args.paths:
            path = Path(p)
            if path.is_dir():
                files += [str(f) for f in path.rglob("*") if f.is_file()]
            elif path.is_file():
                files.append(str(path))
    elif args.base:
        out = sh("git", "diff", "--name-only", "--diff-filter=ACMR", args.base, "HEAD")
        files = [l for l in out.splitlines() if l.strip()]
    else:
        out = sh("git", "diff", "--cached", "--name-only", "--diff-filter=ACMR")
        files = [l for l in out.splitlines() if l.strip()]

    seen: set[str] = set()
    result: list[Path] = []
    for f in files:
        rel = str(Path(f))
        if rel in seen or is_excluded(rel):
            continue
        seen.add(rel)
        fp = (REPO_ROOT / f) if not Path(f).is_absolute() else Path(f)
        if fp.is_file():
            result.append(fp)
    return result


def read_text(fp: Path) -> str | None:
    try:
        data = fp.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:4096]:          # binary
        return None
    text = data.decode("utf-8", errors="replace")
    if len(text) > MAX_BYTES_PER_FILE:
        text = text[:MAX_BYTES_PER_FILE] + "\n...[truncated]..."
    return text


def build_bundle(targets: list[Path]) -> str:
    chunks: list[str] = []
    total = 0
    for fp in targets:
        text = read_text(fp)
        if text is None:
            continue
        rel = fp.relative_to(REPO_ROOT) if str(fp).startswith(str(REPO_ROOT)) else fp
        block = f"===== FILE: {rel} =====\n{text}\n"
        if total + len(block) > MAX_TOTAL_BYTES:
            chunks.append(f"===== [remaining files omitted: budget reached] =====\n")
            break
        chunks.append(block)
        total += len(block)
    return "".join(chunks)


SYSTEM_PROMPT = """\
You are a HOSTILE pre-publication reviewer for an open-source repository: opposing counsel
crossed with a security auditor. The author is about to make these files PUBLIC. Your sole
job: find anything that would leak secrets/credentials, PII, or private/employer context, or
that would embarrass the author if a hostile reader quoted it — using the supplied GENERIC
policy pack as your rubric (grep_patterns are seeds; hunt the shapes/ambiguous cases a regex
misses). Honor `clean_contrast`: synthetic fixture secrets/PII that exist to exercise a
redaction pipeline (sample corpus / test suite, reserved TLDs .example/.test) are NOT findings.

Return ONLY a JSON object, no prose, matching:
{
  "findings": [
    {"file": "<path>", "line_hint": "<line or quote>", "severity": "Critical|High|Medium|Low",
     "category": "<secrets|local-path|pii|identity-deanon|embarrassment|do-not-ship|other>",
     "reason": "<why a hostile reader weaponizes this>", "remediation": "<concrete fix>"}
  ],
  "verdict": "PASS|FAIL",
  "summary": "<one sentence>"
}
verdict is FAIL if and only if any finding is Critical or High. If nothing is wrong, return an
empty findings array and verdict PASS."""


def resolve_backend() -> str | None:
    """Pick the LLM backend. Returns 'cli', 'api', or None (no backend → skip).

    Prefers the local `claude` CLI (free subscription) over the paid API, matching the
    engine's selection spirit but reversing the default: a human running this locally
    wants their subscription, while CI (no CLI) falls through to the API key."""
    override = (os.environ.get("SEMANTIC_AUDIT_BACKEND") or "").strip().lower()
    if override == "cli":
        return "cli" if shutil.which("claude") else None
    if override == "api":
        return "api" if os.environ.get("ANTHROPIC_API_KEY") else None
    if shutil.which("claude"):
        return "cli"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api"
    return None


def _build_user_prompt(policy_text: str, bundle: str) -> str:
    return (
        "GENERIC POLICY PACK (rubric):\n\n```yaml\n" + policy_text + "\n```\n\n"
        "CHANGED FILES TO REVIEW:\n\n" + bundle +
        "\n\nReview every file above. Respond with the JSON object only."
    )


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):                       # tolerate ```json fences
        raw = raw.split("```", 2)[1]
        raw = raw[4:] if raw.lstrip().startswith("json") else raw
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start:end + 1])
        raise


def _call_api(user: str) -> str:
    """API backend — official anthropic SDK over HTTPS (costs credits; used by CI)."""
    try:
        import anthropic
    except ImportError:
        print("semantic-audit: the `anthropic` SDK is not installed "
              "(`pip install anthropic`).", file=sys.stderr)
        sys.exit(2)
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    msg = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content
                   if getattr(b, "type", "") == "text").strip()


def _call_cli(user: str) -> str:
    """CLI backend — shell out to the local `claude` CLI (free subscription, local only).
    Mirrors rageval/llm.py: `claude -p --output-format json --append-system-prompt`."""
    claude = shutil.which("claude")
    if not claude:
        raise RuntimeError("`claude` CLI not on PATH")
    cmd = [claude, "-p", "--output-format", "json", "--model", MODEL,
           "--append-system-prompt", SYSTEM_PROMPT, user]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI exited {proc.returncode}: {proc.stderr[:300]}")
    # The CLI prints one JSON envelope; its `result` field holds the model text.
    try:
        env = json.loads(proc.stdout)
    except json.JSONDecodeError:
        env = None
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    env = json.loads(line)
                except json.JSONDecodeError:
                    continue
    if env is None:
        raise RuntimeError(f"could not parse claude CLI output: {proc.stdout[:300]}")
    if env.get("is_error"):
        raise RuntimeError(f"claude CLI error: {str(env.get('result',''))[:300]}")
    return str(env.get("result", "")).strip()


def call_llm(policy_text: str, bundle: str, backend: str) -> dict:
    user = _build_user_prompt(policy_text, bundle)
    raw = _call_cli(user) if backend == "cli" else _call_api(user)
    return _extract_json(raw)


SEV_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def report(result: dict) -> int:
    findings = sorted(result.get("findings", []),
                      key=lambda f: SEV_ORDER.get(f.get("severity", "Low"), 9))
    if not findings:
        print("semantic-audit: PASS — no leak/embarrassment findings.")
        return 0
    print("semantic-audit findings:\n")
    blocking = False
    for f in findings:
        sev = f.get("severity", "Low")
        if sev in ("Critical", "High"):
            blocking = True
        print(f"  [{sev}] {f.get('file','?')} :: {f.get('line_hint','')}")
        print(f"        category : {f.get('category','?')}")
        print(f"        reason   : {f.get('reason','')}")
        print(f"        fix      : {f.get('remediation','')}\n")
    verdict = result.get("verdict") or ("FAIL" if blocking else "PASS")
    print(f"summary : {result.get('summary','')}")
    print(f"VERDICT : {verdict}")
    return 1 if (verdict == "FAIL" or blocking) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Generic semantic leak audit (LLM hostile reviewer).")
    ap.add_argument("paths", nargs="*", help="files/dirs to scan (default: staged diff)")
    ap.add_argument("--base", help="scan files changed vs this git ref (e.g. origin/main)")
    args = ap.parse_args()

    backend = resolve_backend()
    if backend is None:
        # No local `claude` CLI and no ANTHROPIC_API_KEY → skip gracefully so CI is
        # green before the secret is configured (and local runs without either don't fail).
        print("semantic-audit: no LLM backend (no `claude` CLI, no ANTHROPIC_API_KEY) "
              "— skipping semantic gate.", file=sys.stderr)
        return 0

    if not POLICY_PATH.exists():
        print(f"semantic-audit: missing policy pack at {POLICY_PATH}", file=sys.stderr)
        return 2

    targets = collect_targets(args)
    if not targets:
        print("semantic-audit: PASS — no changed files to review.")
        return 0

    print(f"semantic-audit: reviewing {len(targets)} file(s) via {backend} backend "
          f"({MODEL})...", file=sys.stderr)
    bundle = build_bundle(targets)
    try:
        result = call_llm(POLICY_PATH.read_text(), bundle, backend)
    except Exception as e:  # noqa: BLE001 — a broken audit must not silently pass
        print(f"semantic-audit: ERROR talking to the API: {e}", file=sys.stderr)
        return 2
    return report(result)


if __name__ == "__main__":
    sys.exit(main())
