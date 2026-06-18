"""The LLM backend — one interface, two implementations.

WHY this matters for RAG: the "G" (generation) and the eval judge both need to
call a language model. We want that call to work in two situations:

  * API backend — you have an ANTHROPIC_API_KEY. Uses the official `anthropic`
    SDK over HTTPS. Costs API credits.
  * CLI backend — you have the local `claude` CLI (Claude Code) signed in with a
    Claude.ai subscription. We shell out to `claude -p --output-format json`,
    which uses your subscription and spends NO API credits.

This is the same dual-backend pattern as the sibling `promosite` subproject, kept
deliberately small here. Both backends expose ONE method:

    complete(system: str, prompt: str) -> str

so generate.py and eval.py never need to know which one is active.

Backend selection (priority):
  1. RAGEVAL_LLM_BACKEND=api|cli   (explicit override)
  2. ANTHROPIC_API_KEY present     → api
  3. `claude` on PATH              → cli
  4. otherwise                     → a clear error
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Protocol

from .config import SETTINGS, Settings

try:
    import anthropic  # only needed for the API backend
except ImportError:  # pragma: no cover - import guard
    anthropic = None


class LLMError(RuntimeError):
    """Raised for any backend/credential/transport problem so callers can catch one type."""


class LLMBackend(Protocol):
    """The contract every backend satisfies. A Protocol (structural typing) lets us
    swap implementations without a shared base class — generate.py/eval.py just need
    `.complete(system, prompt) -> str` and a `.name`."""

    name: str

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str: ...


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def resolve_backend(settings: Settings = SETTINGS) -> str:
    """Decide which backend to use. Returns 'api' or 'cli'. Raises LLMError if no
    credential path is available — surfaced to the user with a fix-it message."""
    override = settings.llm_backend
    if override in ("api", "cli"):
        if override == "api" and not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError("RAGEVAL_LLM_BACKEND=api but ANTHROPIC_API_KEY is not set.")
        if override == "cli" and not shutil.which("claude"):
            raise LLMError("RAGEVAL_LLM_BACKEND=cli but the `claude` CLI is not on PATH.")
        return override
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api"
    if shutil.which("claude"):
        return "cli"
    raise LLMError(
        "No LLM backend available. Either set ANTHROPIC_API_KEY, or install the "
        "`claude` CLI (Claude Code) and sign in with your Claude.ai subscription."
    )


def backend_status(settings: Settings = SETTINGS) -> dict:
    """Diagnostics for /health and the CLI: which backend is active and what's present."""
    try:
        active: str | None = resolve_backend(settings)
    except LLMError:
        active = None
    return {
        "active_backend": active,
        "anthropic_api_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "claude_cli": shutil.which("claude") is not None,
        "model": settings.model,
    }


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

class ApiBackend:
    """Calls Anthropic over HTTPS with the official SDK. Needs ANTHROPIC_API_KEY."""

    name = "api"

    def __init__(self, model: str):
        if anthropic is None:
            raise LLMError("The 'anthropic' package is not installed. Run: pip install anthropic")
        # The SDK reads ANTHROPIC_API_KEY from the environment automatically.
        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:  # noqa: BLE001 - normalise to one error type
            raise LLMError(f"Anthropic API call failed: {e}") from e
        # A response is a list of content blocks; collect the text ones.
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "\n".join(parts).strip()


class CliBackend:
    """Shells out to the local `claude` CLI in headless print mode. Uses the user's
    Claude.ai subscription — no API credits. Text-only.

    `claude -p --output-format json` returns a JSON object whose `result` field holds
    the model's text. We parse that out.
    """

    name = "cli"

    def __init__(self, model: str):
        self.bin = shutil.which("claude")
        if not self.bin:
            raise LLMError("`claude` CLI not found on PATH.")
        self.model = model

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        # Note: the CLI controls its own output length; max_tokens is accepted for a
        # uniform interface but not forwarded (the print mode has no such flag).
        cmd = [
            self.bin, "-p",
            "--output-format", "json",
            "--model", self.model,
            "--append-system-prompt", system,
            prompt,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired as e:
            raise LLMError("claude CLI timed out after 600s") from e
        if proc.returncode != 0:
            raise LLMError(f"claude CLI exited {proc.returncode}: {proc.stderr[:300]}")
        data = _parse_cli_json(proc.stdout)
        if data is None:
            raise LLMError(f"could not parse claude CLI output: {proc.stdout[:300]}")
        if data.get("is_error"):
            raise LLMError(f"claude CLI reported error: {str(data.get('result', ''))[:300]}")
        return str(data.get("result", "")).strip()


def _parse_cli_json(stdout: str) -> dict | None:
    """The CLI usually prints one JSON object; some versions stream several lines.
    Try the whole blob first, then fall back to the last parseable `{...}` line."""
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        last = None
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
        return last


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm(settings: Settings = SETTINGS) -> LLMBackend:
    """Build the active backend. Call this once and reuse the instance."""
    backend = resolve_backend(settings)
    if backend == "api":
        return ApiBackend(settings.model)
    return CliBackend(settings.model)
