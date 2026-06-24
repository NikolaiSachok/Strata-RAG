# Adaptive red-team for prompt-injection (design + teaching note)

> **What this is.** An independent, *adaptive* adversarial harness that attacks the engine's own
> chat surface to **discover** prompt-injection bypasses, **measure** how often they work, and
> **grow** the regression corpus from what it finds. It complements — does not replace — the static
> fixture suite (`tests/attack_fixtures.py` + `evaluate_injection_defense()`).
>
> Implements [#30](https://github.com/NikolaiSachok/Strata-RAG/issues/30).

## Why a static fixture suite isn't enough

`tests/attack_fixtures.py` is a *labelled, deterministic* attack corpus with an attack-success-rate
(ASR) gate in CI. That is the **right per-commit layer**: fast, reproducible, regression-proof. But it
has two structural limits:

1. **It re-tests the same payloads every run.** It proves *known* attacks still get caught — not that
   a motivated adversary can't find a *new* one. Author-written fixtures carry the author's blind spot
   (the same *"an author can't review their own code"* problem we solved for code with the review
   panel — here, for security).
2. **The deterministic scanner is literal.** `scan_for_injection` is a set of ASCII regexes. Frontier
   models already ignore naive payloads (`"ignore previous instructions"`), so the live threat is
   **obfuscated** and **composed** payloads — and literal regex is blind to all of them:
   - enclosed alphanumerics — `ⓘⓝⓢⓣⓡⓤⓒⓣⓘⓞⓝ` · full-width · homoglyphs (Cyrillic/Greek lookalikes)
   - zero-width-joiner splitting · leetspeak · base64 · **morse** · ROT13
   - the instruction hidden as a **poem / acrostic**, a translation, or "decode this and follow it"
   - **multi-turn split**: turns benign in isolation that **compose** an attack
   - **indirect**: the payload inside a *retrieved doc / tool observation*, not the user turn

## The two numbers (the conceptual core)

The harness reports two *different* rates. Conflating them is the classic mistake.

| Metric | Question it answers | Expectation |
| --- | --- | --- |
| **Scanner-evasion rate** | Did the *deterministic gate* fail to flag the payload? | High for obfuscated input — and that's the point: it **quantifies the gap** and motivates a normalization pre-pass. |
| **End-to-end ASR** | Did the *target model actually comply*? | The number that matters. Judged by an **oracle on the final answer**, independent of the scanner. |

A payload can evade the scanner yet still fail to hijack the model (model robustness + spotlight/
sentinel framing hold) — or slip past both. We need to see each independently to know *which layer*
earns its place, exactly as `evaluate_injection_defense()` toggles layers and watches ASR move.

## Architecture — a team of roles

Deterministic where it can be (reproducible, free, CI-gateable); LLM only where judgement is
irreducible (**rules enforce, the LLM proposes** — the engine's standing principle).

```
 Strategist ──> Encoder/Mutator ──> Scenario builder ──> Executor ──> Oracle ──> Reporter
 (adaptive LLM    (deterministic       (multi-turn +       (fire at     (success?   (ASR/evasion
  generator)       obfuscation)         indirect docs)      target)      canary +     table +
                                                                         LLM judge)   promote)
```

- **Strategist** (`agent.py`) — adaptive LLM attacker: crafts novel payloads per family, picks
  encoders, and *adapts* to prior results (what got flagged vs. what worked). Falls back to mutating
  the static catalog when no LLM backend is configured, so the harness runs offline.
- **Encoder/Mutator** (`encoders.py`) — **deterministic** obfuscation transforms, each with an
  inverse where decodable (so the oracle and a future normalizer can round-trip). *The concrete
  portfolio piece.*
- **Scenario builder** (`scenarios.py`) — multi-turn split sequences + indirect-doc templates (payload
  planted in fake retrieved content).
- **Executor / Target** (`target.py`) — a `Target` protocol with `HttpChatTarget` (live `/chat`),
  `MockTarget` (deterministic, for CI), and `InProcessTarget` (the agent in-process — a **v1 stub**
  that raises `NotImplementedError` with a TODO; the live path is covered by `HttpChatTarget`).
  Direct, multi-turn, and indirect delivery (`MockTarget` reads the planted doc directly; a live
  target can't inject a corpus doc in v1, so the doc is quoted in a framed user turn).
- **Oracle** (`oracle.py`) — **deterministic canary detection first** (did `RT_CANARY_*`, an exfil
  URL, a system-prompt marker, or the forced persona string appear in the answer?), then an **LLM
  behavioral-hijack judge** only for the ambiguous residue.
- **Reporter** (`report.py`) — ASR + evasion-rate by family × encoder × delivery; emits a markdown
  table and `promote_to_fixtures()` → valid `Attack` objects for **human review before promotion**
  (the corpus grows, but a human still gates what becomes a permanent test).

## Attack taxonomy

The three axes are orthogonal — a case is one **family** × one **encoder** × one **delivery**.
Note that *family* (the malicious GOAL) is distinct from *delivery* (the CHANNEL): multi-turn and
indirect are **deliveries**, not families.

- **Families** (the goal — `payloads.py::BASE_INTENTS`): instruction-override · role-persona-override
  · prompt-leak-exfil · output-format-hijack · data-exfil (markdown-image / URL) · tool-abuse.
- **Encoders** (the obfuscation — `encoders.py::ENCODERS`): plain · enclosed-alnum · full-width ·
  homoglyph · zero-width-split · leetspeak · base64 · morse · ROT13 · acrostic/poem. *(translation
  to a non-EN language is PLANNED, not yet shipped.)*
- **Delivery** (the channel — `scenarios.py`): direct user turn · multi-turn split sequence ·
  indirect via retrieved doc.

## Safety & open-core boundary

- Every payload is **fictional and domain-neutral**, keyed to **canary tokens** (e.g.
  `RT_CANARY_PWNED_7f3a`) — never real exploit content. The harness attacks **only the local engine**
  (authorized self-test).
- **Public (this repo):** the generic red-team engine + neutral payload library + encoders.
  **Private (Career overlay):** any corpus-mimicking payloads and runs against the real corpus.
  Nothing domain-specific ships. Leak-grep + the semantic audit gate the push as usual.
- **Not in the per-commit hot path.** CI runs a bounded deterministic subset (encoders + oracle +
  `MockTarget` ASR). The full adaptive run against a live model is opt-in (`python -m rageval.redteam`)
  / scheduled.

## What v1 delivers vs. what it motivates

**v1 (this branch):** the package + tests green on `MockTarget` + a documented live run producing an
ASR/evasion table. The adaptive Strategist may be a catalog-mutating stub.

**The follow-up it justifies (separate issue):** a **normalization pre-pass** before
`scan_for_injection` — NFKC fold, homoglyph map, strip zero-width, decode-and-rescan (morse/base64/
ROT13) — then re-run the red-team and show the **evasion rate drop measurably**. That before/after is
the headline result: *I built a red-team that broke my own scanner, then closed the gap with numbers
to prove it.*
