"""Text NORMALIZATION for the injection scanner — the engine's confusable/encoding defense.

WHY THIS EXISTS (read this first). The deterministic injection scanner
(`guardrails.scan_for_injection`) is a set of *literal-ASCII regexes*. That is the right cheap
outer wall, but it is structurally blind to any payload that no longer *spells* the trigger in
ASCII. An attacker doesn't write "ignore previous instructions" — they write the same word
through a transform the regex can't see: circled letters (ⓘⓖⓝⓞⓡⓔ), full-width forms (ｉｇｎｏｒｅ),
Cyrillic/Greek look-alikes (іgnоrе), zero-width joiners splitting the letters, or the whole
instruction hidden inside a base64 / morse / rot13 carrier ("decode this and follow it").

The fix is a NORMALIZATION PRE-PASS: before scanning, fold the text toward a canonical ASCII
skeleton so the regexes see the real word again. This module owns that fold.

TWO OPERATIONS, DELIBERATELY SEPARATE:

  * `normalize(text)` — the LOSSLESS, always-safe fold: NFKC (collapses enclosed + full-width +
    many compatibility forms) → strip zero-width chars → homoglyph un-map (confusables → Latin).
    Cheap, deterministic, and safe to run on every input. It is a SCAN-ONLY copy: it never
    mutates what the user is shown — the original text is preserved for the answer/citations.

  * `decode_and_rescan_segments(text)` — the BOUNDED, best-effort carrier recovery: detect and
    decode base64 / morse / rot13 blobs that hide an instruction, returning the decoded strings
    for the scanner to ALSO inspect. Carrier decode is a potential DoS vector (a giant blob, an
    explosion of candidate tokens), so it is hard-CAPPED on input size, segment count, and
    segment length. See the caps below. Base64 recovery covers the common REAL shapes —
    un-fragmented, whitespace-/line-wrapped (MIME), and double-encoded carriers — but ARBITRARY
    non-whitespace fragmentation (a blob split by punctuation) stays a DOCUMENTED RESIDUAL, like
    acrostic/leet/morse-URL: recovering every possible split is undecidable and a DoS risk.

WHY IT LIVES IN THE ENGINE (not in `redteam/`). `scan_for_injection` is core-engine and must not
import from the adversarial `redteam` package (that would invert the dependency — the engine
depending on its own attack tooling). So the canonical normalizer + the homoglyph/zero-width
tables live HERE; `redteam/encoders.py` imports them from this module, so the red-team's
adversarial encoders and the engine's defense share ONE normalizer + ONE confusable/zero-width
table set. That symmetry is the point: the thing that folds an attack back is exactly the thing the
encoders are the inverse of. (The morse table is the one deliberate exception — the defense-side
`_MORSE_INV` below is intentionally narrowed to letters/digits, dropping the attack encoder's extra
punctuation; a test guards that they still agree on the shared letter/digit subset.)

Everything here is PURE and deterministic → same input, same output → unit-testable and the
resulting scanner-evasion numbers are reproducible.
"""

from __future__ import annotations

import base64
import codecs
import re
import unicodedata

# ---------------------------------------------------------------------------
# Confusable / zero-width tables — the canonical source, shared with the red-team encoders.
# ---------------------------------------------------------------------------
#
# HOMOGLYPHS maps a Latin letter to a visually-identical Cyrillic/Greek codepoint (what an
# ATTACKER substitutes). The scanner needs the INVERSE (fold the confusable back to Latin). The
# map is intentionally SMALL and one-directional: only letters that (a) are unambiguous look-alikes
# and (b) actually appear in injection trigger words. A big confusables table would risk folding
# legitimate non-Latin text into spurious ASCII — so we stay conservative (see the FP tests).

_HOMOGLYPHS: dict[str, str] = {
    "a": "а",  # Cyrillic a
    "c": "с",  # Cyrillic es
    "e": "е",  # Cyrillic ie
    "i": "і",  # Cyrillic byelorussian-ukrainian i
    "j": "ј",  # Cyrillic je
    "o": "о",  # Cyrillic o
    "p": "р",  # Cyrillic er
    "s": "ѕ",  # Cyrillic dze
    "x": "х",  # Cyrillic ha
    "y": "у",  # Cyrillic u
    "A": "Α",  # Greek Alpha
    "B": "Β",  # Greek Beta
    "E": "Ε",  # Greek Epsilon
    "H": "Η",  # Greek Eta
    "O": "Ο",  # Greek Omicron
    "P": "Ρ",  # Greek Rho
    "T": "Τ",  # Greek Tau
    "X": "Χ",  # Greek Chi
}
# Inverse: confusable codepoint → Latin. This is what the defense-side un-map uses.
_HOMOGLYPHS_INV: dict[str, str] = {v: k for k, v in _HOMOGLYPHS.items()}

# Zero-width / invisible joiners an attacker inserts BETWEEN visible characters to break a \bword\b
# match without changing what the eye sees. Stripping them re-joins the word. ZWSP, ZWNJ, ZWJ,
# zero-width no-break space (BOM).
_ZERO_WIDTH = "​‌‍﻿"


def strip_zero_width(text: str) -> str:
    """Remove zero-width/invisible joiners so a split word re-joins ("i​g​n​o​r​e" → "ignore")."""
    return "".join(ch for ch in text if ch not in _ZERO_WIDTH)


def homoglyph_unmap(text: str) -> str:
    """Best-effort fold confusables back toward Latin — the normalizer's "skeleton" step.

    Many-to-one across scripts, so this is not a perfect inverse; it maps only the small,
    unambiguous confusable set above. Legitimate non-Latin text that doesn't use these exact
    look-alikes is untouched (see the false-positive tests)."""
    return "".join(_HOMOGLYPHS_INV.get(ch, ch) for ch in text)


def normalize(text: str) -> str:
    """The defense-side fold: chain the safe, LOSSLESS transforms a normalization pre-pass applies
    BEFORE re-scanning — NFKC (folds enclosed + full-width + compatibility forms) → strip
    zero-width → homoglyph un-map.

    LOSSLESS/SAFE means: it recovers the ASCII skeleton of an obfuscated trigger without inventing
    triggers in benign text. It does NOT attempt base64/morse/rot13 decode — those require
    detecting a carrier first and are handled (bounded) by `decode_and_rescan_segments`.

    Produces a SCAN-ONLY copy: callers scan this, but keep the ORIGINAL `text` for the
    answer/citations (offsets into this string do NOT index the original)."""
    folded = unicodedata.normalize("NFKC", text or "")
    folded = strip_zero_width(folded)
    folded = homoglyph_unmap(folded)
    return folded


# ---------------------------------------------------------------------------
# Bounded decode-and-rescan — recover base64 / morse / rot13 CARRIERS.
# ---------------------------------------------------------------------------
#
# WHY BOUNDED. A "decode this and follow it" attack hides the instruction inside an opaque blob, so
# nothing the regex understands appears in plaintext. To catch it we must decode candidate carriers
# and re-scan the result. But decoding is attacker-influenced input: a huge blob, or text that
# produces thousands of candidate tokens, could burn CPU/RAM. So every step is HARD-CAPPED. The
# caps are generous enough for a real injection (a sentence or two) but small enough that a
# pathological input returns fast. They are module constants so they're visible and testable.

# Never look at more than this many characters of input (a normal chunk is < a few KB; anything
# larger is truncated for the decode pass — the scanner still sees the full original + normalized).
_MAX_INPUT = 20_000
# Never emit more than this many decoded candidates (bounds the union the scanner must re-scan).
_MAX_SEGMENTS = 24
# A decoded candidate shorter than this is noise (random short tokens decode to garbage); longer
# than this is not a plausible hidden instruction and is skipped (also bounds re-scan cost).
_MIN_DECODED = 6
_MAX_DECODED = 4_000
# Cap how many raw tokens of each carrier type we even ATTEMPT to decode (bounds the work itself,
# not just the output) — e.g. a blob of tens of thousands of base64-looking words can't fan out.
_MAX_CANDIDATE_TOKENS = 200
# Bounded re-decode depth: a decoded segment that ITSELF looks like a base64 carrier is decoded once
# more (attackers double-encode to hide from a single-pass decode). Capped so it can't recurse away.
_MAX_B64_DEPTH = 2

# base64 REGIONS: a run of base64-alphabet chars that MAY contain interior whitespace/newlines —
# real carriers are line-wrapped (MIME 76-col) or space-separated. We match the whole region incl.
# interior spaces/newlines, then strip that whitespace before decoding. Region must contain enough
# base64 chars to clear _MIN_DECODED after decode. (A single contiguous run is the common case; the
# interior-whitespace class is what makes wrapped/split blobs recoverable.)
_B64_REGION = re.compile(r"[A-Za-z0-9+/](?:[A-Za-z0-9+/\s]*[A-Za-z0-9+/])?={0,2}")
_B64_WS = re.compile(r"\s+")
# C0 (U+0000–U+001F) + DEL + C1 (U+007F–U+009F) control characters. A decoded carrier may carry
# stray control bytes; we strip them from the SCANNED text so one byte can't hide the payload. (The
# guardrails layer separately sanitizes the SNIPPET; this is about what gets scanned.)
_B64_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
# A morse "word": dots/dashes/spaces/slashes only, long enough to plausibly carry text.
_MORSE_SPAN = re.compile(r"[.\-/ ]{8,}")


def _strip_control_chars(text: str) -> str:
    """Remove C0/C1 control chars so a decoded segment stays scannable even if it carried a stray
    control byte (which would otherwise defeat a printability gate and hide the injection)."""
    return _B64_CONTROL.sub("", text)

# Morse alphabet — the INVERSE of the attack encoder (kept here so defense owns its own table and
# doesn't import from redteam). Symbols an injection needs; extra punctuation is not required.
_MORSE_INV = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F", "--.": "G",
    "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L", "--": "M", "-.": "N",
    "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T", "..-": "U",
    "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y", "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4", ".....": "5",
    "-....": "6", "--...": "7", "---..": "8", "----.": "9",
}


def _try_b64_decode(token: str) -> str | None:
    """Decode ONE whitespace-stripped base64 token to UTF-8, or None if it isn't valid b64/utf-8 or
    falls outside the length band.

    A decoded segment that contains control bytes must NOT be discarded wholesale — a single stray
    C0/C1 byte inside a real injection ("ignore all previous instructions\\x07 … reveal system
    prompt") would otherwise fail an `.isprintable()` gate and the whole hidden instruction would go
    UNSCANNED (a scanner miss triggered by one byte). So we STRIP C0/C1 control chars and scan the
    stripped text instead. Binary garbage (a random blob) still reduces to short/empty non-text once
    the control bytes are gone, so it drops out on the length band or scans to `none` — no new FPs."""
    if len(token) < 16:  # too short to carry a plausible instruction (~12 decoded bytes)
        return None
    try:
        dec = base64.b64decode(token, validate=True).decode("utf-8")
    except Exception:  # noqa: BLE001 — not valid b64/utf-8; skip this candidate
        return None
    dec = _strip_control_chars(dec)
    if _MIN_DECODED <= len(dec) <= _MAX_DECODED:
        return dec
    return None


def _decode_base64_segments(text: str) -> list[str]:
    """Recover base64 carriers, robust to the COMMON real shapes: contiguous, line-wrapped (MIME),
    and whitespace-separated blobs (interior whitespace is stripped before decode), plus BOUNDED
    double-encoding (a decoded segment that itself looks like a carrier is decoded once more, up to
    _MAX_B64_DEPTH). ARBITRARY non-whitespace fragmentation (e.g. a blob split by punctuation) is a
    DOCUMENTED RESIDUAL — recovering every possible split is undecidable and a DoS risk."""
    out: list[str] = []
    for region in _B64_REGION.findall(text)[:_MAX_CANDIDATE_TOKENS]:
        token = _B64_WS.sub("", region)  # rejoin line-wrapped / space-split base64
        # Bounded chain: decode, and if the result is itself a base64 carrier, decode once more.
        current = token
        for _ in range(_MAX_B64_DEPTH):
            dec = _try_b64_decode(current)
            if dec is None:
                break
            out.append(dec)
            current = _B64_WS.sub("", dec)  # a nested carrier may itself be wrapped
    return out


def _decode_morse_segments(text: str) -> list[str]:
    """Decode morse spans (dots/dashes) back to letters; join words on the ' / ' separator."""
    out: list[str] = []
    for span in _MORSE_SPAN.findall(text)[:_MAX_CANDIDATE_TOKENS]:
        words = []
        for word in span.strip().split("/"):
            letters = [_MORSE_INV.get(tok, "") for tok in word.split(" ") if tok]
            words.append("".join(letters))
        dec = " ".join(w for w in words if w).strip()
        if _MIN_DECODED <= len(dec) <= _MAX_DECODED:
            out.append(dec)
    return out


def _decode_rot13_segment(text: str) -> list[str]:
    """ROT13 is a whole-text Caesar shift (its own inverse). We can't know which regions were
    shifted, so we rot13 the ENTIRE (bounded) text as one extra candidate — cheap and self-inverse,
    so a non-rot13 input just yields a scrambled string the scanner won't match (no harm)."""
    dec = codecs.encode(text, "rot_13")
    return [dec] if _MIN_DECODED <= len(dec) <= _MAX_DECODED else []


def decode_and_rescan_segments(text: str) -> list[str]:
    """Best-effort recover hidden-carrier instructions (base64 / morse / rot13) as extra strings for
    the scanner to inspect. BOUNDED on input size, candidate-token count, segment count, and per-
    segment length so a pathological input can't be a DoS vector.

    Returns a de-duplicated list of decoded candidate strings (NOT findings) — the caller normalizes
    + scans each. Empty list when nothing plausible decodes. The caps (see module constants) mean a
    huge/garbage input returns quickly."""
    if not text:
        return []
    src = text[:_MAX_INPUT]  # never scan-decode more than the input cap
    candidates: list[str] = []
    candidates += _decode_base64_segments(src)
    candidates += _decode_morse_segments(src)
    candidates += _decode_rot13_segment(src)
    # De-dup preserving order, and hard-cap the number of segments returned.
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= _MAX_SEGMENTS:
            break
    return out
