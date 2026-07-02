"""The OCR provider seam — one interface, swappable backends.

WHY this matters for RAG: a SCANNED document is an image of text. A born-digital extractor
(python-docx, a PDF text layer) sees nothing there — the words only exist as pixels, so they must
be OCR'd before they can be chunked, embedded, and retrieved. The public strata-insurance-corpus
has a whole `ocr` golden class: scan-only documents whose facts live on no born-digital page.

Like the vision seam (vision.py) and the text LLM (llm.py), we factor the OCR engine behind a
small BACKEND seam so the backend is CONFIGURABLE and carries NO hard import dependency.

THE CONTRACT every backend satisfies (OcrProvider):

    ocr(image) -> OcrResult   # {text, provider, confidence?, regions?}

`image` is a `vision.ImageInput` (bytes or a Path). `OcrResult.regions` carries per-word/line
boxes + confidences WHERE THE BACKEND EXPOSES THEM (Tesseract does; the LLM backend does not, so
those fields stay None — an honest "not available" rather than a fabricated number).

Backend selection mirrors the other seams: a registry keyed by name, chosen by
`RAGEVAL_OCR_PROVIDER`.
  * "tesseract" (DEFAULT where feasible) — local, free, offline, via `pytesseract` + the system
    `tesseract` binary. BOTH must be present; if either is missing we degrade to skip-with-reason
    (never crash), so a box without Tesseract just falls through cleanly.
  * "llm" — LLM-as-OCR: reuse the vision provider (vision.py) to READ a page image as text. Lets
    the same deployment do OCR with no extra dependency when a vision model is already configured.
  * "mock" — deterministic, dependency-free, for CI. Never calls a model or a binary.
  * register your own (a cloud OCR service, a local VLM) via `register_ocr_provider`.

GRACEFUL DEGRADATION (the enrichment failure model): `get_ocr_provider(..., strict=False)`
returns `(None, reason)` instead of raising when the backend is absent/misconfigured, so a
per-image OCR failure is a SKIP with a recorded reason — it never aborts an ingest run.

SECURITY CONTRACT for consumers (stated here, wired later by the OCR ingestion path): OCR'd text
is UNTRUSTED — it comes from an image we did not author and may carry a prompt-injection payload.
Consumers MUST route it through redaction + the injection guards (redact.py + guardrails.py)
BEFORE embedding, exactly like any retrieved chunk, and record `modality=ocr` + the source image
+ the provider name in the payload for auditability. This seam returns RAW text; it does not
sanitise for you.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from .config import SETTINGS, Settings
from .vision import ImageInput, VisionError, get_vision_provider, load_image_bytes


class OcrError(RuntimeError):
    """Any backend/binary/decode problem, normalised to one type so a consumer catches ONE
    error regardless of which OCR backend is active (mirrors vision.VisionError / llm.LLMError)."""


@dataclass(frozen=True)
class OcrRegion:
    """One recognised text span with its box + confidence, WHERE the backend exposes them.
    (left, top, width, height) are pixel coordinates; `confidence` is 0–100 (Tesseract's scale)."""

    text: str
    confidence: float | None = None
    box: tuple[int, int, int, int] | None = None  # (left, top, width, height)


@dataclass(frozen=True)
class OcrResult:
    """Extracted text plus provenance + optional structure. `provider` is written into the
    ingestion payload for AUDITABILITY (how this chunk's text was derived). `confidence` is a
    whole-page mean 0–100 when the backend reports per-word scores (Tesseract), else None.
    `regions` is the per-word/line detail when available, else empty."""

    text: str
    provider: str
    confidence: float | None = None
    regions: list[OcrRegion] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@runtime_checkable
class OcrProvider(Protocol):
    """Structural contract (a Protocol): any object with a `name` and an `ocr()` IS a provider."""

    name: str

    def ocr(self, image: ImageInput) -> OcrResult: ...


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class TesseractOcrProvider:
    """Local Tesseract OCR via `pytesseract`. The DEFAULT where feasible: free, offline, and
    exposes per-word confidences + boxes.

    NO hard dependency: `pytesseract` (the Python wrapper) is imported LAZILY in __init__, and we
    check the `tesseract` BINARY is on PATH too — the wrapper installs fine without the binary but
    fails at call time, so we fail fast at construction with a clear reason instead. Either missing
    → OcrError → get_ocr_provider turns it into a skip-with-reason when strict=False."""

    name = "tesseract"

    def __init__(self):
        try:
            import pytesseract  # lazy: only this backend needs it
        except ImportError as e:
            raise OcrError(
                "the 'pytesseract' package is not installed. Run: pip install pytesseract "
                "(and install the system `tesseract` binary)."
            ) from e
        try:
            import PIL.Image  # noqa: F401 - pytesseract reads a PIL image
        except ImportError as e:
            raise OcrError(
                "the 'Pillow' package is not installed. Run: pip install Pillow "
                "(pytesseract reads a PIL image)."
            ) from e
        if not shutil.which("tesseract"):
            raise OcrError(
                "the `tesseract` binary is not on PATH. Install it (e.g. `brew install tesseract` "
                "or `apt-get install tesseract-ocr`)."
            )
        self._pytesseract = pytesseract

    def ocr(self, image: ImageInput) -> OcrResult:
        import io

        import PIL.Image

        data = load_image_bytes(image)
        try:
            img = PIL.Image.open(io.BytesIO(data))
            # image_to_data gives per-word text + confidence + box; we build both the flat text
            # and the structured regions from ONE pass. Output.DICT → column-oriented dict.
            out = self._pytesseract.image_to_data(
                img, output_type=self._pytesseract.Output.DICT
            )
        except Exception as e:  # noqa: BLE001 - normalise to one error type
            raise OcrError(f"tesseract OCR failed: {e}") from e
        regions: list[OcrRegion] = []
        confs: list[float] = []
        words: list[str] = []
        n = len(out.get("text", []))
        for i in range(n):
            word = (out["text"][i] or "").strip()
            if not word:
                continue
            # pytesseract reports -1 for a box with no confidence; skip those from the mean.
            try:
                conf = float(out["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            box = (
                int(out["left"][i]),
                int(out["top"][i]),
                int(out["width"][i]),
                int(out["height"][i]),
            )
            regions.append(
                OcrRegion(text=word, confidence=conf if conf >= 0 else None, box=box)
            )
            words.append(word)
            if conf >= 0:
                confs.append(conf)
        text = " ".join(words)
        mean_conf = round(sum(confs) / len(confs), 2) if confs else None
        return OcrResult(
            text=text,
            provider=self.name,
            confidence=mean_conf,
            regions=regions,
        )


class LlmOcrProvider:
    """LLM-as-OCR: reuse the VISION provider (vision.py) to transcribe a page image to text. Lets
    a deployment that already has a vision model do OCR with no extra dependency. No boxes/
    confidences (a chat model doesn't expose them) — those stay None, honestly.

    It delegates backend resolution to get_vision_provider, so RAGEVAL_VISION_PROVIDER selects the
    underlying model and its degradation applies here too (an unavailable vision backend → OcrError
    → skip-with-reason)."""

    name = "llm"

    # A transcription-focused prompt (distinct from the generic describe prompt): we want the
    # literal text, not a description of the page.
    OCR_PROMPT = (
        "Transcribe ALL text visible in this image verbatim, preserving reading order and line "
        "breaks. Output ONLY the transcribed text — no commentary, no description. If there is no "
        "legible text, output nothing."
    )

    def __init__(self, settings: Settings):
        provider, reason = get_vision_provider(settings, strict=False)
        if provider is None:
            raise OcrError(f"LLM-as-OCR needs a vision backend, which is unavailable: {reason}")
        self._vision = provider

    def ocr(self, image: ImageInput) -> OcrResult:
        try:
            res = self._vision.describe(image, prompt=self.OCR_PROMPT)
        except VisionError as e:
            raise OcrError(f"LLM-as-OCR failed via vision backend {self._vision.name!r}: {e}") from e
        return OcrResult(
            text=res.text,
            provider=f"{self.name}:{res.provider}",
            confidence=None,  # a chat model exposes no calibrated confidence
            regions=[],
            meta={"vision_provider": res.provider, "vision_model": res.model},
        )


class MockOcrProvider:
    """Deterministic, dependency-free OCR for CI and offline dev. Produces a STABLE transcript
    derived from the image bytes so a test can assert on it without a binary or a model."""

    name = "mock"

    def ocr(self, image: ImageInput) -> OcrResult:
        import hashlib

        data = load_image_bytes(image)
        digest = hashlib.sha256(data).hexdigest()[:12]
        return OcrResult(
            text=f"[mock ocr] {len(data)} bytes sha256:{digest}",
            provider=self.name,
            confidence=99.0,
            regions=[
                OcrRegion(text="[mock", confidence=99.0, box=(0, 0, 10, 10)),
                OcrRegion(text="ocr]", confidence=99.0, box=(12, 0, 10, 10)),
            ],
            meta={"bytes": len(data)},
        )


# ---------------------------------------------------------------------------
# Registry + factory (mirrors vision.py)
# ---------------------------------------------------------------------------

OcrBuilder = Callable[[Settings], OcrProvider]

_OCR_REGISTRY: dict[str, OcrBuilder] = {
    "tesseract": lambda s: TesseractOcrProvider(),
    "llm": lambda s: LlmOcrProvider(s),
    "mock": lambda s: MockOcrProvider(),
}


def register_ocr_provider(name: str, builder: OcrBuilder) -> None:
    """PUBLIC extension API: teach the engine a new OCR backend (a cloud OCR service, a local VLM)
    WITHOUT editing this module. `builder(settings) -> OcrProvider` is called lazily by
    get_ocr_provider so construction errors become clean skips. Last-wins (idempotent)."""
    _OCR_REGISTRY[name.lower()] = builder


def available_ocr_providers() -> list[str]:
    """Registered backend names (for /health and diagnostics)."""
    return sorted(_OCR_REGISTRY)


def get_ocr_provider(
    settings: Settings = SETTINGS, *, strict: bool = False
) -> tuple[OcrProvider | None, str | None]:
    """Build the configured OCR backend. Returns `(provider, reason)` — see get_vision_provider
    for the identical contract (degrade → (None, reason) when strict=False, else raise OcrError;
    an UNKNOWN provider name always raises, since a typo'd RAGEVAL_OCR_PROVIDER is a misconfig).

    Usage in a consumer (wired later, by the OCR ingestion path):

        provider, reason = get_ocr_provider(settings)
        if provider is None:
            skip_this_image(reason)           # e.g. "tesseract binary not on PATH"
        else:
            text = provider.ocr(image).text   # then redact + injection-guard BEFORE embedding
    """
    name = (settings.ocr_provider or "tesseract").lower()
    builder = _OCR_REGISTRY.get(name)
    if builder is None:
        raise OcrError(
            f"unknown RAGEVAL_OCR_PROVIDER={name!r}; "
            f"available: {', '.join(available_ocr_providers())}"
        )
    try:
        return builder(settings), None
    except OcrError as e:
        if strict:
            raise
        return None, str(e)


def ocr_status(settings: Settings = SETTINGS) -> dict:
    """Diagnostics for /health: which OCR provider is configured and whether it's usable."""
    provider, reason = get_ocr_provider(settings, strict=False)
    return {
        "provider": settings.ocr_provider,
        "available": provider is not None,
        "reason": reason,
        "registered": available_ocr_providers(),
    }
