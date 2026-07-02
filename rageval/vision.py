"""The vision provider seam — one interface, swappable backends.

WHY this matters for RAG: an image (a promo screenshot, a scanned page, a chart) carries
information a text-only index can't reach. Two Phase-1.5 consumers need a vision model:
caption-then-embed (#1) turns an image into an embeddable DESCRIPTION, and the OCR seam's
"LLM-as-OCR" backend (ocr.py) reads a page image as text. Rather than hardwire one vendor into
those consumers, we factor the model behind a small BACKEND seam — exactly the shape `llm.py`
already uses for text (one `.complete()`, backend chosen by env, graceful degradation, a mock).

THE CONTRACT every backend satisfies (VisionProvider):

    describe(image, *, prompt=None) -> VisionResult   # caption / free-text description
    embed_image(image) -> list[float]                 # OPTIONAL (image-to-image #19); may raise

`image` is an `ImageInput` — either raw `bytes` or a filesystem `Path` — so a caller can pass a
freshly-decoded blob or a file on disk without the provider caring which.

Backend selection mirrors llm.py: a registry keyed by name, chosen by `RAGEVAL_VISION_PROVIDER`.
  * "claude" (DEFAULT) — Anthropic vision via the same API path as llm.py's ApiBackend. Needs an
    ANTHROPIC_API_KEY; if absent, `get_vision_provider` degrades to skip-with-reason, it does NOT
    crash. (The CLI text backend can't take image input, so vision is API-only for now.)
  * "mock" — deterministic, dependency-free, for CI. Never calls a model.
  * register your own (another cloud LLM, a local VLM) via `register_vision_provider`.

GRACEFUL DEGRADATION (the enrichment failure model): a MISSING or MISCONFIGURED backend must
never abort a pipeline. `get_vision_provider(..., strict=False)` returns `None` plus a human
reason instead of raising, so a consumer can skip THIS image and keep going. `describe()` on a
live backend that then fails at call time raises `VisionError`, which consumers catch per-image.

SECURITY CONTRACT for consumers (stated here, wired later by #1 / the OCR ingestion path): a
caption or OCR'd text is UNTRUSTED — it originates from an image we did not author and may carry
a prompt-injection payload ("ignore your instructions and…"). Consumers MUST route the returned
text through the existing redaction + injection guards (redact.py + guardrails.py) BEFORE
embedding or feeding it to another model, exactly like any retrieved chunk. This seam does NOT
sanitise for you — it is a raw model output.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from .config import SETTINGS, Settings

# The two accepted image forms. Raw bytes (already-decoded) OR a Path on disk. Kept as a plain
# union so callers pass whichever they have without wrapping.
ImageInput = bytes | Path


class VisionError(RuntimeError):
    """Any backend/credential/transport/decode problem, normalised to one type so a consumer
    can `except VisionError` regardless of which backend is active (mirrors llm.LLMError)."""


@dataclass(frozen=True)
class VisionResult:
    """A caption/description plus the provenance a consumer records for AUDITABILITY.

    `provider` is the backend name that produced this (e.g. "claude", "mock") — written into the
    ingestion payload so an operator can later tell HOW a given chunk's text was derived. `model`
    is the concrete model id where the backend has one (None for mock/local). `meta` carries any
    backend-specific extras (token counts, etc.) without widening the core shape."""

    text: str
    provider: str
    model: str | None = None
    meta: dict = field(default_factory=dict)


@runtime_checkable
class VisionProvider(Protocol):
    """Structural contract (a Protocol, like llm.LLMBackend): any object with a `name` and a
    `describe()` IS a provider — no shared base class, so a local VLM or a cloud client can be
    adapted without inheriting from us. `embed_image` is OPTIONAL: image-to-image retrieval (#19)
    is future work, so a backend that doesn't support it raises VisionError from that method."""

    name: str

    def describe(self, image: ImageInput, *, prompt: str | None = None) -> VisionResult: ...

    def embed_image(self, image: ImageInput) -> list[float]: ...


# ---------------------------------------------------------------------------
# image bytes + media-type helpers (shared by backends)
# ---------------------------------------------------------------------------

# The magic-number → Anthropic media_type map. We SNIFF the bytes rather than trust a file
# extension, because a consumer may hand us a decoded blob with no name, and because an image
# from an untrusted corpus may be mislabelled. Only the formats the vision API accepts.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # WEBP is RIFF-container; refined below
)


def load_image_bytes(image: ImageInput) -> bytes:
    """Coerce an ImageInput to raw bytes. A Path is read from disk; bytes pass through. Raises
    VisionError (not a bare OSError) so the caller catches ONE error type for the whole seam."""
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    try:
        return Path(image).read_bytes()
    except OSError as e:
        raise VisionError(f"could not read image {image!r}: {e}") from e


def sniff_media_type(data: bytes) -> str:
    """Best-effort image media_type from magic bytes, for the Anthropic image block. Defaults to
    image/png when unknown (the API validates for real; this just fills the required field)."""
    for magic, mt in _MAGIC:
        if data.startswith(magic):
            if mt == "image/webp" and data[8:12] != b"WEBP":
                continue  # RIFF but not WEBP → keep looking / fall through
            return mt
    return "image/png"


# Default instruction for describe() when a consumer passes no prompt. Kept generic (this seam
# has no domain knowledge); #1 will pass a task-specific prompt.
DEFAULT_DESCRIBE_PROMPT = (
    "Describe this image factually and concisely for a search index: what it shows, any visible "
    "text, UI elements, and notable objects. Do not speculate beyond what is visible."
)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class ClaudeVisionProvider:
    """Anthropic vision via the official SDK — the DEFAULT. Same API path as llm.ApiBackend:
    reads ANTHROPIC_API_KEY from the environment, sends an image content block + a text prompt.

    The `anthropic` package is imported LAZILY in __init__ (not at module import) so importing
    rageval.vision never requires the SDK — a deployment that only uses the mock/tesseract path
    pays nothing. A missing key or package raises VisionError, which get_vision_provider turns
    into a clean skip-with-reason when strict=False."""

    name = "claude"

    def __init__(self, model: str):
        try:
            import anthropic  # lazy: only the claude backend needs it
        except ImportError as e:  # pragma: no cover - import guard
            raise VisionError(
                "the 'anthropic' package is not installed. Run: pip install anthropic"
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise VisionError(
                "ANTHROPIC_API_KEY is not set — the Claude vision backend needs an API key "
                "(the `claude` CLI cannot take image input)."
            )
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def describe(self, image: ImageInput, *, prompt: str | None = None) -> VisionResult:
        data = load_image_bytes(image)
        media_type = sniff_media_type(data)
        b64 = base64.standard_b64encode(data).decode("ascii")
        content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64},
            },
            {"type": "text", "text": prompt or DEFAULT_DESCRIBE_PROMPT},
        ]
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": content}],
            )
        except Exception as e:  # noqa: BLE001 - normalise to one error type
            raise VisionError(f"Anthropic vision call failed: {e}") from e
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return VisionResult(
            text="\n".join(parts).strip(),
            provider=self.name,
            model=self.model,
        )

    def embed_image(self, image: ImageInput) -> list[float]:
        # Anthropic exposes no image-embedding endpoint; image-to-image (#19) will use a dedicated
        # CLIP-style backend. Raise the seam's error type so a caller degrades consistently.
        raise VisionError(
            "the Claude vision backend does not support embed_image (image-to-image #19 is future "
            "work; register a CLIP-style backend for it)."
        )


class MockVisionProvider:
    """Deterministic, dependency-free vision for CI and offline dev. Produces a STABLE caption
    derived from the image bytes (a short digest) so a test can assert on it without a live model,
    and a fixed-dimension pseudo-embedding for embed_image. Never touches the network."""

    name = "mock"
    embed_dim = 8

    def describe(self, image: ImageInput, *, prompt: str | None = None) -> VisionResult:
        data = load_image_bytes(image)
        import hashlib

        digest = hashlib.sha256(data).hexdigest()[:12]
        return VisionResult(
            text=f"[mock vision] {len(data)} bytes, sha256:{digest}",
            provider=self.name,
            model=None,
            meta={"prompt": prompt or DEFAULT_DESCRIBE_PROMPT, "bytes": len(data)},
        )

    def embed_image(self, image: ImageInput) -> list[float]:
        data = load_image_bytes(image)
        import hashlib

        h = hashlib.sha256(data).digest()
        # Map the first `embed_dim` bytes to floats in [0, 1) — deterministic + stable.
        return [h[i] / 255.0 for i in range(self.embed_dim)]


# ---------------------------------------------------------------------------
# Registry + factory (mirrors llm.py's factory, generalised to named backends)
# ---------------------------------------------------------------------------

# Builders, not instances: a backend may need credentials/a model id at construction and may
# raise if they're missing, so we defer construction to get_vision_provider (which can turn that
# raise into a skip-with-reason). Keyed by the RAGEVAL_VISION_PROVIDER name.
VisionBuilder = Callable[[Settings], VisionProvider]

_VISION_REGISTRY: dict[str, VisionBuilder] = {
    "claude": lambda s: ClaudeVisionProvider(s.vision_model or s.model),
    "mock": lambda s: MockVisionProvider(),
}


def register_vision_provider(name: str, builder: VisionBuilder) -> None:
    """PUBLIC extension API: teach the engine a new vision backend (another cloud LLM, a local
    VLM) WITHOUT editing this module. `builder(settings) -> VisionProvider` is called lazily by
    get_vision_provider so construction errors become clean skips. Last-wins (idempotent)."""
    _VISION_REGISTRY[name.lower()] = builder


def available_vision_providers() -> list[str]:
    """Registered backend names (for /health and diagnostics)."""
    return sorted(_VISION_REGISTRY)


def get_vision_provider(
    settings: Settings = SETTINGS, *, strict: bool = False
) -> tuple[VisionProvider | None, str | None]:
    """Build the configured vision backend. Returns `(provider, reason)`:

      * success        → (provider, None)
      * degrade        → (None, "<why it's unavailable>")   when strict=False (DEFAULT)
      * strict failure → raises VisionError                 when strict=True

    GRACEFUL DEGRADATION is the default so a consumer can do:

        provider, reason = get_vision_provider(settings)
        if provider is None:
            skip_this_image(reason)      # e.g. "no ANTHROPIC_API_KEY"
        else:
            result = provider.describe(image)

    An UNKNOWN provider name is a configuration ERROR (not a degrade): it always raises, because
    silently ignoring a typo'd RAGEVAL_VISION_PROVIDER would hide a misconfiguration."""
    name = (settings.vision_provider or "claude").lower()
    builder = _VISION_REGISTRY.get(name)
    if builder is None:
        raise VisionError(
            f"unknown RAGEVAL_VISION_PROVIDER={name!r}; "
            f"available: {', '.join(available_vision_providers())}"
        )
    try:
        return builder(settings), None
    except VisionError as e:
        if strict:
            raise
        return None, str(e)


def vision_status(settings: Settings = SETTINGS) -> dict:
    """Diagnostics for /health: which vision provider is configured and whether it's usable."""
    provider, reason = get_vision_provider(settings, strict=False)
    return {
        "provider": settings.vision_provider,
        "available": provider is not None,
        "reason": reason,
        "registered": available_vision_providers(),
    }
