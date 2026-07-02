"""Tests for the pluggable OCR provider seam (ocr.py).

CI-SELF-CONTAINED via the mock backend (no binary, no model, no optional deps). They prove the
same three properties as the vision seam: mock works deterministically; the backend is swappable
by config (RAGEVAL_OCR_PROVIDER); graceful degradation when a backend is absent/misconfigured.

The real-Tesseract test (`test_tesseract_reads_synthetic_png`) runs ONLY when `pytesseract` +
`Pillow` + the `tesseract` binary are all present; otherwise it SKIPs (mirroring the presidio
optional-extra skip pattern). It renders text to a PNG at runtime — the fixture is synthetic and
self-contained, nothing binary is committed.
"""

from __future__ import annotations

import builtins
import dataclasses
import importlib.util
import shutil

import pytest

from rageval.config import SETTINGS
from rageval.ocr import (
    MockOcrProvider,
    OcrError,
    OcrRegion,
    OcrResult,
    available_ocr_providers,
    get_ocr_provider,
    ocr_status,
    register_ocr_provider,
)

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _settings(**over):
    return dataclasses.replace(SETTINGS, **over)


# True only when the WHOLE local-OCR stack is present: the wrapper, PIL (to render + read), and
# the system binary. The Tesseract-present test gates on this; everything else is dependency-free.
_HAS_TESSERACT = (
    importlib.util.find_spec("pytesseract") is not None
    and importlib.util.find_spec("PIL") is not None
    and shutil.which("tesseract") is not None
)


def test_mock_provider_is_deterministic():
    settings = _settings(ocr_provider="mock")
    provider, reason = get_ocr_provider(settings)
    assert reason is None
    assert provider is not None and provider.name == "mock"

    r1 = provider.ocr(_PNG_BYTES)
    r2 = provider.ocr(_PNG_BYTES)
    assert isinstance(r1, OcrResult)
    assert r1.text == r2.text
    assert r1.provider == "mock"
    assert r1.confidence == 99.0
    assert r1.regions and isinstance(r1.regions[0], OcrRegion)


def test_backend_swappable_by_config():
    provider, _ = get_ocr_provider(_settings(ocr_provider="mock"))
    assert provider is not None and provider.name == "mock"
    # Default is tesseract (the wiring; independent of whether the binary is installed).
    assert SETTINGS.ocr_provider == "tesseract"


def test_tesseract_degrades_when_stack_absent(monkeypatch):
    """When pytesseract/PIL/the binary are missing, the tesseract backend must degrade to a
    skip-with-reason — never crash. We simulate 'binary absent' by neutralising shutil.which."""
    if not _HAS_TESSERACT:
        # The stack is genuinely absent → degradation is real; just assert it.
        provider, reason = get_ocr_provider(_settings(ocr_provider="tesseract"))
        assert provider is None
        assert reason  # a human-readable reason string
        return
    # The stack IS present → force the binary to look absent to exercise the degrade path.
    monkeypatch.setattr("rageval.ocr.shutil.which", lambda _name: None)
    provider, reason = get_ocr_provider(_settings(ocr_provider="tesseract"))
    assert provider is None
    assert reason and "tesseract" in reason.lower()


def _block_imports(monkeypatch, *blocked: str) -> None:
    """Make an import raise ImportError when its module name (or a parent of it) is in `blocked`,
    deterministically, no matter what's installed. We wrap builtins.__import__ so the provider's
    LAZY imports fail as if the package were absent — this lets us test the degrade paths on ANY
    machine (a box WITH the full OCR stack still exercises the missing-package branches).

    Matching is on the DOTTED PREFIX: blocking "PIL" also blocks "PIL.Image"; blocking exactly
    "PIL.Image" leaves a bare "import PIL" alone. That distinction matters here — `import
    pytesseract` pulls in PIL internally, so to isolate the provider's OWN `import PIL.Image`
    (line 121) we block just that submodule and leave pytesseract's own import working."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        parts = name.split(".")
        for i in range(len(parts)):
            if ".".join(parts[: i + 1]) in blocked:
                raise ImportError(f"blocked import of {name!r} for the test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_tesseract_degrades_when_pytesseract_missing(monkeypatch):
    """The pytesseract-missing degrade path — forced via a blocked import so it runs on any
    machine, not only where pytesseract is absent."""
    _block_imports(monkeypatch, "pytesseract")
    provider, reason = get_ocr_provider(_settings(ocr_provider="tesseract"))
    assert provider is None
    assert reason and "pytesseract" in reason.lower()


@pytest.mark.skipif(
    importlib.util.find_spec("pytesseract") is None,
    reason="reaching the PIL check requires pytesseract to import first",
)
def test_tesseract_degrades_when_pillow_missing(monkeypatch):
    """The Pillow-missing degrade path — forced by blocking ONLY the `PIL.Image` submodule the
    provider imports at line 121, so `import pytesseract` (which pulls PIL in internally, already
    cached) still succeeds and we reach the PIL branch specifically. Skipped where pytesseract is
    absent (you can't reach the PIL check without it)."""
    import pytesseract  # noqa: F401 - ensure it's cached so its own import doesn't re-run PIL

    _block_imports(monkeypatch, "PIL.Image")
    provider, reason = get_ocr_provider(_settings(ocr_provider="tesseract"))
    assert provider is None
    assert reason and ("pillow" in reason.lower() or "pil" in reason.lower())


def test_llm_ocr_degrades_when_vision_unconfigured(monkeypatch):
    """LLM-as-OCR reuses the vision seam; with no vision backend it degrades, not crashes."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = _settings(ocr_provider="llm", vision_provider="claude")
    provider, reason = get_ocr_provider(settings, strict=False)
    assert provider is None
    assert reason and "vision" in reason.lower()


def test_llm_ocr_uses_mock_vision_backend():
    """With the vision provider set to mock, LLM-as-OCR transcribes via it — proving the reuse of
    the vision seam and that the provider name records BOTH layers for auditability."""
    settings = _settings(ocr_provider="llm", vision_provider="mock")
    provider, reason = get_ocr_provider(settings)
    assert reason is None and provider is not None
    result = provider.ocr(_PNG_BYTES)
    assert result.text  # the mock vision caption becomes the "OCR" text
    assert result.provider == "llm:mock"  # both layers recorded
    assert result.confidence is None       # a chat model exposes no calibrated confidence


def test_unknown_provider_is_a_config_error():
    with pytest.raises(OcrError):
        get_ocr_provider(_settings(ocr_provider="nope"), strict=False)


def test_strict_mode_raises_when_unconfigured(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = _settings(ocr_provider="llm", vision_provider="claude")
    with pytest.raises(OcrError):
        get_ocr_provider(settings, strict=True)


def test_register_custom_provider_roundtrips():
    class DummyOcr:
        name = "dummy"

        def ocr(self, image):
            return OcrResult(text="dummy", provider=self.name)

    try:
        register_ocr_provider("dummy", lambda s: DummyOcr())
        assert "dummy" in available_ocr_providers()
        provider, reason = get_ocr_provider(_settings(ocr_provider="dummy"))
        assert reason is None
        assert provider.ocr(_PNG_BYTES).text == "dummy"
    finally:
        from rageval import ocr

        ocr._OCR_REGISTRY.pop("dummy", None)


def test_ocr_status_reports_mock_available():
    status = ocr_status(_settings(ocr_provider="mock"))
    assert status["provider"] == "mock"
    assert status["available"] is True
    assert "mock" in status["registered"]


@pytest.mark.skipif(not _HAS_TESSERACT, reason="pytesseract + Pillow + tesseract binary required")
def test_tesseract_reads_synthetic_png(tmp_path):
    """Real-Tesseract path: render known text to a PNG, OCR it, assert the text round-trips and
    per-word confidences/regions are populated. Runs only when the local OCR stack is present."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (320, 80), color="white")
    draw = ImageDraw.Draw(img)
    # The default bitmap font is small but legible to Tesseract; keep the text simple + uppercase.
    draw.text((10, 30), "STRATA RAG OCR", fill="black")
    png = tmp_path / "synthetic.png"
    img.save(png)

    provider, reason = get_ocr_provider(_settings(ocr_provider="tesseract"))
    assert provider is not None, reason
    result = provider.ocr(png)
    text = result.text.upper().replace(" ", "")
    # Tesseract on a tiny bitmap font isn't perfect; assert on a robust substring, not exact match.
    assert "STRATA" in text or "RAG" in text or "OCR" in text
    assert result.provider == "tesseract"
    if result.regions:
        assert result.confidence is not None  # mean confidence when any word scored
