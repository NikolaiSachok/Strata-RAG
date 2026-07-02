"""Tests for the pluggable VISION provider seam (vision.py).

These are CI-SELF-CONTAINED: the mock backend needs no model, no network, no optional deps. They
prove the three properties the seam promises: (1) the mock works deterministically; (2) the
backend is swappable by config (RAGEVAL_VISION_PROVIDER); (3) graceful degradation when the
default (Claude) backend has no credentials — a skip-with-reason, never a crash.

The real Claude path needs an ANTHROPIC_API_KEY and is exercised only as a LOCAL smoke (reported
in the PR, not run in CI).
"""

from __future__ import annotations

import dataclasses

import pytest

from rageval.config import SETTINGS
from rageval.vision import (
    MockVisionProvider,
    VisionError,
    VisionResult,
    available_vision_providers,
    get_vision_provider,
    register_vision_provider,
    sniff_media_type,
    vision_status,
)

# A minimal valid PNG header + a few bytes — enough to exercise byte-loading + sniffing without
# any image library. (Not a decodable image; the mock never decodes, it just digests bytes.)
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _settings(**over):
    return dataclasses.replace(SETTINGS, **over)


def test_mock_provider_is_deterministic():
    settings = _settings(vision_provider="mock")
    provider, reason = get_vision_provider(settings)
    assert reason is None
    assert provider is not None and provider.name == "mock"

    r1 = provider.describe(_PNG_BYTES)
    r2 = provider.describe(_PNG_BYTES)
    assert isinstance(r1, VisionResult)
    assert r1.text == r2.text  # deterministic
    assert r1.provider == "mock"  # provenance recorded for auditability
    assert str(len(_PNG_BYTES)) in r1.text


def test_mock_embed_image_is_stable_and_dimensioned():
    provider = MockVisionProvider()
    v1 = provider.embed_image(_PNG_BYTES)
    v2 = provider.embed_image(_PNG_BYTES)
    assert v1 == v2
    assert len(v1) == MockVisionProvider.embed_dim
    assert all(0.0 <= x < 1.0 for x in v1)


def test_backend_swappable_by_config():
    """RAGEVAL_VISION_PROVIDER (here via Settings.vision_provider) selects the backend."""
    mock_settings = _settings(vision_provider="mock")
    provider, _ = get_vision_provider(mock_settings)
    assert provider is not None and provider.name == "mock"

    # Default is claude (proves the default wiring, independent of whether a key is present).
    assert SETTINGS.vision_provider == "claude"


def test_graceful_degradation_when_default_backend_unconfigured(monkeypatch):
    """The DEFAULT backend (claude) with NO API key must degrade to skip-with-reason, not crash."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = _settings(vision_provider="claude")
    provider, reason = get_vision_provider(settings, strict=False)
    assert provider is None
    assert reason and "ANTHROPIC_API_KEY" in reason


def test_strict_mode_raises_when_unconfigured(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = _settings(vision_provider="claude")
    with pytest.raises(VisionError):
        get_vision_provider(settings, strict=True)


def test_unknown_provider_is_a_config_error_not_a_degrade():
    """A typo'd provider name is a misconfiguration — it raises even in non-strict mode, so it
    can't be silently swallowed as 'no backend'."""
    settings = _settings(vision_provider="nope")
    with pytest.raises(VisionError):
        get_vision_provider(settings, strict=False)


def test_register_custom_provider_roundtrips():
    class DummyVision:
        name = "dummy"

        def describe(self, image, *, prompt=None):
            return VisionResult(text="dummy", provider=self.name)

        def embed_image(self, image):
            return [0.0]

    try:
        register_vision_provider("dummy", lambda s: DummyVision())
        assert "dummy" in available_vision_providers()
        provider, reason = get_vision_provider(_settings(vision_provider="dummy"))
        assert reason is None
        assert provider.describe(_PNG_BYTES).text == "dummy"
    finally:
        from rageval import vision

        vision._VISION_REGISTRY.pop("dummy", None)


def test_sniff_media_type():
    assert sniff_media_type(_PNG_BYTES) == "image/png"
    assert sniff_media_type(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    assert sniff_media_type(b"GIF89a...") == "image/gif"
    assert sniff_media_type(b"unknown") == "image/png"  # default


def test_vision_status_reports_mock_available():
    status = vision_status(_settings(vision_provider="mock"))
    assert status["provider"] == "mock"
    assert status["available"] is True
    assert "mock" in status["registered"]
