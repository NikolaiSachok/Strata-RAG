"""Tests for born-digital PDF extraction (#39) — self-contained (no vendored corpus).

Fixtures are GENERATED in-test with reportlab (a born-digital PDF with a real text layer) and with
pypdf (a no-text-layer / "scanned" PDF: pages with NO text operators). So CI needs no committed
binaries and no external corpus — the whole acceptance path is exercised from synthetic inputs.

What these lock down:
  * a born-digital PDF is extracted with PAGE PROVENANCE ([page N] markers, right page numbers);
  * a no-text-layer PDF is DETECTED (extraction.scanned) and never emits fake text;
  * the extractor is PLUGGABLE (a custom backend is honoured);
  * a corrupt PDF degrades to scanned rather than crashing.
"""

from __future__ import annotations

import pytest

from rageval.extract.pdf import (
    PdfExtractor,
    PdfPage,
    PypdfExtractor,
    extract_pdf,
    get_pdf_extractor,
)

reportlab = pytest.importorskip("reportlab")  # dev extra; skip cleanly if absent


def _born_digital_pdf(path, pages: list[str]) -> None:
    """Write a born-digital PDF (real text layer) with one drawn string per page."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    for text in pages:
        y = 720
        for line in text.split("\n"):
            c.drawString(72, y, line)
            y -= 18
        c.showPage()
    c.save()


def _no_text_layer_pdf(path, n_pages: int = 2) -> None:
    """Write a PDF whose pages carry NO text operators (the born-digital analogue of a scan).

    We add empty pages via pypdf — extract_text() returns "" for each, exactly the signal a real
    scanned/image PDF gives. (Rendering actual raster images would need Pillow + a much larger
    fixture; an empty-text-layer page is the equivalent detection case and keeps CI tiny.)"""
    from pypdf import PdfWriter

    w = PdfWriter()
    for _ in range(n_pages):
        w.add_blank_page(width=612, height=792)
    with open(path, "wb") as fh:
        w.write(fh)


# --- born-digital: extracted with page provenance -------------------------------------------

def test_born_digital_pdf_extracted_with_page_provenance(tmp_path):
    pdf = tmp_path / "policy.pdf"
    _born_digital_pdf(pdf, [
        "Policy number HH-0000001\nLine of business Household\nAnnual premium 960",
        "Deductible 250\nLimit 500000\nEndorsements none",
    ])
    ex = extract_pdf(pdf)

    assert not ex.scanned and ex.has_text_layer
    assert ex.page_count == 2 and ex.text_page_count == 2
    assert ex.backend == "pypdf"

    # First-class page provenance: both page markers present, content under the right page.
    text = ex.text()
    assert "[page 1]" in text and "[page 2]" in text
    assert "Policy number HH-0000001" in text
    assert "Endorsements none" in text
    # Page 1 content appears before page 2's marker (reading order preserved).
    assert text.index("Policy number") < text.index("[page 2]")
    # Per-page access carries the right number.
    assert [p.number for p in ex.pages] == [1, 2]
    assert ex.pages[0].has_text


def test_text_without_markers_option(tmp_path):
    pdf = tmp_path / "a.pdf"
    _born_digital_pdf(pdf, ["Hello world this is a born digital page with real text."])
    ex = extract_pdf(pdf)
    assert "[page 1]" not in ex.text(page_markers=False)
    assert "Hello world" in ex.text(page_markers=False)


# --- no-text-layer detection (the scanned case) ---------------------------------------------

def test_no_text_layer_pdf_detected_not_silently_empty(tmp_path):
    pdf = tmp_path / "scanned.pdf"
    _no_text_layer_pdf(pdf, n_pages=3)
    ex = extract_pdf(pdf)

    assert ex.scanned is True and ex.has_text_layer is False
    assert ex.page_count == 3
    assert ex.text_page_count == 0
    # It never fabricates text — the joined text is empty, and the caller flags it (never embeds it).
    assert ex.text().strip() == ""


def test_sparse_but_real_pdf_keeps_its_text(tmp_path):
    """MAJOR-2: a doc with mostly image pages but a REAL text page is NOT scanned — its text is kept
    and the image pages are flagged for OCR, never discarded. "Few text pages" != "no text layer"."""
    from pypdf import PdfReader, PdfWriter

    born = tmp_path / "one.pdf"
    _born_digital_pdf(born, ["just one real page of text here that must survive"])
    w = PdfWriter()
    w.append(PdfReader(str(born)))
    for _ in range(9):
        w.add_blank_page(width=612, height=792)  # 1 text page + 9 image pages
    mixed = tmp_path / "mixed.pdf"
    with open(mixed, "wb") as fh:
        w.write(fh)

    ex = extract_pdf(mixed)
    assert ex.page_count == 10 and ex.text_page_count == 1
    assert ex.scanned is False                 # has a text layer → NOT scanned
    assert ex.has_text_layer is True
    assert ex.partial is True                  # some pages still need OCR
    assert "must survive" in ex.text()         # the real text is NOT dropped
    # The 9 image pages are flagged for an OCR follow-up (not silently lost).
    assert ex.needs_ocr_pages == tuple(range(2, 11))


# --- pluggability + robustness --------------------------------------------------------------

def test_extractor_is_pluggable(tmp_path):
    """A custom backend is honoured — the interface is swappable (pypdf → pdfplumber/pymupdf)."""

    class StubExtractor(PdfExtractor):
        name = "stub"

        def _read_pages(self, path, *, max_pages):
            return ([PdfPage(1, "stub page one has plenty of text"),
                     PdfPage(2, "stub page two also has text content")], False)

    ex = extract_pdf(tmp_path / "ignored.pdf", extractor=StubExtractor())
    assert ex.backend == "stub" and not ex.scanned
    assert "stub page one" in ex.text()


def test_pdf_page_cap_truncates_and_flags(tmp_path):
    """MAJOR-3: an over-cap PDF is truncated at max_pages with truncation recorded (DoS guard)."""

    class ManyPages(PdfExtractor):
        name = "many"
        max_pages = 3

        def _read_pages(self, path, *, max_pages):
            pages, truncated = [], False
            for i in range(1, 11):
                if i > max_pages:
                    truncated = True
                    break
                pages.append(PdfPage(i, f"page {i} content text here"))
            return pages, truncated

    ex = extract_pdf(tmp_path / "big.pdf", extractor=ManyPages())
    assert ex.page_count == 3 and ex.meta["truncated"] is True


def test_corrupt_pdf_degrades_to_scanned(tmp_path):
    bad = tmp_path / "corrupt.pdf"
    bad.write_bytes(b"%PDF-1.4 this is not a real pdf body")
    ex = extract_pdf(bad)
    # No pages recovered → scanned verdict, no crash.
    assert ex.scanned is True and ex.page_count == 0


def test_default_extractor_is_pypdf():
    assert isinstance(get_pdf_extractor(), PypdfExtractor)
