"""Born-digital PDF text extraction (#39) — pluggable, layout-aware, page-provenance-carrying.

WHY PDF is its own extractor (not just "read the bytes"): a PDF is not text — it's a page-oriented
graphics format whose text is a bag of positioned glyph runs. Pulling READABLE text back out
(reading order preserved, pages delimited) is format-specific work, and WHICH library does it is a
choice we want to keep swappable (pypdf today; pdfplumber/pymupdf are drop-ins for richer layout).
So extraction lives behind a tiny interface:

    class PdfExtractor:
        def extract(self, path) -> PdfExtraction   # pages[] + a scanned/no-text-layer flag

THE ONE HARD DISTINCTION this module draws — born-digital vs scanned:
  A *born-digital* PDF has a real text layer (the glyphs carry Unicode) → we extract it. A
  *scanned* PDF is page IMAGES with NO text layer → pypdf returns (near-)empty strings for every
  page. Embedding that yields ZERO useful chunks silently — a corpus blind spot that looks like
  success. So we DETECT the no-text-layer case (`extraction.scanned is True`, `has_text_layer` is
  False) and let the caller SURFACE it as a coverage warning. OCR (turning the page images into
  text) is a DIFFERENT capability — the vision/OCR provider seam — and explicitly out of scope
  here. Detecting-and-flagging is the honest boundary: we never emit empty chunks pretending a
  scan was ingested.

PAGE PROVENANCE: each page's text is kept as a `PdfPage(number, text)` and the joined `text`
inserts a lightweight `[page N]` marker before each page's content. The engine's char-window
chunker is format-agnostic, so this in-band marker is the simplest robust way to carry page
provenance INTO the chunk text (and thus into the retrieved citation) without threading a parallel
page-offset map through every downstream stage. It is deliberately unobtrusive and easy to grep.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path

# A page whose extracted text is at/below this many non-whitespace chars counts as "no text on
# this page" (a scanned image page, or a near-blank divider). Small but non-zero so a stray ligature
# A page needs at LEAST this many non-whitespace chars to count as carrying real text (so a stray
# ligature or lone page number doesn't make a truly image-only page look born-digital).
_MIN_PAGE_TEXT_CHARS = 8

# Hard page cap (MAJOR-3, DoS): never walk more than this many pages from one file. A file beyond it
# is truncated with the truncation surfaced in meta, so a pathological PDF can't pin the CPU.
_DEFAULT_MAX_PAGES = 2000


@dataclass(frozen=True)
class PdfPage:
    """One extracted page: its 1-based `number` and the reading-order `text` (may be empty)."""

    number: int
    text: str

    @property
    def has_text(self) -> bool:
        # >= threshold: a page with exactly _MIN_PAGE_TEXT_CHARS real chars DOES carry text (the old
        # strict > was an off-by-one that dropped a minimal-but-real page).
        return len(self.text.strip()) >= _MIN_PAGE_TEXT_CHARS


@dataclass(frozen=True)
class PdfExtraction:
    """The result of extracting one PDF: its pages + the text-layer verdict.

    THE KEY DISTINCTION (MAJOR-2): "few text pages" is NOT "no text layer". We NEVER discard a doc
    that has ANY extractable text — we keep the text pages we got and, if some pages were image-only,
    flag THOSE pages for OCR without dropping the real text. Only a doc with ZERO text pages is
    `scanned` (fully image/no-text-layer → the whole doc routes to OCR, and there was no real text to
    lose).

    Fields:
      pages       — one PdfPage per page, in document order (empty text for image-only pages).
      scanned     — True ONLY when NO page carries a text layer (fully scanned/image PDF). The caller
                    flags it (coverage warning) and embeds nothing (there is nothing to embed).
      needs_ocr_pages — page numbers that are image-only in a doc that IS partly text-bearing: a
                    PARTIAL doc whose real text is still extracted + embedded, with these pages
                    flagged as an OCR follow-up rather than silently lost.
      backend     — which extractor produced this (auditability; e.g. "pypdf").
    """

    pages: tuple[PdfPage, ...]
    scanned: bool
    needs_ocr_pages: tuple[int, ...] = ()
    backend: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def has_text_layer(self) -> bool:
        """True if ANY page carries extractable text (a partial doc still has a text layer)."""
        return self.text_page_count > 0

    @property
    def partial(self) -> bool:
        """True for a doc that has BOTH text pages and image-only pages (some OCR follow-up needed,
        but real text was extracted and must not be dropped)."""
        return bool(self.needs_ocr_pages) and self.has_text_layer

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text_page_count(self) -> int:
        return sum(1 for p in self.pages if p.has_text)

    def text(self, *, page_markers: bool = True) -> str:
        """Join the pages into one reading-order string.

        With `page_markers` (default) each page's content is preceded by a `[page N]` line, so page
        provenance travels INTO the chunk text and out to the citation. Empty (image-only) pages
        contribute no marker (there is nothing to cite on them)."""
        parts: list[str] = []
        for page in self.pages:
            body = page.text.strip()
            if not body:
                continue
            parts.append(f"[page {page.number}]\n{body}" if page_markers else body)
        return "\n\n".join(parts)


class PdfExtractor(abc.ABC):
    """The swappable PDF-extraction interface. A backend implements `_read_pages`; the base class
    owns the corpus-neutral text-layer detection so every backend classifies consistently."""

    #: short, stable backend id recorded on the extraction for auditability.
    name: str = "base"

    #: hard page cap (DoS guard); overridable per subclass/instance.
    max_pages: int = _DEFAULT_MAX_PAGES

    @abc.abstractmethod
    def _read_pages(self, path: Path, *, max_pages: int) -> tuple[list[PdfPage], bool]:
        """Return (pages, truncated): one PdfPage per page (image-only pages yield empty text), and
        whether the file had MORE pages than max_pages (truncated). Backend-specific."""
        raise NotImplementedError

    def extract(self, path: Path) -> PdfExtraction:
        """Extract `path` → a PdfExtraction.

        Detection (MAJOR-2): a doc with NO text-bearing page is `scanned` (fully image → route to
        OCR; nothing real to lose). A doc with SOME text pages KEEPS its extracted text and flags any
        image-only pages via `needs_ocr_pages` — its real text is never discarded just because other
        pages need OCR."""
        pages, truncated = self._read_pages(Path(path), max_pages=self.max_pages)
        n = len(pages)
        text_pages = sum(1 for p in pages if p.has_text)
        scanned = text_pages == 0            # ONLY a fully text-less doc is scanned
        # In a PARTIAL doc (some text pages), the image-only pages are OCR follow-ups — but we DO NOT
        # discard the text we extracted. In a fully-scanned doc, every page is trivially "needs OCR",
        # captured by `scanned` instead (no per-page list needed there).
        needs_ocr = tuple(p.number for p in pages if not p.has_text) if text_pages else ()
        return PdfExtraction(
            pages=tuple(pages),
            scanned=scanned,
            needs_ocr_pages=needs_ocr,
            backend=self.name,
            meta={"page_count": n, "text_page_count": text_pages, "truncated": truncated},
        )


class PypdfExtractor(PdfExtractor):
    """Default backend: pypdf (pure-Python, dependency-light, portable in CI).

    pypdf's `page.extract_text()` walks the page's text operators in content-stream order, which
    preserves reading order well for born-digital documents. It returns "" for an image-only page —
    exactly the signal the base class turns into the scanned verdict. Swapping to pdfplumber/pymupdf
    for richer table geometry is a one-class change (implement `_read_pages`), by design."""

    name = "pypdf"

    def _read_pages(self, path: Path, *, max_pages: int) -> tuple[list[PdfPage], bool]:
        from pypdf import PdfReader  # lazy: only importable-cost when a PDF is actually read
        from pypdf.errors import PdfReadError

        try:
            reader = PdfReader(str(path))
        except (PdfReadError, OSError, ValueError):
            # A corrupt/unreadable PDF yields NO pages → the base class marks it scanned (flagged),
            # never crashing discovery. (Same tolerance as the .docx path.)
            return [], False
        pages: list[PdfPage] = []
        truncated = False
        for i, page in enumerate(reader.pages, start=1):
            if i > max_pages:  # DoS page cap (MAJOR-3): stop and flag truncation.
                truncated = True
                break
            try:
                raw = page.extract_text() or ""
            except Exception:  # noqa: BLE001 — one bad page never kills the whole document
                raw = ""
            pages.append(PdfPage(number=i, text=_normalize_page_text(raw)))
        return pages, truncated


def _normalize_page_text(raw: str) -> str:
    """Tidy a page's extracted text: normalise line endings and collapse the runs of blank lines
    pypdf often emits, WITHOUT reflowing paragraphs (we keep line breaks so layout-ish structure —
    label/value lines, list items — survives into chunking)."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing spaces per line; drop 3+ consecutive blank lines down to one.
    lines = [ln.rstrip() for ln in text.split("\n")]
    out: list[str] = []
    blanks = 0
    for ln in lines:
        if ln.strip():
            blanks = 0
            out.append(ln)
        else:
            blanks += 1
            if blanks <= 1:
                out.append("")
    return "\n".join(out).strip()


# The default extractor instance. A deployment swaps this by passing its own PdfExtractor to
# extract_pdf(); kept as a module singleton so the common path constructs nothing per call.
_DEFAULT_EXTRACTOR: PdfExtractor = PypdfExtractor()


def get_pdf_extractor() -> PdfExtractor:
    """Return the default PDF extractor (pypdf). A single swap point for a future config knob."""
    return _DEFAULT_EXTRACTOR


def extract_pdf(path: Path, *, extractor: PdfExtractor | None = None) -> PdfExtraction:
    """Extract a PDF with the given (or default) backend. The one call the adapter layer uses."""
    return (extractor or _DEFAULT_EXTRACTOR).extract(Path(path))
