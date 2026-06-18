"""Manifest of the synthetic sample corpus's BINARY/BUILD placeholder files.

WHY this exists. Some corpus shapes carry binary/build artifacts (build archives, keystores,
generated source) that the adapters reference by FILENAME but never read the bytes of. There's
no reason to commit binaries to git, so they're `.gitignore`d — which means on a FRESH CLONE
they're absent, and because git doesn't track empty directories, the directories that hold them
vanish too.

THE FIX (reproducibility without committing binaries): keep this committed MANIFEST of the exact
placeholder paths, and materialise the missing ones on demand —
  * automatically before the test suite (an autouse fixture in tests/conftest.py), and
  * via `python -m rageval.make_sample` for the demo on a fresh clone.

The bundled `northwind`/`atlas` sample families ship entirely as committed text fixtures
(.md/.docx/.txt), so the manifest is EMPTY by default — there are no binaries to recreate.
It exists as the reproducibility seam a custom corpus with binary fixtures would populate
(paths relative to `data/sample/`, content "" for a 0-byte placeholder).
"""

from __future__ import annotations

from pathlib import Path

from .config import SAMPLE_CORPUS_DIR

# (relative path under data/sample/, content). 0-byte placeholders use "".
# Empty for the bundled sample corpus — northwind/atlas are all committed text fixtures.
PLACEHOLDERS: dict[str, str] = {}


def missing(base_dir: Path = SAMPLE_CORPUS_DIR) -> list[Path]:
    """Return the absolute paths of placeholder files that don't yet exist on disk."""
    return [base_dir / rel for rel in PLACEHOLDERS if not (base_dir / rel).exists()]


def materialize(base_dir: Path = SAMPLE_CORPUS_DIR) -> list[Path]:
    """Create any missing placeholder files (and their parent dirs). Idempotent: never
    overwrites an existing file. Returns the list of paths it created."""
    created: list[Path] = []
    for rel, content in PLACEHOLDERS.items():
        path = base_dir / rel
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created.append(path)
    return created
