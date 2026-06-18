"""`python -m rageval.make_sample` — materialise the sample corpus's binary/build placeholders.

Some corpus shapes carry binary-ish placeholder files (build archives, keystores, generated
source) that are `.gitignore`d (no reason to commit binaries) and therefore absent on a fresh
clone. Run this once to recreate them from the committed manifest (`sample_placeholders.py`):

    python -m rageval.make_sample
    python -m rageval.ingest --dry-run

It's idempotent (never overwrites existing files) and creates only fictional 0-byte/stub
content. For the bundled northwind/atlas sample corpus the manifest is EMPTY (all fixtures are
committed text), so this is a no-op; it's the reproducibility seam a custom binary-bearing
corpus would use.
"""

from __future__ import annotations

from .sample_placeholders import PLACEHOLDERS, materialize


def main() -> None:
    created = materialize()
    if created:
        print(f"Created {len(created)} placeholder file(s):")
        for p in created:
            print(f"  + {p}")
    else:
        print(f"All {len(PLACEHOLDERS)} sample placeholders already present — nothing to do.")


if __name__ == "__main__":
    main()
