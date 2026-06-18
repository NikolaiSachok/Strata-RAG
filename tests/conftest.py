"""Pytest session setup shared by the whole suite.

THE PROBLEM this solves: a corpus shape may include BINARY/BUILD placeholder files (build
archives, keystores, generated source) that the adapters reference by FILENAME but never read.
There's no reason to commit binaries to git, so they're `.gitignore`d — which means on a fresh
clone they're absent and the directories that held them vanish too. A discovery test that
expected them would then fail through no fault of the code.

THE FIX: an autouse, session-scoped fixture that materialises any missing placeholders from the
committed manifest (`rageval.sample_placeholders`) before any test runs. So `pytest` passes on a
clean checkout with zero committed binaries. The fixture only CREATES missing files (idempotent,
trivial content), so it's safe to run repeatedly and never clobbers a developer's edits. For the
bundled northwind/atlas sample corpus the manifest is EMPTY, so this is a no-op today.
"""

from __future__ import annotations

import pytest

from rageval.sample_placeholders import materialize


@pytest.fixture(scope="session", autouse=True)
def _ensure_sample_placeholders():
    """Recreate any gitignored sample-corpus placeholder files before the suite runs."""
    materialize()
    yield
