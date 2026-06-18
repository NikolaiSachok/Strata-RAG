"""Source adapters — the extensibility seam of the whole engine.

THE BIG IDEA (read this first): the engine must NOT hardcode where documents live or
how a particular corpus is laid out. A real corpus is heterogeneous and legacy-shaped:
one source-set keeps descriptions in `docs/*.md`, another buries them in `.docx` files
under oddly-named back-end folders. If the ingest pipeline knew those details, adding a
new corpus would mean editing the pipeline.

Instead we put ALL corpus-specific knowledge behind one tiny interface:

    SourceAdapter.discover() -> Iterable[SourceDoc]

Each adapter knows how to walk ONE family of projects and yield `SourceDoc`s. The rest
of the pipeline (classify → chunk → embed → index → eval) only ever sees `SourceDoc`s
and never touches the filesystem layout. So onboarding a new corpus = write a new
adapter; nothing downstream changes. This is the Strategy pattern, and it's exactly the
"source-agnostic" property an enterprise document-intelligence engine needs.
"""

from __future__ import annotations

from ..roster import register_family
from .base import SourceAdapter, SourceDoc
from .registry import discover_all, get_adapters, register_adapter

__all__ = [
    "SourceAdapter",
    "SourceDoc",
    "discover_all",
    "get_adapters",
    "register_adapter",
    "register_family",
]

# Load optional plugins (in-package bootstrap + $RAGEVAL_PLUGINS_DIR) LAST — only now that every
# facade name above is bound. An external plugin follows the documented ergonomic import
# `from rageval.sources import register_adapter, register_family`; triggering the loader earlier
# (e.g. at registry-import time, mid-`__init__`) would import that plugin against a partially
# initialised `rageval.sources` and crash with a circular import. The loader is guarded to run
# exactly once per process.
from .registry import load_optional_plugins as _load_optional_plugins

_load_optional_plugins()
