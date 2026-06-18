"""Adapter registry — maps a corpus root to the right set of adapters, with an OPEN-CORE
extension seam.

WHY a registry: the engine takes ONE `corpus_root` and needs to know which adapters apply.
The folder under `corpus_root` selects the adapter: for the shipped sample corpus,
`data/sample/` contains source-set subfolders (`northwind/`, `atlas/`), each handled by its
own adapter. For a custom corpus you point `RAGENGINE_CORPUS_ROOT` at the parent of those
source-set folders.

EXTENSION SEAM (the important design choice here):
  * The CORE registers the bundled SAMPLE adapters — `northwind` + `atlas`.
  * Anyone extends the engine via the public `register_adapter(folder, cls)` API — NO edit to
    this module is needed to add a corpus.
  * An OPTIONAL bootstrap module, `sources/_private_plugins.py`, lets a deployment register its
    own corpus-specific adapters out-of-tree: this registry tries to import it at startup
    (try/except ImportError → silently skip). When the module is absent (the default), the
    registry holds only northwind/atlas, fully working on the sample, with zero edits here.

    (Deliberately NOT setuptools entry-points — too heavy for an in-repo plugin; the
    try-import-an-optional-module pattern is exactly the right weight here.)

This is still the only place that knows the mapping {folder name → adapter class}. Add a new
corpus = write a SourceAdapter subclass + one `register_adapter(...)` call (in core for a
bundled adapter, or in the optional bootstrap for an out-of-tree one). Nothing else changes.
"""

from __future__ import annotations

from pathlib import Path

from .atlas import AtlasAdapter
from .base import SourceAdapter, SourceDoc
from .northwind import NorthwindAdapter

# folder-name → adapter class. The folder under corpus_root selects the adapter. The CORE
# registers the bundled synthetic SAMPLE adapters; custom adapters add themselves
# via register_adapter() from the optional bootstrap (see _load_private_plugins).
ADAPTER_BY_FOLDER: dict[str, type[SourceAdapter]] = {
    "northwind": NorthwindAdapter,
    "atlas": AtlasAdapter,
}


def register_adapter(folder: str, adapter_cls: type[SourceAdapter]) -> None:
    """Register `adapter_cls` to handle the source-set folder named `folder`.

    This is the PUBLIC extension API. To teach the engine a new corpus, write a
    `SourceAdapter` subclass and call `register_adapter("<folder-name>", YourAdapter)` —
    no edit to the registry's core mapping is needed. The `folder` key matches the
    immediate subdirectory of `corpus_root` the adapter is responsible for (the same key
    `get_adapters` dispatches on), so it must equal the adapter's on-disk folder name.

    Idempotent/last-wins: re-registering a folder replaces the previous class (so an optional
    plugin can override a default if it ever needs to). Raises TypeError if `adapter_cls`
    is not a SourceAdapter subclass — a cheap guard against mis-wiring the seam.
    """
    if not (isinstance(adapter_cls, type) and issubclass(adapter_cls, SourceAdapter)):
        raise TypeError(
            f"register_adapter expects a SourceAdapter subclass, got {adapter_cls!r}"
        )
    ADAPTER_BY_FOLDER[folder] = adapter_cls


def _load_private_plugins() -> None:
    """Best-effort load of the OPTIONAL out-of-tree adapter bootstrap.

    `sources/_private_plugins.py` (absent by default) lets a deployment register its own
    corpus-specific adapters via `register_adapter`. We import it for its side effects and
    SILENTLY skip if it's absent — that ImportError is the expected, normal case, not an error.

    Defensive scope: we only swallow the bootstrap module's OWN absence. If the bootstrap is
    present but itself raises a *different* ImportError (e.g. a typo'd import inside it), we
    re-raise so a real bug in the plugin isn't hidden as "no plugins".
    """
    import importlib
    import importlib.util

    # Probe for the optional module WITHOUT importing it through the (still partially
    # initialised) parent package — `from . import _private_plugins` during package import
    # raises an ImportError naming the parent, not the missing submodule, so we can't reliably
    # distinguish "module absent" from "module broken" that way. find_spec answers the absence
    # question cleanly: None → not present → run sample-only.
    if importlib.util.find_spec(f"{__package__}._private_plugins") is None:
        return
    # Present → import it for its registration side effects. A *broken* import inside a present
    # bootstrap propagates (a real error we must not hide as "no plugins").
    importlib.import_module(f"{__package__}._private_plugins")


# Attempt the optional registration once, at import time. By default this is a no-op (module
# absent); a deployment that ships `_private_plugins.py` can register its own adapters here.
_load_private_plugins()


def get_adapters(corpus_root: Path) -> list[SourceAdapter]:
    """Build one adapter per recognised source-set folder found under `corpus_root`.

    A real corpus root may contain other folders too; we only instantiate adapters for
    folders we have a registered class for, and silently ignore the rest (so an
    unrelated sibling directory doesn't break ingest).
    """
    corpus_root = Path(corpus_root)
    adapters: list[SourceAdapter] = []
    if not corpus_root.exists():
        return adapters
    for child in sorted(p for p in corpus_root.iterdir() if p.is_dir()):
        cls = ADAPTER_BY_FOLDER.get(child.name)
        if cls is not None:
            adapters.append(cls(child))
    return adapters


def discover_all(corpus_root: Path) -> list[SourceDoc]:
    """Run every applicable adapter and collect all candidate SourceDocs.

    This is the single entry point the rest of the pipeline uses to "see" the corpus.
    Returns a flat list (candidates, pre-classification)."""
    docs: list[SourceDoc] = []
    for adapter in get_adapters(corpus_root):
        docs.extend(adapter.discover())
    return docs
