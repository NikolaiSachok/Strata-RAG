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
  * The PUBLIC "bring your own adapters" path: point the `RAGEVAL_PLUGINS_DIR` env var at a
    directory of plugin modules (`*.py`) that live ENTIRELY OUTSIDE this package. On import we
    scan that dir and import each module so it can call `register_adapter()` (and
    `register_family()` from `rageval.sources`) at import time. This lets an external overlay
    extend the engine to its own corpus with NO file copied into the public package — the
    open-core extension story without forking. Unset/absent dir → clean no-op (sample-only); a
    PRESENT-but-broken plugin raises (a real error, never silently swallowed).

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
#
# CONSTRAINT — process-global, ONE corpus per process. This dict (and the declared-facet union +
# classification policies derived from it, plus query_classes._REGISTRY and
# roster._FAMILY_TO_TSV_STEM) is module-level MUTABLE state shared across the interpreter. Two
# corpora registering conflicting adapters/facets in the SAME process can cross-misroute. Today the
# engine serves one corpus per process; the tests reset these globals between cases via an autouse
# fixture (tests/conftest.py). Follow-up: per-corpus registry scoping (an owning Engine/Corpus
# object) — tracked as "follow-up: per-corpus registry scoping".
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


PLUGINS_DIR_ENV = "RAGEVAL_PLUGINS_DIR"


def _load_external_plugins() -> None:
    """Import every plugin module found in `$RAGEVAL_PLUGINS_DIR`, for its side effects.

    The PUBLIC, fork-free extension path. If `RAGEVAL_PLUGINS_DIR` is set and points at an
    existing directory, every `*.py` file directly in it (excluding dunder files like
    `__init__.py`) is imported under a synthetic `rageval_plugin_<name>` module name. Each module
    is expected to call `register_adapter()` (and optionally `register_family()` from
    `rageval.sources`) at import time, wiring its own corpus into the engine WITHOUT copying any
    file into this package.

    Error policy (intentional, mirrors `_load_private_plugins`): an UNSET or NON-EXISTENT dir is a
    clean no-op (the engine stays sample-only). A PRESENT directory whose plugin fails to import
    propagates the exception — a present-but-broken plugin is a real wiring error and must NOT be
    silently skipped (the failure surfaces with the offending file path attached).
    """
    import importlib.util
    import os

    raw = os.environ.get(PLUGINS_DIR_ENV, "").strip()
    if not raw:
        return  # unset → sample-only, no-op
    plugins_dir = Path(raw).expanduser()
    if not plugins_dir.is_dir():
        return  # absent dir → no-op (nothing to load)

    # Deterministic order so a plugin set that depends on import order is reproducible.
    for path in sorted(plugins_dir.glob("*.py")):
        if path.name.startswith("__"):
            continue  # skip __init__.py / dunder helpers
        mod_name = f"rageval_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ImportError(f"could not load plugin spec from {path}")
        module = importlib.util.module_from_spec(spec)
        # No try/except: a broken plugin raises here, with the file in the traceback. We add the
        # path to the message so the operator knows WHICH plugin is broken.
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - re-raise enriched, never swallow
            raise ImportError(f"failed to import plugin {path}: {exc}") from exc


# A module-level guard so the optional registrations run EXACTLY ONCE per process, no matter how
# the loader is reached. The trigger itself lives at the END of `sources/__init__.py` (see
# `load_optional_plugins`), NOT here at registry-import time: an external plugin follows the
# documented ergonomic path `from rageval.sources import register_adapter, register_family`, and
# those facade re-exports are only bound once `__init__.py` has finished importing this module.
# Running the loader here (mid-`__init__`) would import such a plugin against a half-initialised
# `rageval.sources` and fail with a circular import. Deferring to the end of `__init__.py` makes
# the documented import work; the guard makes a standalone `import rageval.sources.registry`
# (which does NOT re-trigger __init__) safe to combine with the normal package import.
_plugins_loaded = False


def load_optional_plugins() -> None:
    """Run the optional in-package + external plugin registrations exactly once.

    Called at the very END of `rageval/sources/__init__.py`, after the package facade
    (`register_adapter`, `register_family`, the adapter classes) is fully bound — so a plugin
    importing `from rageval.sources import register_adapter, register_family` resolves cleanly.

    By default both steps are no-ops (the in-package bootstrap module is absent AND
    RAGEVAL_PLUGINS_DIR is unset) → sample-only. A deployment that ships `_private_plugins.py`,
    OR points RAGEVAL_PLUGINS_DIR at a dir of plugin modules, registers its own adapters/families
    here. Idempotent: subsequent calls are a no-op (the facade is imported once per process, but
    importing `registry` standalone must not double-load plugins)."""
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    # In-package bootstrap first, then the external dir — matches the documented order ("after the
    # in-package plugin bootstrap").
    _load_private_plugins()
    _load_external_plugins()


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


def adapter_class_for_source_set(source_set: str) -> type[SourceAdapter] | None:
    """Resolve the registered adapter CLASS that produces `source_set` (Phase-4 fact/policy hooks).

    The registry keys adapters by their on-disk FOLDER, which usually equals `source_set`; but a
    corpus may split one folder into several source-sets (e.g. a `foo` folder emitting `foo` and
    `foo-variant`). So we match by the adapter's declared `source_set` first (exact), then fall
    back to the base FAMILY (the part before the first '-', mirroring the roster join). Returns None
    when no adapter is registered for it — the caller degrades gracefully (no structured facts).

    This is what lets the corpus-agnostic pipeline (enrich.py) ask an adapter for its structured
    facts WITHOUT the core knowing any adapter or field name."""
    if not source_set:
        return None
    # Exact match on the adapter's declared source_set.
    for cls in ADAPTER_BY_FOLDER.values():
        if getattr(cls, "source_set", None) == source_set:
            return cls
    # Fall back to the base family (before the first '-'), matched against folder key or source_set.
    family = source_set.split("-", 1)[0].lower()
    for folder, cls in ADAPTER_BY_FOLDER.items():
        if folder.lower() == family or str(getattr(cls, "source_set", "")).lower() == family:
            return cls
    return None


def all_declared_facets() -> dict[str, str]:
    """The UNION of every registered adapter's declared facets → {facet_name: facet_type} (#36).

    This is the schema-agnostic source of truth for what the sidecar facet store will ACCEPT and
    what `aggregate` will treat as a QUERYABLE field — derived from adapter DECLARATIONS, never a
    core dataclass. On a facet-name/type collision across adapters, first-registered wins (a warning
    is out of scope here; corpora are expected to namespace or agree). Includes any plugin-loaded
    adapters (they've registered by the time this runs)."""
    ensure_plugins_loaded()
    out: dict[str, str] = {}
    for cls in ADAPTER_BY_FOLDER.values():
        try:
            facets = cls(Path(".")).declared_facets()
        except Exception:  # noqa: BLE001 — a broken adapter never breaks fact-schema resolution
            continue
        for spec in facets:
            out.setdefault(spec.name, spec.type)
    return out


def ensure_plugins_loaded() -> None:
    """Trigger the optional plugin load (idempotent) so out-of-tree adapters are registered before
    we enumerate declared facets. Safe to call repeatedly (guarded to run once per process)."""
    load_optional_plugins()


def discover_all(corpus_root: Path) -> list[SourceDoc]:
    """Run every applicable adapter and collect all candidate SourceDocs.

    This is the single entry point the rest of the pipeline uses to "see" the corpus.
    Returns a flat list (candidates, pre-classification)."""
    docs: list[SourceDoc] = []
    for adapter in get_adapters(corpus_root):
        docs.extend(adapter.discover())
    return docs


def harvest_all_entities(corpus_root: Path) -> list:
    """Run every applicable adapter's `harvest_entities` hook (#41) and collect the STRUCTURED,
    facts-only entities (spreadsheet rows etc.) across the corpus.

    Parallel to `discover_all`, but for the structured/aggregation path: these entities become
    their OWN facts-only sidecar records (never embedded). An adapter that has no tabular data
    overrides nothing → empty. A broken hook degrades to [] for that adapter (never crashes
    ingest)."""
    from .base import HarvestedEntity

    entities: list[HarvestedEntity] = []
    for adapter in get_adapters(corpus_root):
        try:
            entities.extend(adapter.harvest_entities())
        except Exception:  # noqa: BLE001 — a broken tabular harvest never crashes ingest
            continue
    return entities
