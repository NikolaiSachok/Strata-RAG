"""Tests for the EXTERNAL plugin loader (`RAGEVAL_PLUGINS_DIR`) + `register_family()`.

These lock down the public, fork-free "bring your own adapters" path: an overlay drops adapter
modules in a directory, points `RAGEVAL_PLUGINS_DIR` at it, and on registry import those modules
call `register_adapter()` / `register_family()` to extend the engine to a NEW corpus — without
copying any file into the package. Everything here is FICTIONAL (a hypothetical `mycorp` corpus);
no real names/ids/paths ever appear.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from rageval import roster as roster_mod
from rageval.roster import Roster, register_family
from rageval.sources import registry


# --- shared snapshot/restore so a test can mutate the global registries freely ----------------

@pytest.fixture
def clean_registries():
    """Snapshot + restore BOTH global maps (adapter folders, roster families)."""
    saved_adapters = dict(registry.ADAPTER_BY_FOLDER)
    saved_families = dict(roster_mod._FAMILY_TO_TSV_STEM)
    try:
        yield
    finally:
        registry.ADAPTER_BY_FOLDER.clear()
        registry.ADAPTER_BY_FOLDER.update(saved_adapters)
        roster_mod._FAMILY_TO_TSV_STEM.clear()
        roster_mod._FAMILY_TO_TSV_STEM.update(saved_families)


# A plugin module body: a fictional `mycorp` adapter that registers itself + a roster family at
# import time. Written verbatim into a tmp RAGEVAL_PLUGINS_DIR by the tests below.
_MYCORP_PLUGIN = '''
from rageval.sources import register_adapter, register_family
from rageval.sources.base import SourceAdapter, SourceDoc


class MyCorpAdapter(SourceAdapter):
    source_set = "mycorp"

    def discover(self):
        yield SourceDoc(
            project_id="0007", source_set=self.source_set,
            doc_path=self.root / "0007" / "description.md", doc_type="description",
            ext="md", raw_text="a fictional mycorp product description",
        )


register_adapter("mycorp", MyCorpAdapter)
register_family("mycorp", "mycorp")
'''


# --- (a) a plugin in RAGEVAL_PLUGINS_DIR gets imported and its registrations take effect -------

def test_plugins_dir_module_is_imported_and_registers(clean_registries, monkeypatch, tmp_path):
    """A `*.py` in RAGEVAL_PLUGINS_DIR is imported on registry load; its register_adapter()
    wires a new folder→adapter (dispatch works) and register_family() wires a new roster family
    (the roster join works)."""
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "mycorp_plugin.py").write_text(_MYCORP_PLUGIN, encoding="utf-8")

    monkeypatch.setenv("RAGEVAL_PLUGINS_DIR", str(plugins))
    registry._load_external_plugins()  # what the registry runs at import time

    # register_adapter() took effect → dispatch to a real corpus folder works.
    assert "mycorp" in registry.ADAPTER_BY_FOLDER
    corpus = tmp_path / "corpus"
    (corpus / "mycorp").mkdir(parents=True)
    docs = registry.discover_all(corpus)
    assert [d.source_set for d in docs] == ["mycorp"]

    # register_family() took effect → the roster family→stem mapping is present and the loader
    # uses it. Write a fictional mycorp.tsv and confirm the join resolves a publisher.
    assert roster_mod._FAMILY_TO_TSV_STEM["mycorp"] == "mycorp"
    roster_dir = tmp_path / "rosters"
    roster_dir.mkdir()
    (roster_dir / "mycorp.tsv").write_text(
        "№\tID\tPublisher\tBundle\n0007\tcom.x.y\tNimbus Forge\tcom.x.y\n", encoding="utf-8"
    )
    r = Roster(roster_dir)
    assert r.publisher("mycorp", "0007") == "Nimbus Forge"


# --- (b) unset/absent dir = clean no-op (registry stays sample-only) ---------------------------

def test_unset_plugins_dir_is_noop(clean_registries, monkeypatch):
    monkeypatch.delenv("RAGEVAL_PLUGINS_DIR", raising=False)
    before = set(registry.ADAPTER_BY_FOLDER)
    registry._load_external_plugins()
    assert set(registry.ADAPTER_BY_FOLDER) == before  # nothing added


def test_absent_plugins_dir_is_noop(clean_registries, monkeypatch, tmp_path):
    monkeypatch.setenv("RAGEVAL_PLUGINS_DIR", str(tmp_path / "does-not-exist"))
    before = set(registry.ADAPTER_BY_FOLDER)
    registry._load_external_plugins()
    assert set(registry.ADAPTER_BY_FOLDER) == before


# --- (c) register_family() adds a family→stem mapping the roster loader uses -------------------

def test_register_family_adds_mapping_used_by_roster(clean_registries, tmp_path):
    register_family("mycorp", "mycorp")
    roster_dir = tmp_path / "rosters"
    roster_dir.mkdir()
    (roster_dir / "mycorp.tsv").write_text(
        "№\tID\tPublisher\tBundle\n0042\tcom.x.y\tAmber Hollow\tcom.x.y\n", encoding="utf-8"
    )
    r = Roster(roster_dir)
    # Shared family file: every mycorp* subset joins the same TSV.
    assert r.publisher("mycorp", "0042") == "Amber Hollow"
    assert r.publisher("mycorp-extra", "0042-sp01") == "Amber Hollow"
    # An unregistered family is still null (graceful degradation, unchanged).
    assert r.publisher("unknown-set", "0042") is None


def test_register_family_rejects_empty_args(clean_registries):
    with pytest.raises(ValueError):
        register_family("", "mycorp")
    with pytest.raises(ValueError):
        register_family("mycorp", "")


# --- (d) a present-but-broken plugin raises (not silently skipped) -----------------------------

def test_broken_plugin_raises(clean_registries, monkeypatch, tmp_path):
    """A plugin that fails to import must surface as an error — a present-but-broken plugin is a
    real wiring bug, never a silent 'no plugins'."""
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "broken_plugin.py").write_text(
        "import this_module_does_not_exist_xyz  # noqa\n", encoding="utf-8"
    )
    monkeypatch.setenv("RAGEVAL_PLUGINS_DIR", str(plugins))
    with pytest.raises(ImportError):
        registry._load_external_plugins()


def test_dunder_files_are_skipped(clean_registries, monkeypatch, tmp_path):
    """An __init__.py in the plugins dir is NOT imported as a standalone plugin module (so a dir
    can carry package scaffolding without it being double-loaded)."""
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    # A dunder file that would explode if imported — it must be skipped.
    (plugins / "__init__.py").write_text("raise RuntimeError('should be skipped')\n", encoding="utf-8")
    monkeypatch.setenv("RAGEVAL_PLUGINS_DIR", str(plugins))
    registry._load_external_plugins()  # no raise → dunder skipped


# --- (e) REGRESSION: the ergonomic facade import works during package import ------------------
#
# A plugin whose top-level imports are EXACTLY the documented ergonomic path
# `from rageval.sources import register_adapter, register_family` must load when the loader runs
# at `import rageval.sources` time — NOT only when `registry._load_external_plugins()` is called
# directly. The bug: the loader was triggered at `registry.py` MODULE level, i.e. mid-execution
# of `sources/__init__.py`, BEFORE the facade re-exported `register_adapter`/`register_family`.
# So the plugin's first import hit a partially-initialised `rageval.sources` and raised a circular
# ImportError. The fix moved the trigger to the END of `sources/__init__.py`. The other tests in
# this file call `_load_external_plugins()` directly (facade already imported), so they would NOT
# catch the regression — this one drives the real `import rageval.sources` path in a fresh
# interpreter, which is exactly what an overlay does in production.

# A plugin whose top-level imports are ONLY the ergonomic facade path (the documented, README
# copy-paste import). Under the bug this module fails to import with a circular ImportError.
_ERGONOMIC_PLUGIN = '''
from rageval.sources import register_adapter, register_family
from rageval.sources.base import SourceAdapter, SourceDoc


class ErgoCorpAdapter(SourceAdapter):
    source_set = "ergocorp"

    def discover(self):
        yield SourceDoc(
            project_id="0001", source_set=self.source_set,
            doc_path=self.root / "0001" / "description.md", doc_type="description",
            ext="md", raw_text="a fictional ergocorp product description",
        )


register_adapter("ergocorp", ErgoCorpAdapter)
register_family("ergocorp", "ergocorp")
'''


def test_ergonomic_facade_import_loads_via_package_import(tmp_path):
    """A plugin importing `from rageval.sources import register_adapter, register_family` loads
    cleanly when triggered by `import rageval.sources` itself (the production path), WITHOUT a
    circular-import error, and its registrations take effect.

    Run in a FRESH interpreter: the loader runs exactly once per process at first
    `import rageval.sources`, so the env var must be set before that import — a subprocess is the
    faithful way to reproduce what an external overlay does (and what the bug broke)."""
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "ergocorp_plugin.py").write_text(_ERGONOMIC_PLUGIN, encoding="utf-8")

    # Importing rageval.sources MUST trigger the loader (end-of-__init__) → plugin imported via the
    # ergonomic facade path → registrations visible. Assert all of it inside the child so a
    # circular ImportError fails the test with the real traceback.
    child = textwrap.dedent(
        """
        import rageval.sources as s          # triggers load_optional_plugins() at end of __init__
        from rageval.sources import registry
        assert "ergocorp" in registry.ADAPTER_BY_FOLDER, "register_adapter() did not take effect"
        from rageval import roster as roster_mod
        assert roster_mod._FAMILY_TO_TSV_STEM.get("ergocorp") == "ergocorp", \\
            "register_family() did not take effect"
        # facade re-exports are bound (the names the plugin imported)
        assert hasattr(s, "register_adapter") and hasattr(s, "register_family")
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", child],
        env={"RAGEVAL_PLUGINS_DIR": str(plugins), "PATH": ""},
        capture_output=True,
        text=True,
    )
    # On the bug, the child crashes with: ImportError: cannot import name 'register_adapter' from
    # partially initialized module 'rageval.sources' (most likely due to a circular import).
    assert result.returncode == 0, (
        "ergonomic-facade plugin failed to load via `import rageval.sources` "
        f"(circular-import regression?):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_package_import_loads_plugins_exactly_once(tmp_path):
    """`import rageval.sources` (once per process) loads external plugins exactly once, and a
    later standalone `import rageval.sources.registry` does NOT re-load them (guard holds)."""
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    # A plugin that COUNTS its own imports via a sentinel file — a double-load would append twice.
    (plugins / "counter_plugin.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "Path(os.environ['ERGO_COUNT_FILE']).open('a').write('x')\n",
        encoding="utf-8",
    )
    count_file = tmp_path / "count.txt"
    child = textwrap.dedent(
        """
        import rageval.sources                 # first import → loader runs once
        import rageval.sources.registry as r   # standalone import must NOT re-trigger the loader
        r.load_optional_plugins()              # explicit re-call must be a guarded no-op
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", child],
        env={
            "RAGEVAL_PLUGINS_DIR": str(plugins),
            "ERGO_COUNT_FILE": str(count_file),
            "PATH": "",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert count_file.read_text() == "x", "external plugin loaded more than once per process"
