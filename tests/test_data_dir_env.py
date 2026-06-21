"""Tests for RAGEVAL_DATA_DIR — the overlay-configurable data-dir override (issue #14).

THE DECISION under test: the engine's *mutable / corpus-specific* paths (sidecar, rules, golden,
manifests, and the top-level roster dir) must resolve under an overlay's data folder when it
pip-installs the engine, NOT under the engine's own package dir. RAGEVAL_DATA_DIR provides that,
read at config IMPORT time so the module-level constants (used as default params like
`def connect(path=SIDECAR_PATH)`) reflect it. Default (no env) stays PROJECT_ROOT — the bundled
sample + every existing behaviour are byte-for-byte unchanged (back-compat).

We re-resolve config in a FRESH subprocess per case so the env is read at the real import point
and the shared in-process `rageval.config` is never mutated for the rest of the suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# A tiny program that imports rageval.config and prints the resolved paths as JSON, so each case
# observes config as it would resolve in a clean process with the given environment.
_PROBE = """
import json
import rageval.config as c
print(json.dumps({
    "DATA_DIR": str(c.DATA_DIR),
    "PROJECT_ROOT": str(c.PROJECT_ROOT),
    "SIDECAR_PATH": str(c.SIDECAR_PATH),
    "RULES_PATH": str(c.RULES_PATH),
    "GOLDEN_PATH": str(c.GOLDEN_PATH),
    "MANIFEST_DIR": str(c.MANIFEST_DIR),
    "ROSTER_DIR": str(c.ROSTER_DIR),
    "roster_dir_setting": c.Settings.load().roster_dir,
}))
"""


def _resolve_config(**env_overrides: str | None) -> dict[str, str]:
    """Import rageval.config in a fresh subprocess under the given env and return its paths."""
    env = dict(os.environ)
    # Drop any data-dir/roster envs the developer's shell might carry, then apply the case's.
    for key in ("RAGEVAL_DATA_DIR", "RAGEVAL_ROSTER_DIR"):
        env.pop(key, None)
    for key, val in env_overrides.items():
        if val is None:
            env.pop(key, None)
        else:
            env[key] = val
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env, capture_output=True, text=True, check=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    return json.loads(out.stdout)


# --- RAGEVAL_DATA_DIR set → all data paths resolve under it ------------------

def test_data_dir_set_relocates_all_paths(tmp_path):
    data_dir = tmp_path / "overlay-data"
    data_dir.mkdir()
    cfg = _resolve_config(RAGEVAL_DATA_DIR=str(data_dir))

    expected = data_dir.resolve()
    assert cfg["DATA_DIR"] == str(expected)
    assert cfg["SIDECAR_PATH"] == str(expected / "rageval.sqlite")
    assert cfg["RULES_PATH"] == str(expected / "corpus-rules.yaml")
    assert cfg["GOLDEN_PATH"] == str(expected / "eval" / "golden.yaml")
    assert cfg["MANIFEST_DIR"] == str(expected / "manifests")
    # ROSTER_DIR also moves under DATA_DIR (so ONE env relocates all of an overlay's data).
    assert cfg["ROSTER_DIR"] == str(expected / "data")


def test_data_dir_expands_user_and_resolves(tmp_path):
    # `~` expansion + .resolve() — pass an absolute path and confirm it round-trips resolved.
    cfg = _resolve_config(RAGEVAL_DATA_DIR=str(tmp_path))
    assert cfg["DATA_DIR"] == str(tmp_path.resolve())


# --- unset → back-compat: everything resolves under PROJECT_ROOT -------------

def test_unset_falls_back_to_project_root():
    cfg = _resolve_config()  # no RAGEVAL_DATA_DIR / RAGEVAL_ROSTER_DIR
    root = Path(cfg["PROJECT_ROOT"])
    assert cfg["DATA_DIR"] == str(root)
    assert cfg["SIDECAR_PATH"] == str(root / "rageval.sqlite")
    assert cfg["RULES_PATH"] == str(root / "corpus-rules.yaml")
    assert cfg["GOLDEN_PATH"] == str(root / "eval" / "golden.yaml")
    assert cfg["MANIFEST_DIR"] == str(root / "manifests")
    assert cfg["ROSTER_DIR"] == str(root / "data")


def test_unset_bundled_sample_paths_still_valid():
    """Back-compat: with no env, the committed sample data the engine ships actually exists at the
    resolved paths (rules + golden are committed; the roster dir holds the sample subtree)."""
    cfg = _resolve_config()
    assert Path(cfg["RULES_PATH"]).is_file()
    assert Path(cfg["GOLDEN_PATH"]).is_file()
    assert Path(cfg["ROSTER_DIR"]).is_dir()


# --- roster precedence: RAGEVAL_ROSTER_DIR wins; DATA_DIR is the fallback ----

def test_roster_dir_env_takes_precedence_over_data_dir(tmp_path):
    """RAGEVAL_ROSTER_DIR is the more-specific override and must win for the roster dir even when
    RAGEVAL_DATA_DIR is also set (an overlay can split rosters out of its main data folder)."""
    data_dir = tmp_path / "overlay-data"
    roster_dir = tmp_path / "rosters-elsewhere"
    data_dir.mkdir()
    roster_dir.mkdir()
    cfg = _resolve_config(
        RAGEVAL_DATA_DIR=str(data_dir),
        RAGEVAL_ROSTER_DIR=str(roster_dir),
    )
    # Settings.load() carries the explicit roster override (resolved by roster._roster_dir_for).
    assert cfg["roster_dir_setting"] == str(roster_dir)
    # The base ROSTER_DIR constant still follows DATA_DIR (the fallback), but the setting wins.
    assert cfg["ROSTER_DIR"] == str(data_dir.resolve() / "data")


def test_roster_falls_back_to_data_dir_when_only_data_dir_set(tmp_path):
    """When ONLY RAGEVAL_DATA_DIR is set, the roster dir bases on DATA_DIR too — so one env var
    points the engine at ALL of an overlay's data (sidecar/rules/golden/manifests + rosters)."""
    data_dir = tmp_path / "overlay-data"
    data_dir.mkdir()
    cfg = _resolve_config(RAGEVAL_DATA_DIR=str(data_dir))
    # No explicit roster override → Settings.roster_dir is empty (roster.py derives from ROSTER_DIR).
    assert cfg["roster_dir_setting"] == ""
    assert cfg["ROSTER_DIR"] == str(data_dir.resolve() / "data")


# --- empty / whitespace env → fall back to PROJECT_ROOT (the classic empty-env bug) ----

def test_empty_or_whitespace_env_falls_back_to_project_root():
    """An empty or whitespace-only RAGEVAL_DATA_DIR must fall back to PROJECT_ROOT — NOT resolve to
    CWD or a literal-space directory. This is the single most important edge case for an env path."""
    root = _resolve_config()["PROJECT_ROOT"]
    for val in ("", "   "):
        cfg = _resolve_config(RAGEVAL_DATA_DIR=val)
        assert cfg["DATA_DIR"] == root, f"{val!r} should fall back to PROJECT_ROOT"
        assert cfg["SIDECAR_PATH"] == str(Path(root) / "rageval.sqlite")
