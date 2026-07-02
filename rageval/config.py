"""Central configuration — every tunable in one place, all env-overridable.

WHY a config module: an enterprise RAG pipeline has a LOT of knobs (which corpus,
which embedding model, how many chunks to retrieve, HNSW ANN parameters, hybrid
fusion weights, the re-ranker model, the sidecar path). Scattering them across
modules makes the system impossible to reason about and tune. Collecting them here
means you can read the whole "shape" of the system at a glance, and override any of
it from the environment (or a .env file) without touching code.

Nothing here imports heavy libraries (no qdrant-client / sentence-transformers /
anthropic), so it stays cheap to import from tests and the API layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Repo paths. PROJECT_ROOT is the folder that contains `data/`, the metadata sidecar,
# corpus-rules.yaml, and eval/. We resolve relative to THIS file so the package works
# no matter where you launch it from.
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
SAMPLE_CORPUS_DIR = PROJECT_ROOT / "data" / "sample"  # the synthetic, committed corpus

# DATA_DIR is the folder the engine's *mutable / corpus-specific* data resolves under: the
# metadata sidecar, corpus-rules.yaml, the golden eval set, the manifests, and (when no more
# specific override is set) the roster TSVs. It defaults to PROJECT_ROOT so the bundled sample
# corpus + every existing behaviour are byte-for-byte UNCHANGED with no env set.
#
# WHY this exists: after the open-core split the engine ships as its own package that a
# *consuming overlay* pip-installs, while the overlay's REAL data (sidecar, rules, golden,
# manifests) lives WITH the overlay — not inside the engine's package dir. Hardcoding these to
# PROJECT_ROOT makes the served engine look for the sidecar in its own install dir, so
# aggregation/lookup queries fail ("unable to open database file"). RAGEVAL_DATA_DIR lets an
# overlay point the engine at its own data folder with ONE env var (companion to
# RAGEVAL_PLUGINS_DIR for adapters and RAGEVAL_ROSTER_DIR for rosters).
#
# Read at IMPORT time so the module-level path constants below — used as default params like
# `def connect(path=SIDECAR_PATH)` — reflect the override.
# `.strip()` so an empty OR whitespace-only env value falls back to PROJECT_ROOT, rather than a
# truthy "   " resolving to a literal-space dir (the classic empty-env footgun).
_data_dir_env = os.environ.get("RAGEVAL_DATA_DIR", "").strip()
DATA_DIR = Path(_data_dir_env).expanduser().resolve() if _data_dir_env else PROJECT_ROOT

# Roster TSVs (project-id → authoritative publisher; see roster.py). The synthetic sample
# rosters (data/sample/*.tsv) ship with the repo. The loader is GENERIC and degrades to null
# when a roster file is absent (a corpus need not supply a roster at all).
# Based on DATA_DIR so an overlay's RAGEVAL_DATA_DIR also relocates the top-level roster dir;
# RAGEVAL_ROSTER_DIR still takes precedence (resolved in roster.py / Settings.load).
ROSTER_DIR = DATA_DIR / "data"                         # top-level rosters (for a custom corpus)
SAMPLE_ROSTER_DIR = SAMPLE_CORPUS_DIR                  # fictional sample rosters
RULES_PATH = DATA_DIR / "corpus-rules.yaml"            # the trusted relevance ruleset
GOLDEN_PATH = DATA_DIR / "eval" / "golden.yaml"        # labelled eval questions
SIDECAR_PATH = DATA_DIR / "rageval.sqlite"             # structured metadata sidecar (gitignored)
MANIFEST_DIR = DATA_DIR / "manifests"                  # dry-run manifests (gitignored)

# Base name for the Qdrant collection (the "table" the chunks live in). The ACTUAL name is
# DERIVED PER EMBEDDING MODEL (see Settings.collection_name) so two models' indexes coexist
# for a side-by-side A/B — e.g. "rageval_chunks_all_minilm_l6_v2" vs "..._all_mpnet_base_v2".
COLLECTION_BASE = "rageval_chunks"

# Suffix appended to the collection name when the corpus is the synthetic SAMPLE (data/sample/).
# This isolates the demo/test corpus into its OWN collection so a sample ingest can NEVER upsert
# its fictional points into a real-corpus collection (the model-derived name alone is shared by
# every corpus that uses that model — a sample ingest and a real ingest under the same embedder
# would otherwise land in the SAME collection, and because ingest UPSERTs, the sample points
# would persist and leak into real-corpus retrieval). See Settings.collection_name.
SAMPLE_COLLECTION_SUFFIX = "sample"


def is_sample_corpus(corpus_root: Path) -> bool:
    """True when `corpus_root` IS or is UNDER the shipped synthetic sample corpus (data/sample/).

    The single sample-vs-real predicate, shared by the roster directory logic (roster.py)
    and the Qdrant collection-name suffix (Settings.collection_name) — so "this is the sample
    corpus" means exactly one thing across the engine. Robust to a sub-path of data/sample/ and
    to a non-existent path (resolve(strict=False))."""
    try:
        root = corpus_root.resolve()
        sample = SAMPLE_CORPUS_DIR.resolve()
    except OSError:
        return False
    return root == sample or sample in root.parents


def slug(text: str) -> str:
    """Turn a model id into a Qdrant-safe collection suffix: lowercase, every run of
    non-alphanumerics → a single '_', stripped of leading/trailing '_'.

    e.g. "sentence-transformers/all-MiniLM-L6-v2" → "sentence_transformers_all_minilm_l6_v2".
    Pure + deterministic → both ingest and retrieve compute the SAME name from the same model,
    which is the whole point: the index a model wrote is the index it reads back."""
    import re

    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

# DEFAULT corpus intent. "Noise vs signal" only means something relative to a stated
# purpose, so the relevance classifier (classify.py) judges every file against THIS.
# Flip it (e.g. to a code/technical-doc RAG) and re-classify with NO code change — only
# this string + corpus-rules.yaml change. (Kept domain-neutral; see corpus-rules.yaml.)
DEFAULT_CORPUS_INTENT = (
    "user-facing application content — themes, product names, feature descriptions, and "
    "promotional copy; NOT implementation/build/technical docs, changelogs, or "
    "agent-authored planning files."
)


def _env(name: str, default: str) -> str:
    """Read an env var, falling back to a default. Centralised so the defaults are
    visible in one list rather than sprinkled through os.getenv calls."""
    return os.environ.get(name, default).strip() or default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_float_opt(name: str) -> float | None:
    """Optional float: unset/blank/unparseable → None (the feature stays OFF). Used for
    knobs whose ABSENCE is meaningful (a disabled threshold ≠ a threshold of 0.0)."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of all knobs, read from the environment once.

    Call `Settings.load()` to build it. Frozen so a module can't accidentally mutate
    shared config at runtime. Tests build their own via `dataclasses.replace(...)`.
    """

    # --- Corpus -------------------------------------------------------------
    # Where the documents live. Defaults to the synthetic sample corpus that SHIPS
    # with the repo, so everything runs out of the box and tests are deterministic.
    # Point at a real corpus by setting RAGENGINE_CORPUS_ROOT — it is NEVER copied
    # into the repo and its index/sidecar are gitignored.
    corpus_root: Path
    corpus_intent: str   # what counts as "signal"; drives the relevance classifier

    # Directory holding the roster TSVs (project-id → authoritative publisher; roster.py).
    # Empty → DERIVE from corpus_root (sample corpus uses data/sample/, any other uses data/).
    # Override with RAGEVAL_ROSTER_DIR (e.g. to point the demo at a custom roster dir).
    roster_dir: str

    # --- LLM (generation + judge + enrichment + the Tier-2 rule advisor) ----
    # Backend selection mirrors the original demo: explicit override, else API key,
    # else `claude` CLI. Resolved lazily in llm.py so importing config is free.
    llm_backend: str  # "auto" | "api" | "cli"
    model: str        # Anthropic model id used for generation, judging, enrichment.

    # --- Multimodal provider seams (Phase 1.5; see vision.py / ocr.py) ------
    # PLUGGABLE backends for the two image→text tasks, selected by name from a registry.
    # Both default to a real backend where feasible and degrade gracefully (skip-with-reason,
    # never crash) when that backend is absent/misconfigured — the same failure model as
    # enrichment. A deterministic "mock" backend covers CI so no test needs a live model.
    #   vision_provider — describe/caption an image. "claude" (default, via the llm.py API path),
    #                     "mock" (CI), or any registered backend (another cloud LLM / local VLM).
    #   ocr_provider    — image→text OCR. "tesseract" (default where the binary+pytesseract are
    #                     present; degrades if absent), "llm" (reuse the vision provider as OCR),
    #                     "mock" (CI), or any registered backend (cloud OCR / local VLM).
    # The chosen provider is recorded in any output payload for auditability. NO hard import
    # dependency on a cloud SDK or Tesseract — backends lazy-import their libs (vision.py/ocr.py).
    vision_provider: str  # "claude" | "mock" | <registered name>
    ocr_provider: str     # "tesseract" | "llm" | "mock" | <registered name>
    vision_model: str     # model id for the Claude vision backend (defaults to `model`)

    # --- Enrichment concurrency --------------------------------------------
    # The enrichment pass makes ONE blocking LLM call per project (~50s each on the `claude`
    # CLI). Sequentially over a large corpus (hundreds of projects) that's hours. The calls are
    # independent and I/O-bound (a subprocess/HTTP wait that RELEASES the GIL), so a bounded
    # ThreadPool collapses the wall-clock to ~minutes. This is the worker count.
    enrich_concurrency: int  # parallel enrichment workers (RAGEVAL_ENRICH_CONCURRENCY, default 8)

    # --- PII detection backend (the DETECTOR under the policy-aware PII layer; see pii.py) ---
    # "regex" (DEFAULT, dependency-free, emails only) | "presidio" (optional NER: PERSON,
    # PHONE_NUMBER, IBAN, CREDIT_CARD, …). The keep/redact POLICY (redact.py) is unchanged
    # whichever backend runs. presidio needs the [pii] extra + a spaCy model (download separately).
    pii_backend: str        # "regex" | "presidio"
    pii_spacy_model: str    # spaCy model for the presidio backend (en_core_web_sm default; _lg for prod)

    # --- Embeddings ---------------------------------------------------------
    embeddings: str   # "local" (sentence-transformers) | "openai"
    embed_model: str  # model id for the chosen embeddings backend
    embed_dim: int    # vector dimensionality (must match the model; used to create the Qdrant collection)

    # Explicit collection override. Empty → DERIVE the name from the embedding model (so
    # MiniLM and mpnet indexes coexist). Set RAGEVAL_COLLECTION to pin an exact name (e.g.
    # to A/B two configs that share a model, or to eval an externally-built collection).
    collection_override: str

    # --- Qdrant + HNSW ------------------------------------------------------
    qdrant_url: str        # e.g. http://localhost:6333
    # HNSW (Hierarchical Navigable Small World) is the approximate-nearest-neighbour
    # (ANN) graph index Qdrant uses. The three knobs below trade recall vs. speed/RAM:
    #   m            — edges per node in the graph. Higher = better recall, more RAM.
    #   ef_construct — candidate-list size while BUILDING the graph. Higher = better
    #                  graph quality, slower indexing.
    #   ef_search    — candidate-list size while SEARCHING. Higher = better recall,
    #                  slower queries. (Set per-query at search time.)
    hnsw_m: int
    hnsw_ef_construct: int
    hnsw_ef_search: int

    # --- Retrieval / hybrid / rerank ---------------------------------------
    top_k: int            # final number of chunks handed to the generator
    candidate_k: int      # how many each retriever (dense + sparse) fetches before fusion
    rrf_k: int            # RRF constant (see retrieve.py); 60 is the canonical default
    use_rerank: bool      # run the cross-encoder re-ranker over the fused candidates?
    rerank_model: str     # cross-encoder model id
    # Optional ABSOLUTE rerank-score FLOOR applied AFTER reranking: drop any hit whose
    # cross-encoder rerank_score < this value (top-k WITH a relevance threshold). None =
    # DISABLED (default) → behaviour is byte-for-byte the legacy fixed top-k. See retrieve.py
    # for WHY this is a floor and not a top-p/nucleus cutoff (scores aren't a calibrated
    # probability distribution, so the threshold is empirical and model-specific → off by default).
    min_rerank_score: float | None

    # --- Chunking -----------------------------------------------------------
    chunk_size: int     # target characters per chunk
    chunk_overlap: int  # characters shared between adjacent chunks (preserves context across cuts)

    # --- Prompt-injection guardrails (Phase 1 security; see guardrails.py) ---
    # Each defense layer is a SEPARATE toggle on purpose: defense-in-depth means no single
    # layer is trusted, and being able to flip layers off one at a time makes the engine a
    # teaching artifact — you can measure each layer's individual contribution to the
    # injection-attack-success-rate. All default ON.
    guard_input_scan: bool     # scan retrieved chunks for injection patterns BEFORE generation
    guard_normalize: bool      # normalization pre-pass (NFKC/zero-width/homoglyph/decode) before scanning
    guard_spotlight: bool      # wrap passages in random sentinels + treat them as inert DATA
    guard_output_validate: bool  # validate the answer AFTER generation (exfil/fake-cite/leak)
    guard_quarantine: bool     # DROP (don't just flag) chunks with a high-severity injection hit
    guard_llm_classifier: bool  # also run a 2nd-tier LLM injection classifier (flag-only; costs a call)

    @property
    def collection_name(self) -> str:
        """The Qdrant collection this config reads/writes. An explicit override wins;
        otherwise it's derived from the embedding model so each model gets its OWN index,
        PLUS a `_sample` suffix when the corpus is the synthetic sample.

        This is what makes a clean A/B possible: ingest under MiniLM writes
        `rageval_chunks_all_minilm_l6_v2`, ingest under mpnet writes
        `rageval_chunks_all_mpnet_base_v2`, and an eval pointed at either compares like with
        like. (You can also force a name with --collection / RAGEVAL_COLLECTION.)

        CORPUS ISOLATION: the model-derived name is shared by EVERY corpus using that model, so
        a sample ingest and a real ingest under the same embedder would collide in one collection
        — and because ingest UPSERTs (doesn't recreate), the sample's fictional points would
        persist and leak into real-corpus retrieval (caught once as fake demo apps surfacing in a
        real answer's citations). So the synthetic sample gets its OWN collection via a `_sample`
        suffix; the REAL-corpus name is UNCHANGED (existing real indexes stay valid, no re-ingest).
        The model still differentiates (A/B intact) and an explicit override still wins."""
        if self.collection_override:
            return self.collection_override
        name = f"{COLLECTION_BASE}_{slug(self.embed_model)}"
        if is_sample_corpus(self.corpus_root):
            name = f"{name}_{SAMPLE_COLLECTION_SUFFIX}"
        return name

    @classmethod
    def load(cls) -> "Settings":
        def _flag(name: str, default: bool) -> bool:
            return _env(name, "true" if default else "false").lower() in ("1", "true", "yes")

        embeddings = _env("RAGEVAL_EMBEDDINGS", "local").lower()
        is_local = embeddings == "local"
        return cls(
            corpus_root=Path(
                _env("RAGENGINE_CORPUS_ROOT", str(SAMPLE_CORPUS_DIR))
            ).expanduser(),
            corpus_intent=_env("RAGEVAL_CORPUS_INTENT", DEFAULT_CORPUS_INTENT),
            roster_dir=_env("RAGEVAL_ROSTER_DIR", ""),
            llm_backend=_env("RAGEVAL_LLM_BACKEND", "auto").lower(),
            model=_env("RAGEVAL_MODEL", "claude-opus-4-8"),
            vision_provider=_env("RAGEVAL_VISION_PROVIDER", "claude").lower(),
            ocr_provider=_env("RAGEVAL_OCR_PROVIDER", "tesseract").lower(),
            # Empty → fall back to `model` (resolved in vision.py). A separate knob lets you pin a
            # cheaper/faster vision model without changing the generation/judge model.
            vision_model=_env("RAGEVAL_VISION_MODEL", ""),
            enrich_concurrency=max(1, _env_int("RAGEVAL_ENRICH_CONCURRENCY", 8)),
            pii_backend=_env("RAGEVAL_PII_BACKEND", "regex").lower(),
            pii_spacy_model=_env("RAGEVAL_PII_SPACY_MODEL", "en_core_web_sm"),
            embeddings=embeddings,
            embed_model=_env(
                "RAGEVAL_EMBED_MODEL",
                # Default = all-mpnet-base-v2 (local, 768-dim): clearly better retrieval than
                # all-MiniLM-L6-v2 (384-dim) at the cost of ~2x dims/time — worth it on this
                # corpus, and LOCAL keeps the private content on-device (no third-party send).
                # MiniLM remains the A/B baseline: RAGEVAL_EMBED_MODEL=all-MiniLM-L6-v2 RAGEVAL_EMBED_DIM=384.
                "all-mpnet-base-v2" if is_local else "text-embedding-3-small",
            ),
            # MUST match the model: 768 = all-mpnet-base-v2; 384 = all-MiniLM-L6-v2;
            # 1536 = text-embedding-3-small. The Qdrant collection is created with exactly this size.
            embed_dim=_env_int("RAGEVAL_EMBED_DIM", 768 if is_local else 1536),
            collection_override=_env("RAGEVAL_COLLECTION", ""),
            qdrant_url=_env("RAGEVAL_QDRANT_URL", "http://localhost:6333"),
            hnsw_m=_env_int("RAGEVAL_HNSW_M", 16),
            hnsw_ef_construct=_env_int("RAGEVAL_HNSW_EF_CONSTRUCT", 100),
            hnsw_ef_search=_env_int("RAGEVAL_HNSW_EF_SEARCH", 64),
            top_k=_env_int("RAGEVAL_TOP_K", 5),
            candidate_k=_env_int("RAGEVAL_CANDIDATE_K", 20),
            rrf_k=_env_int("RAGEVAL_RRF_K", 60),
            use_rerank=_env("RAGEVAL_USE_RERANK", "true").lower() in ("1", "true", "yes"),
            rerank_model=_env("RAGEVAL_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            min_rerank_score=_env_float_opt("RAGEVAL_MIN_RERANK_SCORE"),  # None = floor disabled
            chunk_size=_env_int("RAGEVAL_CHUNK_SIZE", 800),
            chunk_overlap=_env_int("RAGEVAL_CHUNK_OVERLAP", 150),
            guard_input_scan=_flag("RAGEVAL_GUARD_INPUT_SCAN", True),
            guard_normalize=_flag("RAGEVAL_GUARD_NORMALIZE", True),
            guard_spotlight=_flag("RAGEVAL_GUARD_SPOTLIGHT", True),
            guard_output_validate=_flag("RAGEVAL_GUARD_OUTPUT_VALIDATE", True),
            guard_quarantine=_flag("RAGEVAL_GUARD_QUARANTINE", True),
            guard_llm_classifier=_flag("RAGEVAL_GUARD_LLM_CLASSIFIER", False),
        )


# A module-level default instance for convenience. Tests build their own with
# overridden fields via dataclasses.replace().
SETTINGS = Settings.load()
