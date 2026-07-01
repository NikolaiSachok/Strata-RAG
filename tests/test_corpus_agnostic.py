"""Phase-4 proof: a SYNTHETIC SECOND corpus works with ZERO engine-core changes (#36/#37/#38).

This is the load-bearing corpus-agnosticism test. It defines a completely fictional second corpus
shape — a DIFFERENT descriptor schema with GENUINELY NON-APP fields, DIFFERENT filename
conventions, and an EXTRA query class — entirely from the PUBLIC adapter contract, and proves each
of the three decoupled seams works without touching any core (non-adapter) module:

  #36  structured-fact harvest + schema-agnostic store — the adapter DECLARES non-app facets
                                  (`premium` int, `cause_of_loss` text) and a secret-ish
                                  `national_id`, its `harvest_facts` lifts them from a different
                                  descriptor schema, and they are STORED in the generic facet store
                                  and RETRIEVABLE via aggregation/lookup — with NO new field name in
                                  any core module. (The old test only remapped keys onto the same 3
                                  app columns, so it never exercised a non-app field — that's the
                                  gap this replaces.)
  #37  classification policy    — the adapter DECLARES a per-corpus allow_ext (`rst`) + FileRules
                                  (drop / metadata-only by filename), applied via the generic core
                                  mechanism.
  #38  query-class registry     — the corpus registers an extra `multi_hop` class with its own
                                  deterministic detector + executor, routed deterministic-first.

Everything here is fictional (a `covercorp` insurance-shaped corpus); no real names/ids appear.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rageval import aggregate, query_classes as qcmod
from rageval.classify import CorpusRules, PolicyResolver, classify, partition
from rageval.dispatch import dispatch
from rageval.facts import FacetSpec, FieldWhitelistHarvester, StructuredFact
from rageval.generate import Answer
from rageval.sidecar import ProjectRecord, all_projects, connect, upsert_project
from rageval.sources import registry
from rageval.sources.base import ClassificationPolicy, FileRule, SourceAdapter, SourceDoc


# ===========================================================================
# A fictional SECOND corpus with GENUINELY NON-APP structured facts.
# ===========================================================================

# A DIFFERENT descriptor schema than the sample's config.yaml: a claim manifest with non-app keys
# plus a secret-shaped `national_id` and an `api_token` secret block that must NEVER be lifted.
_CLAIM_WHITELIST = {"premium": "premium", "loss": "cause_of_loss", "insured": "insured_name"}
_CLAIM_HARVESTER = FieldWhitelistHarvester(_CLAIM_WHITELIST)


class CoverCorpAdapter(SourceAdapter):
    """A second-corpus adapter with NON-APP declared facets, its own descriptor schema + policy."""

    source_set = "covercorp"

    # Per-project descriptor data (inline for the test; a real adapter reads a file). Two projects
    # so an aggregation (group_by / count) has something to group.
    _DESCRIPTORS = {
        "c1": {"premium": 1200, "loss": "burglary", "insured": "Ada Byte",
               "api_token": "SECRET_never_lift", "national_id": "must-not-store"},
        "c2": {"premium": 800, "loss": "flood", "insured": "Grace Stack",
               "api_token": "SECRET_never_lift"},
    }

    def discover(self):
        for pid in self._DESCRIPTORS:
            yield SourceDoc(project_id=pid, source_set=self.source_set,
                            doc_path=self.root / pid / "summary.rst",
                            doc_type="description", ext="rst",
                            raw_text=f"A fictional {pid} claim summary " * 5)
            yield SourceDoc(project_id=pid, source_set=self.source_set,
                            doc_path=self.root / pid / "buildlog.rst",
                            doc_type="other", ext="rst",
                            raw_text="internal processing log noise " * 5)

    def declared_facets(self):
        # GENUINELY non-app facets, with types — the positive, fail-closed allowlist. Note NO
        # `national_id` facet is declared, so even if harvested it can never be stored (fail-closed).
        return (
            FacetSpec("premium", "int", "the claim premium amount"),
            FacetSpec("cause_of_loss", "text", "the peril category"),
            FacetSpec("insured_name", "text", "the insured party"),
        )

    def harvest_facts(self, project_id: str, project_dir: Path):
        src = self._DESCRIPTORS.get(project_id, {})
        fields = _CLAIM_HARVESTER.lift(src)
        # premium must survive as an int through the whitelist primitive (it coerces to str for the
        # generic lift, but the declared facet type re-coerces at store). Re-read raw for the int.
        for field_name, value in fields.items():
            raw = src.get(next(k for k, v in _CLAIM_WHITELIST.items() if v == field_name))
            yield StructuredFact(project_id, field_name, raw if raw is not None else value,
                                 provenance="descriptor")

    def classification_policy(self) -> ClassificationPolicy:
        return ClassificationPolicy(
            allow_ext=frozenset({"rst"}),
            file_rules=(
                FileRule(action="drop", name="buildlog.rst", reason="covercorp log (noise)"),
            ),
        )


# --- an EXTRA query class: multi_hop (cross-document join) ---------------------------------------

def _multi_hop_detect(question: str):
    if "compare" in question.lower() and "across" in question.lower():
        return (0.95, {"hops": 2})
    return None


def _multi_hop_execute(question: str, decision, pipeline):
    return Answer(question=question, answer="multi-hop covercorp answer", sources=[], chunks=[],
                  routing={"executed_route": "multi_hop", "hops": decision.slots.get("hops")})


@pytest.fixture
def cover_corpus(tmp_path):
    """Register the second corpus's adapter + query class. The autouse _reset_registries fixture
    (conftest) restores the global registries after the test, so no snapshot/restore is needed
    here. Lays out a matching corpus root."""
    registry.register_adapter("covercorp", CoverCorpAdapter)
    qcmod.register_query_class(qcmod.QueryClass(
        name="multi_hop", detect=_multi_hop_detect, execute=_multi_hop_execute,
        describe="cross-document join across multiple projects"))
    root = tmp_path / "corpus"
    (root / "covercorp" / "c1").mkdir(parents=True)
    (root / "covercorp" / "c2").mkdir(parents=True)
    yield root


def _enrich_and_store(db: Path):
    """Enrich covercorp's projects (no LLM) and write them to a sidecar at `db`. Returns records."""
    from rageval.enrich import enrich_project

    adapter = CoverCorpAdapter(Path("corpus") / "covercorp")
    docs = list(adapter.discover())
    conn = connect(db)
    recs = []
    for pid in ("c1", "c2"):
        pdocs = [d for d in docs if d.project_id == pid]
        # Point one doc under an on-disk project dir so enrich recovers it for harvest_facts.
        rec = enrich_project(None, "covercorp", pid, pdocs, chunk_count=1)
        upsert_project(conn, rec)
        recs.append(rec)
    conn.close()
    return recs


# ===========================================================================
# #36 — NON-APP facts are STORED and RETRIEVABLE with no core field name.
# ===========================================================================

def test_second_corpus_nonapp_facts_are_queryable_via_aggregate(cover_corpus, tmp_path):
    """The REAL #36 proof: a genuinely non-app facet (`premium` int, `cause_of_loss` text) round-
    trips into the schema-agnostic store and is QUERYABLE via aggregation/lookup — with NO field
    name added to any core module."""
    db = tmp_path / "side.sqlite"
    _enrich_and_store(db)

    # It's a queryable field now (declared facet), NOT a hardcoded column.
    assert "premium" in aggregate.allowed_fields()
    assert "cause_of_loss" in aggregate.allowed_fields()

    # group_by over a non-app facet works.
    res = aggregate.execute("group_by_count", field="cause_of_loss", sidecar_path=db)
    counts = {r["cause_of_loss"]: r["count"] for r in res.rows}
    assert counts == {"burglary": 1, "flood": 1}

    # list distinct over a non-app facet works.
    res = aggregate.execute("list", field="cause_of_loss", sidecar_path=db)
    assert [r["cause_of_loss"] for r in res.rows] == ["burglary", "flood"]

    # filter on a non-app facet works (premium stored as int → equality by its text form).
    res = aggregate.execute("count", filter={"cause_of_loss": "flood"}, sidecar_path=db)
    assert res.rows[0]["count"] == 1

    # lookup a single project's non-app facet.
    res = aggregate.execute("lookup", field="premium", filter={"key": "covercorp/c1"},
                            sidecar_path=db)
    assert res.rows[0]["premium"] == "1200"


def test_second_corpus_facet_roundtrips_through_record(cover_corpus, tmp_path):
    db = tmp_path / "side.sqlite"
    _enrich_and_store(db)
    conn = connect(db)
    recs = {r.key: r for r in all_projects(conn)}
    conn.close()
    assert recs["covercorp/c1"].fact("cause_of_loss") == "burglary"
    assert recs["covercorp/c1"].fact("premium") == 1200          # decoded back to int
    assert recs["covercorp/c1"].fact("insured_name") == "Ada Byte"


def test_second_corpus_undeclared_and_secret_facts_are_never_stored(cover_corpus, tmp_path):
    """Fail-closed: `national_id` is present in the descriptor but NOT declared → never stored; the
    secret `api_token` is neither whitelisted nor declared → never stored."""
    db = tmp_path / "side.sqlite"
    _enrich_and_store(db)
    conn = connect(db)
    recs = {r.key: r for r in all_projects(conn)}
    conn.close()
    c1 = recs["covercorp/c1"]
    assert "national_id" not in c1.facts        # undeclared → fail-closed
    assert "api_token" not in c1.facts          # secret → never lifted
    blob = repr(c1.facts).lower()
    assert "must-not-store" not in blob and "secret" not in blob
    # national_id is not even a queryable field.
    assert "national_id" not in aggregate.allowed_fields()


def test_undeclared_facet_is_not_queryable(cover_corpus):
    """A field that isn't a generic column or a declared facet is rejected by the aggregate guard."""
    with pytest.raises(aggregate.AggregateError):
        aggregate.execute("list", field="national_id")


# ===========================================================================
# MAJOR — a MISTYPED fact degrades ONE facet; the ingest batch CONTINUES.
# ===========================================================================

def test_mistyped_fact_is_skipped_and_batch_continues(cover_corpus, tmp_path):
    """A custom adapter emitting a mistyped fact (premium declared int, value 'oops') must NOT abort
    the whole ingest: that ONE facet is skipped, the record's other facets persist, and the batch
    completes."""
    db = tmp_path / "side.sqlite"
    conn = connect(db)
    rec = ProjectRecord(project_id="c9", source_set="covercorp", chunk_count=1)
    rec.facts = {"premium": "oops-not-an-int", "cause_of_loss": "fire"}
    rec.facts_provenance = {"premium": "descriptor", "cause_of_loss": "descriptor"}
    upsert_project(conn, rec)  # MUST NOT raise despite the mistyped premium
    # A second project still writes fine (batch continues).
    rec2 = ProjectRecord(project_id="c10", source_set="covercorp", chunk_count=1)
    rec2.facts = {"premium": 500, "cause_of_loss": "theft"}
    upsert_project(conn, rec2)
    got = {r.key: r for r in all_projects(conn)}
    conn.close()
    # The mistyped premium was skipped; the good facet on the same record survived.
    assert "premium" not in got["covercorp/c9"].facts
    assert got["covercorp/c9"].fact("cause_of_loss") == "fire"
    # The other project is fully intact.
    assert got["covercorp/c10"].fact("premium") == 500


# ===========================================================================
# #37 — per-corpus allow_ext + FileRules classify a different corpus shape.
# ===========================================================================

def test_second_corpus_allow_ext_and_file_rules_no_core_change(cover_corpus):
    rules = CorpusRules.load()
    docs = list(CoverCorpAdapter(cover_corpus / "covercorp").discover())
    policy = CoverCorpAdapter(cover_corpus / "covercorp").classification_policy()
    by = {(d.project_id, d.doc_path.name): d for d in docs}
    # .rst content is INCLUDED only because THIS corpus declared `rst` in allow_ext.
    ok = classify(by[("c1", "summary.rst")], rules, policy)
    assert ok.include and ok.reason == "ok"
    # A FileRule drops the corpus's build-log noise.
    bl = classify(by[("c1", "buildlog.rst")], rules, policy)
    assert not bl.include and "log" in bl.reason


def test_second_corpus_allow_ext_does_not_leak_into_another_corpus(cover_corpus):
    """`allow_ext` is per-corpus: covercorp declaring `rst` must NOT make .rst allowed for a corpus
    that didn't declare it (the sample's northwind)."""
    rules = CorpusRules.load()
    other = SourceDoc(project_id="0001", source_set="northwind",
                      doc_path=Path("northwind") / "0001" / "notes.rst",
                      doc_type="description", ext="rst", raw_text="x" * 200)
    dec = classify(other, rules, PolicyResolver().policy_for("northwind"))
    assert not dec.include and "ext not allowed" in dec.reason


def test_second_corpus_partition_end_to_end(cover_corpus):
    docs = list(CoverCorpAdapter(cover_corpus / "covercorp").discover())
    included, excluded = partition(docs, CorpusRules.load())
    inc = {d.doc_path.name for d, _ in included}
    exc = {d.doc_path.name for d, _ in excluded}
    assert "summary.rst" in inc       # declared ext
    assert "buildlog.rst" in exc      # dropped by file rule


# ===========================================================================
# #38 — a registered extra query class is detected + executed by the router.
# ===========================================================================

class _StubPipeline:
    def __init__(self, llm=None):
        self.llm = llm
        self.semantic_calls = 0

    def answer(self, question: str) -> Answer:
        self.semantic_calls += 1
        return Answer(question=question, answer="semantic", sources=[], chunks=[])


def test_registered_query_class_is_detected_deterministically(cover_corpus):
    from rageval import router

    d = router.route("compare premiums across claims", llm=None)
    assert d.route == "multi_hop"
    assert d.method == "rule"
    assert d.slots.get("hops") == 2


def test_registered_query_class_is_dispatched_and_executed(cover_corpus):
    ans = dispatch("compare premiums across claims", _StubPipeline(), use_rules=True)
    assert ans.answer == "multi-hop covercorp answer"
    assert ans.routing["executed_route"] == "multi_hop"


def test_registered_query_class_in_valid_routes(cover_corpus):
    assert "multi_hop" in qcmod.valid_routes()
    for generic in ("semantic", "aggregation", "lookup", "hybrid"):
        assert generic in qcmod.valid_routes()


def test_register_query_class_rejects_generic_name_collision():
    with pytest.raises(ValueError):
        qcmod.register_query_class(qcmod.QueryClass(name="semantic"))
