"""Phase-4 proof: a SYNTHETIC SECOND corpus works with ZERO engine-core changes (#36/#37/#38).

This is the load-bearing corpus-agnosticism test. It defines a completely fictional second corpus
shape — a DIFFERENT descriptor schema, DIFFERENT filename conventions, and an EXTRA query class —
entirely from the PUBLIC adapter contract, and proves each of the three decoupled seams works
without touching any core module:

  #36  structured-fact harvest  — the adapter's `harvest_facts` lifts a different descriptor
                                  schema (a `manifest.yaml` with `product:`/`site:` keys) into the
                                  sidecar, whitelisted + fail-closed on secrets, with no core edit.
  #37  classification policy    — the adapter DECLARES a per-corpus allow_ext (`rst`) + FileRules
                                  (drop / metadata-only / retype by filename), and the classifier
                                  applies them via its generic mechanism.
  #38  query-class registry     — the corpus registers an extra `multi_hop` class with its own
                                  deterministic detector + executor, and the router/dispatch route
                                  to it — deterministic-first, no core intent string added.

Everything here is fictional (a `widgetco` corpus); no real names/ids/paths appear.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rageval import query_classes as qcmod
from rageval.classify import CorpusRules, PolicyResolver, classify, partition
from rageval.dispatch import dispatch
from rageval.facts import FieldWhitelistHarvester, StructuredFact
from rageval.generate import Answer
from rageval.sources import registry
from rageval.sources.base import ClassificationPolicy, FileRule, SourceAdapter, SourceDoc


# ===========================================================================
# A fictional SECOND corpus, defined purely from the public contract.
# ===========================================================================

# A DIFFERENT descriptor schema than the sample's config.yaml: a `manifest.yaml` with product/site
# blocks and a secret block that must NEVER be lifted.
_WIDGET_MANIFEST_WHITELIST = {"title": "app_name", "host": "domain", "sku": "app_number"}
_WIDGET_HARVESTER = FieldWhitelistHarvester(_WIDGET_MANIFEST_WHITELIST)


class WidgetCoAdapter(SourceAdapter):
    """A second-corpus adapter with its OWN descriptor schema, filename conventions, and policy."""

    source_set = "widgetco"

    def discover(self):
        # One fictional project with a couple of docs; discovery just yields candidates.
        yield SourceDoc(project_id="w1", source_set=self.source_set,
                        doc_path=self.root / "w1" / "overview.rst",
                        doc_type="description", ext="rst",
                        raw_text="A fictional widget product overview " * 5)
        yield SourceDoc(project_id="w1", source_set=self.source_set,
                        doc_path=self.root / "w1" / "buildlog.rst",
                        doc_type="other", ext="rst",
                        raw_text="internal build log noise " * 5)
        yield SourceDoc(project_id="w1", source_set=self.source_set,
                        doc_path=self.root / "w1" / "facts.yaml",
                        doc_type="other", ext="yaml",
                        raw_text="structured metadata " * 5)

    def harvest_facts(self, project_id: str, project_dir: Path):
        # A DIFFERENT descriptor schema (product/site) lifted via the SAME core primitive.
        source = {"title": "Widget Pro", "host": "widget.test", "sku": "42",
                  "api_key": "SECRET_never_lift"}
        fields = _WIDGET_HARVESTER.lift({**source.get("product", {}), **source})
        for field_name, value in fields.items():
            yield StructuredFact(project_id, field_name, value, provenance="descriptor")
        if fields.get("domain"):
            yield StructuredFact(project_id, "landing_url",
                                 f"https://{fields['domain']}", provenance="derived")

    def classification_policy(self) -> ClassificationPolicy:
        # DIFFERENT conventions: this corpus writes .rst content (not md/txt), a `buildlog.rst`
        # that is noise (drop), and a `facts.yaml` that is metadata-only (enrich, not embedded).
        return ClassificationPolicy(
            allow_ext=frozenset({"rst"}),
            file_rules=(
                FileRule(action="drop", name="buildlog.rst", reason="widgetco build log (noise)"),
                FileRule(action="metadata_only", name="facts.yaml", doc_type="metadata",
                         reason="widgetco structured metadata (enrich-only)"),
            ),
        )


# --- an EXTRA query class: multi_hop (cross-document join) ---------------------------------------

def _multi_hop_detect(question: str):
    """Deterministic detector: a phrasing that clearly asks for a cross-document hop."""
    if "compare" in question.lower() and "across" in question.lower():
        return (0.95, {"hops": 2})
    return None


def _multi_hop_execute(question: str, decision, pipeline):
    """Execute the extra class: here a trivial fictional two-step that returns a labelled Answer."""
    return Answer(question=question, answer="multi-hop widgetco answer", sources=[], chunks=[],
                  routing={"executed_route": "multi_hop", "hops": decision.slots.get("hops")})


@pytest.fixture
def widget_corpus(tmp_path, monkeypatch):
    """Register the second corpus's adapter + query class, snapshot/restore the global registries
    so no other test is affected, and lay out a matching corpus root."""
    saved_adapters = dict(registry.ADAPTER_BY_FOLDER)
    saved_qc = qcmod.registered_classes()
    registry.register_adapter("widgetco", WidgetCoAdapter)
    qcmod.register_query_class(qcmod.QueryClass(
        name="multi_hop", detect=_multi_hop_detect, execute=_multi_hop_execute,
        describe="cross-document join across multiple projects"))
    root = tmp_path / "corpus"
    (root / "widgetco" / "w1").mkdir(parents=True)
    try:
        yield root
    finally:
        registry.ADAPTER_BY_FOLDER.clear()
        registry.ADAPTER_BY_FOLDER.update(saved_adapters)
        qcmod._REGISTRY.clear()
        qcmod._REGISTRY.update(saved_qc)


# ===========================================================================
# #36 — a DIFFERENT descriptor schema feeds the sidecar, secrets excluded.
# ===========================================================================

def test_second_corpus_structured_facts_no_core_change(widget_corpus):
    facts = list(WidgetCoAdapter(widget_corpus / "widgetco").harvest_facts(
        "w1", widget_corpus / "widgetco" / "w1"))
    d = {f.field: f.value for f in facts}
    # A totally different descriptor schema maps onto the generic sidecar slots.
    assert d["app_name"] == "Widget Pro"
    assert d["domain"] == "widget.test"
    assert d["app_number"] == "42"
    assert d["landing_url"] == "https://widget.test"
    # The whitelist/fail-closed invariant holds for the new schema: no secret is ever lifted.
    blob = repr(facts).lower()
    assert "secret" not in blob and "api_key" not in d


def test_second_corpus_facts_reach_the_record_generically(widget_corpus):
    """The core enrich path consumes the adapter's facts by generic field name — no core edit."""
    from rageval.enrich import enrich_project

    docs = list(WidgetCoAdapter(widget_corpus / "widgetco").discover())
    # Point a doc_path under the on-disk project so enrich can recover the project dir.
    rec = enrich_project(None, "widgetco", "w1", docs, chunk_count=1)
    # NB the harvester ignores the on-disk path (fixture schema is inline), so facts always apply.
    assert rec.app_name == "Widget Pro"
    assert rec.domain == "widget.test"
    assert rec.landing_url == "https://widget.test"


# ===========================================================================
# #37 — per-corpus allow_ext + FileRules classify a different corpus shape.
# ===========================================================================

def test_second_corpus_allow_ext_and_file_rules_no_core_change(widget_corpus):
    rules = CorpusRules.load()
    docs = list(WidgetCoAdapter(widget_corpus / "widgetco").discover())
    policy = WidgetCoAdapter(widget_corpus / "widgetco").classification_policy()

    by_name = {d.doc_path.name: d for d in docs}
    # .rst content is INCLUDED only because THIS corpus declared `rst` in allow_ext (the shared
    # corpus-rules.yaml baseline is md/txt/docx and would otherwise drop it).
    ov = classify(by_name["overview.rst"], rules, policy)
    assert ov.include and ov.reason == "ok"
    # A FileRule drops the corpus's build-log noise...
    bl = classify(by_name["buildlog.rst"], rules, policy)
    assert not bl.include and "build log" in bl.reason
    # ...and a FileRule marks facts.yaml metadata-only (INCLUDE-but-enrich-only, not embedded).
    fy = classify(by_name["facts.yaml"], rules, policy)
    assert fy.include and fy.metadata_only


def test_second_corpus_allow_ext_does_not_leak_into_another_corpus(widget_corpus):
    """`allow_ext` is per-corpus: widgetco declaring `rst` must NOT make .rst allowed for a corpus
    that didn't declare it (e.g. the sample's northwind). Proves isolation."""
    rules = CorpusRules.load()
    other = SourceDoc(project_id="0001", source_set="northwind",
                      doc_path=Path("northwind") / "0001" / "notes.rst",
                      doc_type="description", ext="rst", raw_text="x" * 200)
    # Resolve northwind's OWN policy (does not include rst) → the doc is dropped.
    resolver = PolicyResolver()
    dec = classify(other, rules, resolver.policy_for("northwind"))
    assert not dec.include and "ext not allowed" in dec.reason


def test_second_corpus_partition_end_to_end(widget_corpus):
    """partition() resolves widgetco's policy by source_set automatically (no core change)."""
    docs = list(WidgetCoAdapter(widget_corpus / "widgetco").discover())
    included, excluded = partition(docs, CorpusRules.load())
    inc = {d.doc_path.name for d, _ in included}
    exc = {d.doc_path.name for d, _ in excluded}
    assert "overview.rst" in inc          # declared ext
    assert "facts.yaml" in inc            # metadata-only is INCLUDED
    assert "buildlog.rst" in exc          # dropped by file rule


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


def test_registered_query_class_is_detected_deterministically(widget_corpus):
    from rageval import router

    # Deterministic-first: the registered multi_hop detector fires BEFORE any LLM call.
    d = router.route("compare features across projects", llm=None)
    assert d.route == "multi_hop"
    assert d.method == "rule"
    assert d.slots.get("hops") == 2


def test_registered_query_class_is_dispatched_and_executed(widget_corpus):
    ans = dispatch("compare features across projects", _StubPipeline(), use_rules=True)
    assert ans.answer == "multi-hop widgetco answer"
    assert ans.routing["executed_route"] == "multi_hop"


def test_registered_query_class_in_valid_routes(widget_corpus):
    assert "multi_hop" in qcmod.valid_routes()
    # And the generic classes are still there (extras EXTEND, never replace).
    for generic in ("semantic", "aggregation", "lookup", "hybrid"):
        assert generic in qcmod.valid_routes()


def test_register_query_class_rejects_generic_name_collision():
    with pytest.raises(ValueError):
        qcmod.register_query_class(qcmod.QueryClass(name="semantic"))
