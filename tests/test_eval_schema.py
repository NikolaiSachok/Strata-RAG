"""Deterministic tests for the eval layer's parsing, gating, and output schema.

The LLM judge itself is non-deterministic, but everything AROUND it is pure and must
be reliable: parsing the judge's JSON, repairing malformed fields, the gate decision,
and the exact output schema the API returns. We test those with canned judge strings —
no LLM call — so the contract is locked down.
"""

from __future__ import annotations

import pytest

from rageval.eval import (
    Dimension,
    compute_gate,
    parse_eval,
)

GOOD_JUDGE_OUTPUT = """
{
  "faithfulness": {"score": 5, "severity": "none", "reason": "All claims supported."},
  "answer_relevance": {"score": 4, "severity": "minor", "reason": "Mostly on point."},
  "findings": ["nit: could cite passage 2"]
}
"""

FENCED_JUDGE_OUTPUT = """Here is my verdict:
```json
{"faithfulness": {"score": 1, "severity": "critical", "reason": "Hallucinated a fact."},
 "answer_relevance": {"score": 5, "severity": "none", "reason": "On topic."},
 "findings": []}
```
thanks!"""


def test_parse_good_output_schema():
    result = parse_eval(GOOD_JUDGE_OUTPUT)
    d = result.to_dict()
    # Exact top-level schema the API promises.
    assert set(d.keys()) == {"faithfulness", "answer_relevance", "overall_pass", "findings"}
    for dim in ("faithfulness", "answer_relevance"):
        assert set(d[dim].keys()) == {"score", "severity", "reason"}
    assert isinstance(d["overall_pass"], bool)
    assert isinstance(d["findings"], list)


def test_good_output_passes_gate():
    # none + minor are both below the "major" gate → pass.
    assert parse_eval(GOOD_JUDGE_OUTPUT).overall_pass is True


def test_fenced_output_is_parsed_and_fails_gate():
    # A critical faithfulness severity must fail the gate even though relevance is fine.
    result = parse_eval(FENCED_JUDGE_OUTPUT)
    assert result.faithfulness.severity == "critical"
    assert result.overall_pass is False


def test_score_is_clamped_to_1_5():
    raw = (
        '{"faithfulness": {"score": 9, "severity": "none", "reason": "x"},'
        ' "answer_relevance": {"score": 0, "severity": "none", "reason": "y"},'
        ' "findings": []}'
    )
    result = parse_eval(raw)
    assert result.faithfulness.score == 5
    assert result.answer_relevance.score == 1


def test_unknown_severity_is_repaired_from_score():
    raw = (
        '{"faithfulness": {"score": 2, "severity": "weird", "reason": "x"},'
        ' "answer_relevance": {"score": 5, "severity": "none", "reason": "y"},'
        ' "findings": []}'
    )
    result = parse_eval(raw)
    # score 2 → default "major"; that should fail the default gate.
    assert result.faithfulness.severity == "major"
    assert result.overall_pass is False


def test_gate_logic_is_pure():
    none = Dimension(5, "none", "")
    major = Dimension(2, "major", "")
    assert compute_gate(none, none) is True
    assert compute_gate(none, major) is False
    assert compute_gate(major, none) is False
    # custom stricter threshold: fail on "minor" or worse.
    minor = Dimension(4, "minor", "")
    assert compute_gate(none, minor, threshold="minor") is False


def test_no_json_raises():
    with pytest.raises(ValueError):
        parse_eval("the model refused and wrote only prose")
