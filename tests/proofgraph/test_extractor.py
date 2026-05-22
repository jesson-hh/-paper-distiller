"""Tests for proofgraph.extractor — extract_segment (grounding gate) and self_check."""
from __future__ import annotations
import json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEG_TEXT = (
    "Theorem 4.3. For all f in the function class, ||f||_2 <= C n^{-1/2}. "
    "Proof. By Bernstein's inequality we bound the tail probability. "
    "Applying Dudley chaining to the empirical process yields the claim. □"
)


def _make_segment(text=SEG_TEXT):
    """Build a Segment directly (avoid importing segment() to keep tests fast)."""
    from paper_distiller.proofgraph.reader import Segment
    return Segment(
        id=0,
        kind_hint="proof",
        section="2 Main Result",
        text=text,
        char_start=0,
        char_end=len(text),
        is_proof_block=True,
    )


def _memory():
    from paper_distiller.proofgraph.memory import RunningMemory
    return RunningMemory()


# ---------------------------------------------------------------------------
# Task 3.4 — extract_segment with grounding gate
# ---------------------------------------------------------------------------

GROUNDED_QUOTE = "By Bernstein's inequality we bound the tail probability."
FABRICATED_QUOTE = "By the Central Limit Theorem we prove normality of residuals."


class _SingleCallLLM:
    """Mock LLM returning a fixed canned response on each call."""
    def __init__(self, response: str):
        self._response = response
        self.call_count = 0

    def complete(self, messages, temperature=0.2, response_format=None):
        self.call_count += 1
        return self._response


class _TwoNodeLLM:
    """Returns two nodes: one grounded (GROUNDED_QUOTE), one fabricated."""
    def __init__(self):
        self.call_count = 0

    def complete(self, messages, temperature=0.2, response_format=None):
        self.call_count += 1
        return json.dumps({"nodes": [
            {
                "kind": "proof_step",
                "text": "Bound tail probability via Bernstein",
                "source_quote": GROUNDED_QUOTE,
                "techniques": ["Bernstein"],
                "refs": [],
            },
            {
                "kind": "proof_step",
                "text": "Normality of residuals",
                "source_quote": FABRICATED_QUOTE,
                "techniques": [],
                "refs": [],
            },
        ]})


def test_extract_segment_keeps_grounded_drops_fabricated():
    """The grounding gate must admit the node with the verbatim quote and
    reject (or mark unsupported) the node with a fabricated quote."""
    from paper_distiller.proofgraph.extractor import extract_segment
    seg = _make_segment()
    llm = _TwoNodeLLM()
    accepted = extract_segment(seg, _memory(), llm)
    # Only the grounded node should be returned
    assert len(accepted) == 1
    assert accepted[0].source_quote == GROUNDED_QUOTE
    assert accepted[0].status != "unsupported"


def test_extract_segment_fabricated_node_not_in_accepted():
    """The fabricated node must not appear in the accepted list."""
    from paper_distiller.proofgraph.extractor import extract_segment
    seg = _make_segment()
    llm = _TwoNodeLLM()
    accepted = extract_segment(seg, _memory(), llm)
    quotes = [n.source_quote for n in accepted]
    assert FABRICATED_QUOTE not in quotes


def test_extract_segment_retries_on_failed_gate():
    """When the first LLM call returns a fabricated quote, a retry call is made.
    After two calls still failing, the node is dropped."""
    from paper_distiller.proofgraph.extractor import extract_segment

    # Both calls return only the fabricated node — should be dropped after retry
    llm = _SingleCallLLM(json.dumps({"nodes": [{
        "kind": "proof_step",
        "text": "Made up claim",
        "source_quote": FABRICATED_QUOTE,
    }]}))
    seg = _make_segment()
    accepted = extract_segment(seg, _memory(), llm)
    assert accepted == []
    # Should have made 2 calls: initial + one retry
    assert llm.call_count == 2


def test_extract_segment_all_grounded_all_accepted():
    """When all nodes have valid verbatim quotes, all are accepted."""
    from paper_distiller.proofgraph.extractor import extract_segment

    q1 = "By Bernstein's inequality we bound the tail probability."
    q2 = "Applying Dudley chaining to the empirical process yields the claim."
    llm = _SingleCallLLM(json.dumps({"nodes": [
        {"kind": "proof_step", "text": "step1", "source_quote": q1},
        {"kind": "proof_step", "text": "step2", "source_quote": q2},
    ]}))
    seg = _make_segment()
    accepted = extract_segment(seg, _memory(), llm)
    assert len(accepted) == 2


def test_extract_segment_empty_llm_response_returns_empty():
    """Garbled/empty LLM output produces an empty list (no crash)."""
    from paper_distiller.proofgraph.extractor import extract_segment
    llm = _SingleCallLLM("garbage not json")
    seg = _make_segment()
    accepted = extract_segment(seg, _memory(), llm)
    assert accepted == []


def test_extract_segment_returns_extracted_node_objects():
    """Accepted nodes are ExtractedNode instances with the right status."""
    from paper_distiller.proofgraph.extractor import extract_segment
    from paper_distiller.proofgraph.extraction_schema import ExtractedNode
    llm = _SingleCallLLM(json.dumps({"nodes": [{
        "kind": "proof_step",
        "text": "Bernstein tail bound",
        "source_quote": GROUNDED_QUOTE,
        "techniques": ["Bernstein"],
    }]}))
    seg = _make_segment()
    accepted = extract_segment(seg, _memory(), llm)
    assert len(accepted) == 1
    assert isinstance(accepted[0], ExtractedNode)
    assert accepted[0].status == "extracted"
    assert "Bernstein" in accepted[0].techniques


# ---------------------------------------------------------------------------
# Task 3.5 — self_check
# ---------------------------------------------------------------------------

class _SuspiciousLabelLLM:
    """Returns a self-check verdict marking 'Step 2' as suspicious."""
    def __init__(self):
        self.call_count = 0

    def complete(self, messages, temperature=0.2, response_format=None):
        self.call_count += 1
        return json.dumps({"suspicious_labels": ["Step 2"]})


class _EmptyVerdictLLM:
    """Returns an empty/garbled self-check verdict."""
    def __init__(self, response="{}"):
        self._response = response
        self.call_count = 0

    def complete(self, messages, temperature=0.2, response_format=None):
        self.call_count += 1
        return self._response


def _make_nodes_with_labels():
    from paper_distiller.proofgraph.extraction_schema import ExtractedNode
    return [
        ExtractedNode(kind="proof_step", text="step1", source_quote="By Bernstein", label="Step 1"),
        ExtractedNode(kind="proof_step", text="step2", source_quote="Applying Dudley chaining to the empirical process", label="Step 2"),
    ]


def test_self_check_marks_suspicious_node():
    from paper_distiller.proofgraph.extractor import self_check
    seg = _make_segment()
    nodes = _make_nodes_with_labels()
    llm = _SuspiciousLabelLLM()
    result = self_check(seg, nodes, llm)
    labels_suspicious = [n.label for n in result if n.status == "suspicious"]
    assert "Step 2" in labels_suspicious


def test_self_check_leaves_others_unchanged():
    from paper_distiller.proofgraph.extractor import self_check
    seg = _make_segment()
    nodes = _make_nodes_with_labels()
    llm = _SuspiciousLabelLLM()
    result = self_check(seg, nodes, llm)
    step1 = next(n for n in result if n.label == "Step 1")
    assert step1.status != "suspicious"


def test_self_check_empty_verdict_no_crash():
    from paper_distiller.proofgraph.extractor import self_check
    seg = _make_segment()
    nodes = _make_nodes_with_labels()
    llm = _EmptyVerdictLLM("{}")
    result = self_check(seg, nodes, llm)
    assert all(n.status != "suspicious" for n in result)


def test_self_check_garbled_verdict_no_crash():
    from paper_distiller.proofgraph.extractor import self_check
    seg = _make_segment()
    nodes = _make_nodes_with_labels()
    llm = _EmptyVerdictLLM("not json")
    result = self_check(seg, nodes, llm)
    # No crash, nodes unchanged
    assert len(result) == 2
    assert all(n.status == "extracted" for n in result)
