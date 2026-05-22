"""Tests for proofgraph.pipeline — build_graph_for_paper orchestration."""
from __future__ import annotations
import json

# Fabricated (non-verbatim) quotes for rejection tests
FABRICATED_QUOTE_A = "The proof uses Gaussian tail estimates and union bounds."
FABRICATED_QUOTE_B = "By the Central Limit Theorem we conclude normality."


# ---------------------------------------------------------------------------
# A tiny fake paper whose text contains a theorem statement and a proof.
# ---------------------------------------------------------------------------

FAKE_PAPER = """\
1 Introduction
We study a simple problem.

2 Main Result
Theorem 1. For all x, f(x) <= C.

Proof. By Bernstein's inequality we bound the tail probability directly.
This follows immediately from the bound. □

3 Discussion
Future work remains open.
"""

# Quote verbatim from the theorem segment
THEOREM_QUOTE = "For all x, f(x) <= C."
# Quote verbatim from the proof segment
PROOF_QUOTE = "By Bernstein's inequality we bound the tail probability directly."
PROOF_QUOTE2 = "This follows immediately from the bound."


class _DispatchLLMWithSelfCheck:
    """Mock LLM dispatching by content type.

    Self-check prompts contain "You are reviewing" + "suspicious_labels" in the
    template → return no-suspicious verdict.
    Extraction prompts for the proof segment contain both proof quotes verbatim
    in the segment_text block AND the is_proof_block hint.
    Extraction prompts for the theorem segment contain THEOREM_QUOTE.
    Everything else returns empty.
    """
    def __init__(self):
        self.call_count = 0
        self._theorem_extraction = json.dumps({"nodes": [{
            "kind": "theorem",
            "label": "Theorem 1",
            "text": "For all x, f(x) <= C.",
            "source_quote": THEOREM_QUOTE,
            "techniques": [],
            "refs": [],
        }]})
        self._no_suspicious = json.dumps({"suspicious_labels": []})
        self._proof_extraction = json.dumps({"nodes": [
            {
                "kind": "proof_step",
                "label": "Step 1",
                "text": "Bernstein tail bound",
                "source_quote": PROOF_QUOTE,
                "techniques": ["Bernstein"],
                "refs": [{"rel": "depends_on", "target": "Theorem 1"}],
            },
            {
                "kind": "proof_step",
                "label": "Step 2",
                "text": "Follows from bound",
                "source_quote": PROOF_QUOTE2,
                "techniques": [],
                "refs": [{"rel": "depends_on", "target": "Lemma 9"}],
            },
        ]})
        self._empty = json.dumps({"nodes": []})

    def complete(self, messages, temperature=0.2, response_format=None):
        content = messages[0]["content"] if messages else ""
        self.call_count += 1
        # Self-check prompts start with "You are reviewing extracted mathematical"
        if content.startswith("You are reviewing extracted mathematical"):
            return self._no_suspicious
        # Extraction prompt for proof segment: contains both proof quotes verbatim
        # (they appear in the segment_text block of the formatted prompt)
        if PROOF_QUOTE in content and PROOF_QUOTE2 in content:
            return self._proof_extraction
        # Extraction prompt for theorem segment
        if THEOREM_QUOTE in content and "Kind hint: theorem" in content:
            return self._theorem_extraction
        # Headings / discussion
        return self._empty


def test_build_graph_writes_nodes_and_edges(tmp_path):
    from paper_distiller.proofs.store import ProofStore
    from paper_distiller.proofgraph.pipeline import build_graph_for_paper, CoverageReport
    store = ProofStore(tmp_path / "proofs.db")
    llm = _DispatchLLMWithSelfCheck()
    report = build_graph_for_paper(
        store, "1234.5678", FAKE_PAPER,
        paper_slug="fake-paper", llm=llm,
    )
    assert isinstance(report, CoverageReport)
    nodes = store.nodes_by_paper("1234.5678")
    assert len(nodes) >= 2  # at least the theorem + proof steps


def test_build_graph_creates_depends_on_edge(tmp_path):
    from paper_distiller.proofs.store import ProofStore
    from paper_distiller.proofgraph.pipeline import build_graph_for_paper
    store = ProofStore(tmp_path / "proofs.db")
    llm = _DispatchLLMWithSelfCheck()
    build_graph_for_paper(store, "1234.5678", FAKE_PAPER, paper_slug="fp", llm=llm)

    nodes = store.nodes_by_paper("1234.5678")
    # Find Step 1 node
    step1 = next((n for n in nodes if n.label == "Step 1"), None)
    assert step1 is not None, f"Step 1 not found; nodes={[n.label for n in nodes]}"
    # Find Theorem 1 node
    thm = next((n for n in nodes if n.label == "Theorem 1"), None)
    assert thm is not None

    # The edge from Step 1 → Theorem 1 must exist
    edges = store.out_edges(step1.id, rel="depends_on")
    assert any(e.dst_id == thm.id for e in edges), (
        f"No depends_on edge from Step1 to Theorem1; edges={edges}"
    )


def test_build_graph_dangling_ref_becomes_gap(tmp_path):
    from paper_distiller.proofs.store import ProofStore
    from paper_distiller.proofgraph.pipeline import build_graph_for_paper
    store = ProofStore(tmp_path / "proofs.db")
    llm = _DispatchLLMWithSelfCheck()
    report = build_graph_for_paper(store, "1234.5678", FAKE_PAPER, paper_slug="fp", llm=llm)

    nodes = store.nodes_by_paper("1234.5678")
    step2 = next((n for n in nodes if n.label == "Step 2"), None)
    assert step2 is not None, f"Step 2 not found; nodes={[n.label for n in nodes]}"
    assert step2.status == "gap", f"Expected gap but got {step2.status}"
    assert report.gaps >= 1


def test_build_graph_coverage_report_segments(tmp_path):
    from paper_distiller.proofs.store import ProofStore
    from paper_distiller.proofgraph.pipeline import build_graph_for_paper
    store = ProofStore(tmp_path / "proofs.db")
    llm = _DispatchLLMWithSelfCheck()
    report = build_graph_for_paper(store, "1234.5678", FAKE_PAPER, paper_slug="fp", llm=llm)
    assert report.segments_processed == report.segments_total
    assert report.segments_total > 0
    assert sum(report.nodes_by_kind.values()) >= 2


def test_build_graph_nodes_by_kind_sums_to_node_count(tmp_path):
    from paper_distiller.proofs.store import ProofStore
    from paper_distiller.proofgraph.pipeline import build_graph_for_paper
    store = ProofStore(tmp_path / "proofs.db")
    llm = _DispatchLLMWithSelfCheck()
    report = build_graph_for_paper(store, "1234.5678", FAKE_PAPER, paper_slug="fp", llm=llm)
    nodes = store.nodes_by_paper("1234.5678")
    assert sum(report.nodes_by_kind.values()) == len(nodes)


def test_build_graph_idempotent_no_duplicates(tmp_path):
    from paper_distiller.proofs.store import ProofStore
    from paper_distiller.proofgraph.pipeline import build_graph_for_paper
    store = ProofStore(tmp_path / "proofs.db")
    llm1 = _DispatchLLMWithSelfCheck()
    llm2 = _DispatchLLMWithSelfCheck()
    build_graph_for_paper(store, "1234.5678", FAKE_PAPER, paper_slug="fp", llm=llm1)
    count_after_first = len(store.nodes_by_paper("1234.5678"))
    build_graph_for_paper(store, "1234.5678", FAKE_PAPER, paper_slug="fp", llm=llm2)
    count_after_second = len(store.nodes_by_paper("1234.5678"))
    assert count_after_first == count_after_second, (
        f"Idempotency broken: {count_after_first} → {count_after_second}"
    )


def test_build_graph_returns_coverage_report_fields(tmp_path):
    from paper_distiller.proofs.store import ProofStore
    from paper_distiller.proofgraph.pipeline import build_graph_for_paper, CoverageReport
    store = ProofStore(tmp_path / "proofs.db")
    llm = _DispatchLLMWithSelfCheck()
    report = build_graph_for_paper(store, "1234.5678", FAKE_PAPER, paper_slug="fp", llm=llm)
    assert isinstance(report.segments_total, int)
    assert isinstance(report.segments_processed, int)
    assert isinstance(report.proof_blocks, int)
    assert isinstance(report.nodes_by_kind, dict)
    assert isinstance(report.rejected_quotes, int)
    assert isinstance(report.gaps, int)
    assert isinstance(report.obligations, list)


class _RejectingLLM:
    """LLM that returns one grounded node + two fabricated nodes on extraction,
    and no-suspicious on self-check.  On retry it returns the same fabricated
    nodes (so they stay rejected)."""

    def __init__(self):
        self.call_count = 0
        # A text fragment that is verbatim in FAKE_PAPER (theorem segment)
        self._grounded = THEOREM_QUOTE
        self._resp = json.dumps({"nodes": [
            {
                "kind": "theorem",
                "label": "Theorem 1",
                "text": "For all x, f(x) <= C.",
                "source_quote": THEOREM_QUOTE,
                "techniques": [],
                "refs": [],
            },
            {
                "kind": "proof_step",
                "text": "Fake step A",
                "source_quote": FABRICATED_QUOTE_A,
                "refs": [],
            },
            {
                "kind": "proof_step",
                "text": "Fake step B",
                "source_quote": FABRICATED_QUOTE_B,
                "refs": [],
            },
        ]})
        self._no_suspicious = json.dumps({"suspicious_labels": []})
        self._empty = json.dumps({"nodes": []})

    def complete(self, messages, temperature=0.2, response_format=None):
        self.call_count += 1
        content = messages[0]["content"] if messages else ""
        if content.startswith("You are reviewing extracted mathematical"):
            return self._no_suspicious
        if THEOREM_QUOTE in content and "Kind hint: theorem" in content:
            return self._resp
        return self._empty


def test_build_graph_rejected_quotes_counts_dropped_nodes(tmp_path):
    """report.rejected_quotes must reflect nodes dropped by the grounding gate."""
    from paper_distiller.proofs.store import ProofStore
    from paper_distiller.proofgraph.pipeline import build_graph_for_paper
    store = ProofStore(tmp_path / "proofs.db")
    llm = _RejectingLLM()
    report = build_graph_for_paper(store, "1234.5678", FAKE_PAPER, paper_slug="fp", llm=llm)
    # The theorem segment produces 1 accepted + 2 rejected
    assert report.rejected_quotes == 2


class _TwoDanglingRefsLLM:
    """LLM that returns one proof_step node with TWO dangling refs and no-suspicious
    on self-check.  The theorem segment returns a named theorem so label_to_id is
    populated for one label; the two refs point at labels that don't exist."""

    # A text verbatim in FAKE_PAPER proof segment
    STEP_QUOTE = "By Bernstein's inequality we bound the tail probability directly."

    def __init__(self):
        self.call_count = 0
        self._theorem_resp = json.dumps({"nodes": [{
            "kind": "theorem",
            "label": "Theorem 1",
            "text": "For all x, f(x) <= C.",
            "source_quote": THEOREM_QUOTE,
            "refs": [],
        }]})
        self._proof_resp = json.dumps({"nodes": [{
            "kind": "proof_step",
            "label": "Step 1",
            "text": "Bernstein tail bound",
            "source_quote": self.STEP_QUOTE,
            "techniques": [],
            "refs": [
                {"rel": "depends_on", "target": "Lemma 99"},
                {"rel": "depends_on", "target": "Lemma 100"},
            ],
        }]})
        self._no_suspicious = json.dumps({"suspicious_labels": []})
        self._empty = json.dumps({"nodes": []})

    def complete(self, messages, temperature=0.2, response_format=None):
        self.call_count += 1
        content = messages[0]["content"] if messages else ""
        if content.startswith("You are reviewing extracted mathematical"):
            return self._no_suspicious
        if self.STEP_QUOTE in content:
            return self._proof_resp
        if THEOREM_QUOTE in content and "Kind hint: theorem" in content:
            return self._theorem_resp
        return self._empty


def test_build_graph_two_dangling_refs_count_one_gap(tmp_path):
    """A node with two dangling refs must increment report.gaps only once,
    and its status must be 'gap'."""
    from paper_distiller.proofs.store import ProofStore
    from paper_distiller.proofgraph.pipeline import build_graph_for_paper
    store = ProofStore(tmp_path / "proofs.db")
    llm = _TwoDanglingRefsLLM()
    report = build_graph_for_paper(store, "1234.5678", FAKE_PAPER, paper_slug="fp", llm=llm)

    nodes = store.nodes_by_paper("1234.5678")
    step = next((n for n in nodes if n.label == "Step 1"), None)
    assert step is not None, f"Step 1 not found; labels={[n.label for n in nodes]}"
    assert step.status == "gap", f"Expected 'gap' but got '{step.status}'"
    assert report.gaps == 1, f"Expected gaps=1 but got {report.gaps}"
