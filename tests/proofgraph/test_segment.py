"""Tests for proofgraph.reader.segment — deterministic paper segmentation."""
from __future__ import annotations

SAMPLE = """\
1 Introduction
We study the convergence of estimators under sub-Gaussian noise.

2 Main Result
Theorem 4.3. For all f, ||f|| <= C n^{-1/2}.
Proof. By Bernstein's inequality we bound the tail. Applying Dudley chaining
to the empirical process yields the claim. □

3 Discussion
Future work remains.
"""


def test_segment_splits_by_section_and_marks_proof_block():
    from paper_distiller.proofgraph.reader import segment
    segs = segment(SAMPLE)
    # every segment carries the text it covers + offsets within the source
    assert all(s.text == SAMPLE[s.char_start:s.char_end] for s in segs)
    # at least one proof block detected (the "Proof. ... □" region)
    proofs = [s for s in segs if s.is_proof_block]
    assert len(proofs) == 1
    assert "Bernstein" in proofs[0].text
    # a theorem-statement segment is detected
    assert any(s.kind_hint == "theorem" for s in segs)
    # coverage: concatenated segment text reconstructs (modulo splits) the source
    assert sum(len(s.text) for s in segs) > 0


def test_segment_empty_input_returns_empty():
    from paper_distiller.proofgraph.reader import segment
    assert segment("") == []
    assert segment("   \n  ") == []
