"""Tests for proofgraph.linker — find_candidates, classify_pair, link_paper."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_distiller.proofs.store import Edge, Node, ProofStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_store(tmp_path: Path) -> tuple[ProofStore, Node, Node, Node]:
    """Seed a ProofStore with 3-paper fixture: A (Bernstein), B (Bernstein), C (SGD)."""
    store = ProofStore(tmp_path / "proofs.db")

    id_a = store.add_node(Node(
        paper_arxiv_id="A",
        kind="theorem",
        text="Bound via Bernstein concentration",
        techniques=["Bernstein"],
    ))
    id_b = store.add_node(Node(
        paper_arxiv_id="B",
        kind="theorem",
        text="We use Bernstein concentration to bound the tail",
        techniques=["Bernstein"],
    ))
    id_c = store.add_node(Node(
        paper_arxiv_id="C",
        kind="theorem",
        text="convex optimization via gradient descent",
        techniques=["SGD"],
    ))

    node_a = store.get_node(id_a)
    node_b = store.get_node(id_b)
    node_c = store.get_node(id_c)
    return store, node_a, node_b, node_c


# ---------------------------------------------------------------------------
# Task 5.1 — find_candidates
# ---------------------------------------------------------------------------

class TestFindCandidates:
    def test_finds_cross_paper_technique_match(self, tmp_path):
        from paper_distiller.proofgraph.linker import find_candidates

        store, node_a, node_b, node_c = _seed_store(tmp_path)
        candidates = find_candidates(store, node_a, k=6)

        ids = [n.id for n in candidates]
        assert node_b.id in ids, "B (same technique 'Bernstein') should be a candidate"

    def test_excludes_same_paper(self, tmp_path):
        from paper_distiller.proofgraph.linker import find_candidates

        store, node_a, node_b, node_c = _seed_store(tmp_path)
        # Add a second node from paper A
        id_a2 = store.add_node(Node(
            paper_arxiv_id="A",
            kind="lemma",
            text="Another Bernstein bound for paper A",
            techniques=["Bernstein"],
        ))
        node_a2 = store.get_node(id_a2)
        candidates = find_candidates(store, node_a, k=10)
        ids = [n.id for n in candidates]
        assert node_a2.id not in ids, "Same-paper nodes must be excluded"

    def test_excludes_self(self, tmp_path):
        from paper_distiller.proofgraph.linker import find_candidates

        store, node_a, node_b, node_c = _seed_store(tmp_path)
        candidates = find_candidates(store, node_a, k=6)
        ids = [n.id for n in candidates]
        assert node_a.id not in ids, "Node itself must be excluded"

    def test_excludes_unrelated_paper(self, tmp_path):
        from paper_distiller.proofgraph.linker import find_candidates

        store, node_a, node_b, node_c = _seed_store(tmp_path)
        candidates = find_candidates(store, node_a, k=6)
        ids = [n.id for n in candidates]
        assert node_c.id not in ids, "C (SGD, unrelated) should not appear"

    def test_dedup_by_id(self, tmp_path):
        from paper_distiller.proofgraph.linker import find_candidates

        store, node_a, node_b, node_c = _seed_store(tmp_path)
        candidates = find_candidates(store, node_a, k=10)
        ids = [n.id for n in candidates]
        assert len(ids) == len(set(ids)), "No duplicate node ids"

    def test_respects_k_limit(self, tmp_path):
        from paper_distiller.proofgraph.linker import find_candidates

        store, node_a, node_b, node_c = _seed_store(tmp_path)
        # Add many extra Bernstein nodes from different papers
        for i in range(20):
            store.add_node(Node(
                paper_arxiv_id=f"X{i}",
                kind="theorem",
                text="Bernstein concentration tail bound",
                techniques=["Bernstein"],
            ))
        candidates = find_candidates(store, node_a, k=5)
        assert len(candidates) <= 5


# ---------------------------------------------------------------------------
# Task 5.2 — classify_pair
# ---------------------------------------------------------------------------

class _StubLLM:
    """Minimal stub: records calls, returns preset responses."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list] = []

    def complete(self, messages, temperature=0.0, response_format=None) -> str:
        self.calls.append(messages)
        return self._responses.pop(0)


class TestClassifyPair:
    def test_returns_valid_rel(self, tmp_path):
        from paper_distiller.proofgraph.linker import classify_pair

        store, node_a, node_b, _ = _seed_store(tmp_path)
        llm = _StubLLM(['{"rel":"same_as","justification":"both state the same Bernstein bound"}'])
        rel, just = classify_pair(node_a, node_b, llm)
        assert rel == "same_as"
        assert "Bernstein" in just

    def test_none_rel_returns_none(self, tmp_path):
        from paper_distiller.proofgraph.linker import classify_pair

        store, node_a, node_b, _ = _seed_store(tmp_path)
        llm = _StubLLM(['{"rel":"none","justification":"unrelated"}'])
        rel, just = classify_pair(node_a, node_b, llm)
        assert rel is None
        assert just == "unrelated"

    def test_invalid_rel_abstains(self, tmp_path):
        from paper_distiller.proofgraph.linker import classify_pair

        store, node_a, node_b, _ = _seed_store(tmp_path)
        llm = _StubLLM(['{"rel":"invented_relation","justification":"foo"}'])
        rel, just = classify_pair(node_a, node_b, llm)
        assert rel is None, "Invalid rel must abstain"

    def test_garbage_json_abstains(self, tmp_path):
        from paper_distiller.proofgraph.linker import classify_pair

        store, node_a, node_b, _ = _seed_store(tmp_path)
        llm = _StubLLM(["this is not json at all!!!"])
        rel, just = classify_pair(node_a, node_b, llm)
        assert rel is None, "Garbage JSON must abstain"

    def test_all_valid_rels_accepted(self, tmp_path):
        from paper_distiller.proofgraph.linker import classify_pair

        store, node_a, node_b, _ = _seed_store(tmp_path)
        valid = ["same_as", "specializes", "generalizes", "uses_lemma", "contradicts"]
        for rel_name in valid:
            llm = _StubLLM([json.dumps({"rel": rel_name, "justification": "ok"})])
            rel, _ = classify_pair(node_a, node_b, llm)
            assert rel == rel_name, f"Expected {rel_name} to be accepted"

    def test_calls_llm_with_messages(self, tmp_path):
        from paper_distiller.proofgraph.linker import classify_pair

        store, node_a, node_b, _ = _seed_store(tmp_path)
        llm = _StubLLM(['{"rel":"uses_lemma","justification":"B uses A result"}'])
        classify_pair(node_a, node_b, llm)
        assert len(llm.calls) == 1, "LLM must be called exactly once per pair"
        # The messages must contain both nodes' text
        all_text = " ".join(
            m.get("content", "") for m in llm.calls[0] if isinstance(m, dict)
        )
        assert node_a.text in all_text or node_b.text in all_text
