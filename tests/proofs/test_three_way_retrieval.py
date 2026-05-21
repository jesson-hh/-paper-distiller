"""Tests for v1.9 three-way retrieval (A: store-augmented keyword,
B: FTS5 text match, C: LLM pre-extract)."""

from __future__ import annotations

from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Strategy B: FTS5 text match
# ---------------------------------------------------------------------------

def _seed_store_with_theorems(store):
    """Insert 3 distinct theorems for retrieval tests."""
    from paper_distiller.proofs.store import ProofSidecar

    store.ingest_sidecar(
        ProofSidecar(
            theorems=[{
                "name": "Bernstein bound",
                "statement": "Subexponential concentration via Bernstein method.",
                "proof_sketch": "Moment generating function bound.",
                "techniques_used": ["Bernstein"],
            }],
            key_techniques=["Bernstein"],
        ),
        "paper-A",
    )
    store.ingest_sidecar(
        ProofSidecar(
            theorems=[{
                "name": "Wasserstein duality",
                "statement": "Kantorovich-Rubinstein duality for Wasserstein-1.",
                "proof_sketch": "Convex duality.",
                "techniques_used": ["Wasserstein", "Kantorovich"],
            }],
            key_techniques=["Wasserstein", "Kantorovich"],
        ),
        "paper-B",
    )
    store.ingest_sidecar(
        ProofSidecar(
            theorems=[{
                "name": "Rademacher complexity bound",
                "statement": "Generalization bound via Rademacher complexity.",
                "proof_sketch": "Symmetrization argument.",
                "techniques_used": ["Rademacher", "Symmetrization"],
            }],
            key_techniques=["Rademacher", "Symmetrization"],
        ),
        "paper-C",
    )


def test_retrieve_by_text_match_finds_via_statement(tmp_path):
    from paper_distiller.proofs.store import ProofStore
    store = ProofStore(tmp_path / "proofs.db")
    _seed_store_with_theorems(store)
    # Abstract that doesn't mention "Bernstein" by name but mentions concentration
    results = store.retrieve_by_text_match(
        "We derive subexponential concentration via moment generating functions.",
        limit=5,
    )
    names = {r.name for r in results}
    assert "Bernstein bound" in names
    store.close()


def test_retrieve_by_text_match_handles_empty(tmp_path):
    from paper_distiller.proofs.store import ProofStore
    store = ProofStore(tmp_path / "proofs.db")
    assert store.retrieve_by_text_match("") == []
    assert store.retrieve_by_text_match("   ") == []
    # All stop words — no usable tokens
    assert store.retrieve_by_text_match("the and of to in on") == []
    store.close()


def test_retrieve_by_text_match_respects_limit(tmp_path):
    from paper_distiller.proofs.store import ProofStore
    store = ProofStore(tmp_path / "proofs.db")
    _seed_store_with_theorems(store)
    # Query that should match all 3
    results = store.retrieve_by_text_match(
        "concentration Bernstein Wasserstein Kantorovich Rademacher complexity",
        limit=2,
    )
    assert len(results) <= 2
    store.close()


def test_list_canonical_technique_names(tmp_path):
    from paper_distiller.proofs.store import ProofStore
    store = ProofStore(tmp_path / "proofs.db")
    _seed_store_with_theorems(store)
    names = store.list_canonical_technique_names()
    assert "Bernstein" in names
    assert "Wasserstein" in names
    assert "Rademacher" in names
    store.close()


# ---------------------------------------------------------------------------
# Strategy A: store-augmented keyword scan
# ---------------------------------------------------------------------------

def test_gather_candidate_techniques_uses_store(tmp_path):
    """When abstract mentions a vault-learned technique name, candidate
    list should include it even if it's NOT in the hardcoded list."""
    from paper_distiller.agents.processor import _gather_candidate_techniques
    from paper_distiller.proofs.store import ProofStore, ProofSidecar
    from paper_distiller.sources.arxiv import Paper

    store = ProofStore(tmp_path / "proofs.db")
    # Register a technique that's NOT in the hardcoded list
    store.ingest_sidecar(
        ProofSidecar(key_techniques=["Csiszar f-divergence"]),
        "paper-X",
    )
    paper = Paper(
        source="arxiv", paper_id="P1",
        title="Test", authors=[],
        abstract="We use Csiszar f-divergence for our analysis.",
        published="", pdf_url="", arxiv_id="P1",
    )
    result = _gather_candidate_techniques(paper, store, llm=None)
    assert "Csiszar f-divergence" in result
    store.close()


# ---------------------------------------------------------------------------
# Strategy C: LLM pre-extract
# ---------------------------------------------------------------------------

def test_llm_extract_techniques_parses_response():
    from paper_distiller.agents.processor import _llm_extract_techniques
    from paper_distiller.sources.arxiv import Paper

    llm = MagicMock()
    llm.complete.return_value = (
        "Hölder inequality\n"
        "Bernstein concentration\n"
        "- Rademacher complexity\n"
        "1. Wasserstein distance\n"
        "Fenchel duality (Sec 3)\n"
    )
    paper = Paper(
        source="arxiv", paper_id="P1", title="t", authors=[],
        abstract="abstract", published="", pdf_url="", arxiv_id="P1",
    )
    techs = _llm_extract_techniques(paper, llm)
    assert "Hölder inequality" in techs
    assert "Bernstein concentration" in techs
    assert "Rademacher complexity" in techs
    assert "Wasserstein distance" in techs
    # Trailing paren should be stripped
    assert "Fenchel duality" in techs


def test_llm_extract_techniques_disabled_by_env(monkeypatch):
    from paper_distiller.agents.processor import _llm_extract_techniques
    from paper_distiller.sources.arxiv import Paper

    monkeypatch.setenv("PD_LLM_TECH_EXTRACT", "0")
    llm = MagicMock()
    llm.complete.return_value = "Hölder"
    paper = Paper(
        source="arxiv", paper_id="P1", title="t", authors=[],
        abstract="x", published="", pdf_url="", arxiv_id="P1",
    )
    assert _llm_extract_techniques(paper, llm) == []
    llm.complete.assert_not_called()


def test_llm_extract_techniques_handles_llm_failure():
    from paper_distiller.agents.processor import _llm_extract_techniques
    from paper_distiller.sources.arxiv import Paper

    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("network down")
    paper = Paper(
        source="arxiv", paper_id="P1", title="t", authors=[],
        abstract="x", published="", pdf_url="", arxiv_id="P1",
    )
    # Should swallow exception and return []
    assert _llm_extract_techniques(paper, llm) == []


# ---------------------------------------------------------------------------
# A + B + C combined via _gather_candidate_techniques
# ---------------------------------------------------------------------------

def test_gather_candidate_techniques_dedups_across_sources(tmp_path):
    """Same name from hardcoded + store + LLM should appear once."""
    from paper_distiller.agents.processor import _gather_candidate_techniques
    from paper_distiller.proofs.store import ProofStore, ProofSidecar
    from paper_distiller.sources.arxiv import Paper

    store = ProofStore(tmp_path / "proofs.db")
    store.ingest_sidecar(
        ProofSidecar(key_techniques=["Hölder"]),
        "paper-X",
    )
    llm = MagicMock()
    llm.complete.return_value = "Hölder\nBernstein concentration\n"
    paper = Paper(
        source="arxiv", paper_id="P1", title="t", authors=[],
        abstract="We use Hölder's inequality.",
        published="", pdf_url="", arxiv_id="P1",
    )
    result = _gather_candidate_techniques(paper, store, llm=llm)
    # Hölder appears only once (case-insensitive dedup)
    holder_hits = sum(1 for t in result if "hölder" in t.lower() or "holder" in t.lower())
    assert holder_hits == 1
    # Bernstein should be present from LLM
    assert any("Bernstein" in t for t in result)
    store.close()
