"""Diagnostic: distill TWO related papers and verify:
1. First paper's proof_sidecar gets extracted + stored
2. Second paper's distillation receives prior theorems as context
3. ProofStore stats reflect both papers
"""

import sys
import tempfile
import time
from pathlib import Path

# UTF-8 stdout for Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from paper_distiller.arxiv_local.store import Store, _default_dir
from paper_distiller.arxiv_local.search import search as local_search_fn
from paper_distiller.distill.article import distill
from paper_distiller.llm.openai_compatible import LLMClient
from paper_distiller.pipeline import fetch_with_fallback
from paper_distiller.config import load_config
from paper_distiller.proofs.store import ProofStore
from paper_distiller.vault.crosslink import WikiIndex


def _distill_one(paper, cfg, llm, proof_store, prior_theorems=None):
    print(f"\n=== distilling {paper.arxiv_id} ===")
    print(f"  title: {paper.title[:60]}")
    print(f"  authors: {', '.join(paper.authors[:3])}")
    if prior_theorems:
        print(f"  prior theorems injected: {len(prior_theorems)}")
        for t in prior_theorems[:3]:
            print(f"    - {t.name} (techniques: {t.techniques_used[:3]})")
    else:
        print(f"  prior theorems: 0 (cold start)")

    tmpdir = Path(tempfile.mkdtemp(prefix="diag-proof-"))
    print(f"  fetching PDF...")
    t0 = time.time()
    full_text = fetch_with_fallback(paper, cfg, tmpdir)
    print(f"  PDF: {time.time() - t0:.1f}s · {len(full_text):,} chars")

    print(f"  calling LLM (deep distill + proof_sidecar)...")
    t0 = time.time()
    article = distill(
        paper, full_text, WikiIndex(entries=[]), llm,
        prior_theorems=prior_theorems,
    )
    print(f"  LLM: {time.time() - t0:.1f}s · {llm.total_tokens_in:,} in / {llm.total_tokens_out:,} out cumulative")

    # Inspect proof_sidecar
    sc = article.proof_sidecar
    print(f"  → article body: {len(article.body):,} chars, {article.body.count('## ')} sections")
    print(f"  → proof_sidecar:")
    print(f"      theorems: {len(sc.theorems)}")
    for t in sc.theorems[:3]:
        techs = t.get("techniques_used", [])
        print(f"        - {t.get('name')}: techniques={techs[:3]}")
    print(f"      definitions: {len(sc.key_definitions)}")
    print(f"      key_techniques: {len(sc.key_techniques)}")
    print(f"        {sc.key_techniques[:8]}")

    # Persist to store
    result = proof_store.ingest_sidecar(sc, paper.arxiv_id, article.slug)
    print(f"  → ingested: {result}")
    return article


def main():
    # Setup
    cfg = load_config(
        vault_path=r"G:\Math research Agent\wiki",
        topic="diffusion", source="arxiv",
    )
    llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model)
    proof_store = ProofStore(Path(__file__).parent / "diag_proofs.db")
    # Reset between runs so we see a clean cold start
    proof_store._conn.execute("DELETE FROM theorems")
    proof_store._conn.execute("DELETE FROM techniques")
    proof_store._conn.commit()

    print(f"=== v1.8 end-to-end diag ===")
    print(f"model: {cfg.model}")
    print(f"proof store: {proof_store.path}  (cleared)")
    print(f"initial: {proof_store.theorem_count()} theorems")

    # Pull two papers
    arxiv_store = Store(_default_dir() / "arxiv.db")
    candidates = local_search_fn(arxiv_store, "Yuling Jiao", n=2)
    arxiv_store.close()

    if len(candidates) < 2:
        print("need ≥2 papers; abort")
        return
    paper_a, paper_b = candidates[0], candidates[1]

    # First paper — cold start
    art_a = _distill_one(paper_a, cfg, llm, proof_store, prior_theorems=None)

    print(f"\n--- ProofStore state after paper 1 ---")
    print(f"  theorems: {proof_store.theorem_count()}")
    print(f"  techniques: {proof_store.technique_count()}")
    print(f"  techniques sample: {[t.name for t in proof_store.list_techniques(8)]}")

    # Second paper — should get RAG injection
    # Use technique candidates from paper_b's abstract
    from paper_distiller.agents.processor import _extract_candidate_techniques
    candidates_b = _extract_candidate_techniques(paper_b)
    print(f"\n--- candidate techniques for paper B ---")
    print(f"  abstract keyword hits: {candidates_b}")
    prior = proof_store.retrieve_relevant(candidates_b)
    print(f"  retrieved prior theorems: {len(prior)}")

    art_b = _distill_one(paper_b, cfg, llm, proof_store, prior_theorems=prior)

    print(f"\n--- ProofStore state after paper 2 ---")
    print(f"  total theorems: {proof_store.theorem_count()}")
    print(f"  total techniques: {proof_store.technique_count()}")
    print(f"  papers covered: {proof_store.paper_count()}")

    # Cross-paper retrieval test
    print(f"\n--- cross-paper retrieval test ---")
    if proof_store.technique_count() > 0:
        # Pick a technique used by paper A and look for theorems
        first_tech = proof_store.list_techniques(1)
        if first_tech:
            tech_name = first_tech[0].name
            results = proof_store.theorems_using_technique(tech_name)
            print(f"  theorems_using_technique({tech_name!r}): {len(results)} hits")
            for t in results[:3]:
                print(f"    - {t.name} from {t.paper_arxiv_id}")

    print(f"\n=== summary ===")
    print(f"  total LLM cost: CNY {llm.estimated_cost_cny:.4f}")
    print(f"  total tokens: {llm.total_tokens_in:,} in / {llm.total_tokens_out:,} out")
    print(f"  vault articles: 2")
    print(f"  proof store: {proof_store.theorem_count()} theorems · "
          f"{proof_store.technique_count()} techniques · "
          f"{proof_store.paper_count()} papers")

    proof_store.close()


if __name__ == "__main__":
    main()
