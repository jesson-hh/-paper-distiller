"""PaperProcessor — fanout agent: one sub-agent per paper. Each does
fetch + extract + distill independently, in parallel.

v1.8: each distillation also extracts a `proof_sidecar` (theorems +
techniques) and persists it into the per-vault ProofStore for future
cross-paper retrieval. Before distilling a paper, we query the store for
likely-relevant prior theorems (based on the paper abstract's mentioned
techniques) and inject them as context.

Per-paper LLM failures are logged + dropped — they don't abort the run.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path

from ..distill.article import distill as distill_article
from ..llm.openai_compatible import LLMError
from ..pipeline import fetch_with_fallback
from ..proofs.store import open_for_vault
from ..vault.crosslink import load_index
from .base import Agent, Context

# Module-level lock — fine because each run() invocation creates a fresh DAG.
# Multiple _DistillOne instances inside one fanout invocation share this lock
# to serialize their ctx.shared["articles"].append() calls.
_articles_lock = asyncio.Lock()
_proof_store_lock = asyncio.Lock()


# Hand-curated list of common math/CS technique names to scan for in the
# paper abstract. Cheap pre-filter that picks "candidate techniques" for
# proof store retrieval — much cheaper than an LLM extraction pass.
_TECHNIQUE_HINTS = [
    "Hölder", "Holder", "Cauchy-Schwarz", "Bernstein", "Hoeffding",
    "Markov inequality", "Chebyshev", "Jensen", "Pinsker",
    "Dudley", "chaining", "Rademacher", "VC dimension",
    "Lipschitz", "Sobolev", "Lipschitz extension",
    "Wasserstein", "optimal transport", "Kantorovich",
    "Talagrand", "concentration", "sub-Gaussian", "sub-Exponential",
    "martingale", "Doob",
    "ReLU", "neural network approximation", "Barron",
    "Bayesian", "posterior", "variational",
    "diffusion", "score-matching", "Langevin", "Stein", "Fisher",
    "gradient descent", "SGD", "Adam", "momentum",
    "convex", "non-convex", "saddle point",
    "RKHS", "kernel", "GP",
    "PAC-Bayes", "minimax", "Le Cam",
    "TV distance", "KL divergence", "f-divergence", "IPM",
]


def _extract_candidate_techniques(paper) -> list[str]:
    """Cheap keyword scan of title + abstract for known technique names.

    Used to pre-fetch potentially relevant theorems from the ProofStore
    before LLM distillation. Best-effort — false negatives are OK (we
    just won't inject as much context); false positives also OK (the LLM
    can ignore irrelevant references).
    """
    haystack = " ".join([
        getattr(paper, "title", "") or "",
        getattr(paper, "abstract", "") or "",
    ]).lower()
    found = []
    for tech in _TECHNIQUE_HINTS:
        if tech.lower() in haystack:
            found.append(tech)
    return found


# Strategy C: cheap dedicated LLM call to extract candidate techniques
# from title+abstract. ~150 tokens out per paper, ~¥0.005, ~5-10s.
# Disable via PD_LLM_TECH_EXTRACT=0 if cost matters.
_TECH_EXTRACT_PROMPT = """\
List 5-10 specific mathematical techniques, inequalities, or theoretical \
frameworks this paper LIKELY uses. Use canonical short English names \
(e.g. 'Hölder inequality', 'Bernstein concentration', 'Rademacher complexity', \
'Wasserstein distance', 'Fenchel duality', 'Lipschitz extension'). One per \
line, no numbering, no explanations.

Title: {title}
Abstract: {abstract}
"""


def _llm_extract_techniques(paper, llm) -> list[str]:
    """Strategy C: LLM extracts candidate techniques from abstract alone.

    Catches papers whose abstract doesn't mention the specific technique
    by name but where an LLM can infer (e.g. "we prove convergence rate
    for the GAN estimator" → Bernstein concentration, Dudley chaining).

    Returns up to 15 technique names. Returns [] on any error.
    """
    import os
    if os.getenv("PD_LLM_TECH_EXTRACT", "1").lower() in ("0", "false", ""):
        return []
    title = getattr(paper, "title", "") or ""
    abstract = getattr(paper, "abstract", "") or ""
    if not (title or abstract):
        return []
    prompt = _TECH_EXTRACT_PROMPT.format(title=title, abstract=abstract[:2000])
    try:
        raw = llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        )
    except Exception:
        return []
    techs: list[str] = []
    for line in (raw or "").strip().split("\n"):
        line = line.strip().lstrip("-*•").lstrip("0123456789.) ").strip()
        # Drop trailing parens with explanation, e.g. "Hölder (Sec 3)"
        if "(" in line:
            line = line.split("(")[0].strip()
        if line and 3 <= len(line) <= 60:
            techs.append(line)
    return techs[:15]


def _gather_candidate_techniques(paper, proof_store, llm=None) -> list[str]:
    """Combine 3 strategies for finding candidate techniques.

    A: hardcoded keyword scan + augment with vault-learned canonical names
    B: NOT here (B is FTS5 text match, done separately at retrieval time)
    C: optional LLM pre-extract from abstract

    Returns deduplicated list. Order: hardcoded → vault-learned → LLM
    (so retrieve_relevant tries cheapest/most-reliable first).
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        key = name.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(name)

    # A.1 — hardcoded
    for t in _extract_candidate_techniques(paper):
        _add(t)

    # A.2 — augment with store-known canonical names that appear in abstract
    if proof_store is not None:
        try:
            haystack = (
                (getattr(paper, "title", "") or "") + " "
                + (getattr(paper, "abstract", "") or "")
            ).lower()
            for name in proof_store.list_canonical_technique_names(limit=500):
                if not name:
                    continue
                if name.lower() in haystack:
                    _add(name)
        except Exception:
            pass

    # C — LLM pre-extract (skip if disabled by env or no llm)
    if llm is not None:
        for t in _llm_extract_techniques(paper, llm):
            _add(t)

    return out


class _DistillOne:
    def __init__(self, paper, idx, total, tmpdir, wiki_index, proof_store):
        self.name = f"paper-processor[{idx + 1}/{total}]"
        self.deps: list[str] = []
        self._paper = paper
        self._tmpdir = tmpdir
        self._wiki_index = wiki_index
        self._proof_store = proof_store

    async def run(self, ctx: Context) -> dict:
        title_preview = (getattr(self._paper, "title", "") or "")[:50]
        try:
            try:
                ctx.on_status(
                    self.name,
                    activity=f"PDF fetch: {self._paper.arxiv_id}",
                )
            except Exception:
                pass
            full_text = await asyncio.to_thread(
                fetch_with_fallback, self._paper, ctx.cfg, self._tmpdir,
            )

            # Pre-fetch relevant prior theorems from the ProofStore using
            # 3 complementary strategies (v1.9):
            #   A. hardcoded keyword scan + vault-learned canonical names
            #   B. FTS5 BM25 match of title+abstract against theorem corpus
            #   C. optional LLM-extracted candidate techniques
            # Merge + dedupe, cap at max_total.
            prior_theorems = []
            if self._proof_store is not None:
                try:
                    ctx.on_status(self.name, activity="proof RAG: gathering candidates")
                except Exception:
                    pass

                candidates = await asyncio.to_thread(
                    _gather_candidate_techniques,
                    self._paper, self._proof_store, ctx.llm,
                )
                by_technique = []
                if candidates:
                    try:
                        by_technique = await asyncio.to_thread(
                            self._proof_store.retrieve_relevant,
                            candidates,
                        )
                    except Exception:
                        by_technique = []

                # Strategy B — FTS5 match over title + abstract text
                by_text = []
                try:
                    haystack = " ".join([
                        getattr(self._paper, "title", "") or "",
                        getattr(self._paper, "abstract", "") or "",
                    ])
                    by_text = await asyncio.to_thread(
                        self._proof_store.retrieve_by_text_match,
                        haystack, 6,
                    )
                except Exception:
                    by_text = []

                # Merge + dedupe (preserve order: technique-matches first)
                seen_ids: set = set()
                for thm in by_technique + by_text:
                    if thm.id is None or thm.id in seen_ids:
                        continue
                    seen_ids.add(thm.id)
                    prior_theorems.append(thm)
                    if len(prior_theorems) >= 12:
                        break

                try:
                    ctx.on_status(
                        self.name,
                        activity=(
                            f"proof RAG: {len(candidates)} candidates · "
                            f"{len(prior_theorems)} prior theorems "
                            f"(technique:{len(by_technique)} + text:{len(by_text)})"
                        ),
                    )
                except Exception:
                    pass

            try:
                ctx.on_status(
                    self.name,
                    activity=(
                        f"LLM distill: {title_preview}"
                        + (f"  (+{len(prior_theorems)} prior theorems)"
                           if prior_theorems else "")
                    ),
                )
            except Exception:
                pass
            article = await asyncio.to_thread(
                distill_article, self._paper, full_text,
                self._wiki_index, ctx.llm, prior_theorems,
            )
        except LLMError:
            if ctx.cfg.verbose:
                print(f"  distill failed for {self._paper.arxiv_id}")
            return {}
        async with _articles_lock:
            current = ctx.shared.get("articles", [])
            current.append(article)
            ctx.shared["articles"] = current

        # Persist sidecar into proof store (serialized via lock)
        if self._proof_store is not None and article.proof_sidecar.theorems:
            try:
                async with _proof_store_lock:
                    await asyncio.to_thread(
                        self._proof_store.ingest_sidecar,
                        article.proof_sidecar,
                        self._paper.arxiv_id or "",
                        article.slug,
                    )
            except Exception:
                pass  # never let proof-store errors abort distillation
        return {}


class PaperProcessor:
    """Fanout agent — produces N _DistillOne sub-agents at runtime.

    Sub-agents MUST have deps=[] — they run as a synthetic single-level
    fanout, not via topological sort.
    """
    name = "paper-processor"
    deps = ["candidate-ranker"]

    def expand(self, ctx: Context) -> list[Agent]:
        # Always setdefault — never clobber. QA-mode accumulates articles
        # across rounds; a round with zero ranked papers must not wipe them.
        ctx.shared.setdefault("articles", [])
        papers = ctx.shared.get("ranked", [])
        if not papers:
            return []
        tmpdir = Path(tempfile.mkdtemp(prefix="paper-distiller-"))
        wiki_index = load_index(ctx.vault)
        # Open the per-vault ProofStore once and share it across sub-agents.
        # _proof_store_lock serializes writes; concurrent reads are safe.
        # Guard against MagicMock'd vaults in tests where ctx.vault.root
        # would coerce to a string path under MagicMock/ rather than a real
        # directory — leaking artifacts into the repo.
        try:
            from pathlib import Path as _Path
            root = ctx.vault.root
            if not isinstance(root, (str, _Path)):
                proof_store = None
            else:
                proof_store = open_for_vault(root)
        except Exception:
            proof_store = None
        return [
            _DistillOne(p, i, len(papers), tmpdir, wiki_index, proof_store)
            for i, p in enumerate(papers)
        ]
