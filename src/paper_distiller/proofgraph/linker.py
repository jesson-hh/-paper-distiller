"""Cross-paper linker: find candidate node pairs and classify their relation.

Phase 5 of the proof-graph build pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from paper_distiller.proofs.store import Node, ProofStore


# ---------------------------------------------------------------------------
# Valid cross-paper relation types (from the data model / spec §6).
# ---------------------------------------------------------------------------

VALID_RELS: frozenset[str] = frozenset(
    {"same_as", "specializes", "generalizes", "uses_lemma", "contradicts"}
)


# ---------------------------------------------------------------------------
# Task 5.1 — find_candidates (deterministic, cross-paper only)
# ---------------------------------------------------------------------------


def find_candidates(store: ProofStore, node: Node, k: int = 6) -> list[Node]:
    """Return up to *k* cross-paper candidate nodes for *node*.

    Strategy (deterministic, no LLM):
      1. Technique overlap — for each technique on *node*, query
         ``store.nodes_using_technique(t, limit=k)``.
      2. FTS5 text match — ``store.search_nodes(node.text, limit=k)``.

    Guarantees:
      - Excludes any node whose ``paper_arxiv_id == node.paper_arxiv_id``.
      - Excludes ``node`` itself (same ``id``).
      - Deduplicates by ``id`` (first-seen order: technique matches first).
      - Returns first ``k`` after dedup.
    """
    seen: dict[int, Node] = {}

    def _add(candidate: Node) -> None:
        if candidate.id is None:
            return
        if candidate.paper_arxiv_id == node.paper_arxiv_id:
            return
        if candidate.id == node.id:
            return
        if candidate.id not in seen:
            seen[candidate.id] = candidate

    # Technique overlap (strategy A)
    for technique in node.techniques or []:
        for cand in store.nodes_using_technique(technique, limit=k):
            _add(cand)

    # FTS text match (strategy B)
    for cand in store.search_nodes(node.text, limit=k):
        _add(cand)

    return list(seen.values())[:k]


# ---------------------------------------------------------------------------
# Task 5.2 — classify_pair (LLM, small context)
# ---------------------------------------------------------------------------


def _load_prompt_template() -> str:
    """Load the link_classify prompt template from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / "link_classify.md"
    return prompt_path.read_text(encoding="utf-8")


def classify_pair(
    node_a: Node,
    node_b: Node,
    llm,
) -> tuple[str | None, str]:
    """Classify the relation between two cross-paper nodes using the LLM.

    Returns ``(rel, justification)`` where ``rel`` is one of ``VALID_RELS``
    or ``None`` (abstain) when the LLM returns ``"none"``, an invalid relation,
    or produces unparseable output.  Never invents an edge.
    """
    try:
        template = _load_prompt_template()
        prompt = template.format(
            text_a=node_a.text or "",
            source_quote_a=node_a.source_quote or "",
            text_b=node_b.text or "",
            source_quote_b=node_b.source_quote or "",
        )
    except Exception:
        return (None, "")

    messages = [
        {"role": "system", "content": (
            "You are a mathematical proof-graph linker. "
            "Output only valid JSON."
        )},
        {"role": "user", "content": prompt},
    ]

    try:
        raw = llm.complete(messages, temperature=0.0, response_format="json")
        data = json.loads(raw)
        rel = data.get("rel", "none")
        justification = str(data.get("justification") or "")
    except Exception:
        return (None, "")

    if rel not in VALID_RELS:
        return (None, justification)

    return (rel, justification)
