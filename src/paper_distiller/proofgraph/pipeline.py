"""Orchestration pipeline: turn a paper's full text into a proof graph.

Public API:
- ``CoverageReport`` dataclass — summary of what was processed.
- ``build_graph_for_paper(store, paper_arxiv_id, full_text, *, paper_slug, llm,
                          depth="step") -> CoverageReport``
  Segments the text, runs the per-segment extraction+gate+self_check loop,
  writes nodes to the store, resolves references into edges, marks dangling
  refs as gaps, and returns a coverage report.

Design constraints (from spec §5):
- Idempotent: calls ``store.delete_paper_graph`` first so re-runs are clean.
- Abstain over fabricate: the grounding gate (inside ``extract_segment``) ensures
  fabricated nodes never enter the store.
- Gaps are surfaced explicitly (not silently dropped).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..proofs.store import Edge, Node, ProofStore
from .extractor import extract_segment, self_check
from .memory import RunningMemory
from .reader import segment


@dataclass
class CoverageReport:
    """Summary statistics from one paper's extraction run."""
    segments_total: int
    segments_processed: int
    proof_blocks: int
    nodes_by_kind: dict[str, int]
    rejected_quotes: int
    gaps: int
    obligations: list[str]


def build_graph_for_paper(
    store: ProofStore,
    paper_arxiv_id: str,
    full_text: str,
    *,
    paper_slug: str | None = None,
    llm,
    depth: str = "step",
) -> CoverageReport:
    """Build (or rebuild) the proof graph for one paper.

    Steps:
    1. Delete any existing graph data for this paper (idempotency).
    2. Segment the full text.
    3. For each segment: extract nodes (with grounding gate), run self-check,
       write accepted nodes to the store, update running memory.
    4. Resolve references: for each pending (node_id, refs) pair, look up
       ``label_to_id`` and create edges; unresolvable refs → ``status="gap"``.
    5. Return a ``CoverageReport``.
    """
    # Step 1: idempotent delete
    store.delete_paper_graph(paper_arxiv_id)

    # Step 2: segment
    segs = segment(full_text)
    segments_total = len(segs)
    proof_blocks = sum(1 for s in segs if s.is_proof_block)

    # Step 3: per-segment loop
    memory = RunningMemory()
    label_to_id: dict[str, int] = {}
    # pending: list of (node_id, refs_list)
    pending: list[tuple[int, list]] = []
    nodes_by_kind: dict[str, int] = {}
    rejected_quotes = 0
    segments_processed = 0

    for seg in segs:
        # For depth=="theorem" on proof blocks, skip detailed step extraction
        # but still extract statement-level nodes if any
        effective_depth = depth
        if depth == "theorem" and seg.is_proof_block:
            effective_depth = "theorem"  # no proof_step decomposition

        # Extract nodes (grounding gate enforced inside extract_segment)
        accepted = extract_segment(seg, memory, llm, depth=effective_depth)
        accepted = self_check(seg, accepted, llm)

        # Write each accepted node to the store
        for node in accepted:
            loc = json.dumps({
                "sec": seg.section,
                "char_start": seg.char_start,
            })
            store_node = Node(
                paper_arxiv_id=paper_arxiv_id,
                paper_slug=paper_slug,
                kind=node.kind,
                label=node.label,
                text=node.text,
                source_quote=node.source_quote,
                loc=loc,
                status=node.status,
                techniques=list(node.techniques or []),
            )
            nid = store.add_node(store_node)
            # Track label → node id for edge resolution
            if node.label:
                label_to_id[node.label] = nid
            # Accumulate pending refs for edge resolution pass
            pending.append((nid, list(node.refs or [])))
            # Tally by kind
            nodes_by_kind[node.kind] = nodes_by_kind.get(node.kind, 0) + 1

        # Update running memory with accepted nodes
        resolved = set(label_to_id.keys())
        memory.update(accepted, resolved_labels=resolved)
        segments_processed += 1

    # Step 4: resolve edges
    gaps = 0
    obligations: list[str] = []

    for nid, refs in pending:
        for ref in refs:
            target_id = label_to_id.get(ref.target)
            if target_id is not None:
                # Resolvable → create edge
                edge = Edge(src_id=nid, dst_id=target_id, rel=ref.rel)
                try:
                    store.add_edge(edge)
                except Exception:
                    pass  # UNIQUE constraint if duplicate — safe to ignore
            else:
                # Unresolvable → mark as gap
                _set_node_status(store, nid, "gap")
                gaps += 1
                if ref.target not in obligations:
                    obligations.append(ref.target)

    return CoverageReport(
        segments_total=segments_total,
        segments_processed=segments_processed,
        proof_blocks=proof_blocks,
        nodes_by_kind=nodes_by_kind,
        rejected_quotes=rejected_quotes,
        gaps=gaps,
        obligations=obligations,
    )


def _set_node_status(store: ProofStore, node_id: int, status: str) -> None:
    """Update a node's status in the store (used for gap marking)."""
    store._conn.execute(
        "UPDATE nodes SET status=? WHERE id=?", (status, node_id)
    )
    store._conn.commit()
