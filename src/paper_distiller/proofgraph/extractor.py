"""Per-segment LLM extraction with the grounding gate and self-check pass.

Public API:
- ``extract_segment(segment, memory, llm, depth="step") -> list[ExtractedNode]``
  Calls the LLM, runs every node through the grounding gate, retries once on
  failure, drops/marks unsupported nodes that still fail.  Returns only
  accepted (grounded) nodes.

- ``self_check(segment, nodes, llm) -> list[ExtractedNode]``
  Cheap follow-up pass: asks the LLM which nodes over-claim beyond the segment
  text; marks those ``status="suspicious"``.  Tolerates all LLM failures.
"""
from __future__ import annotations

import json
from pathlib import Path

from .extraction_schema import ExtractedNode, parse_extraction
from .reader import Segment, verify_quote

_PROMPT_PATH = Path(__file__).parent / "prompts" / "extract.md"

# Grounding gate threshold: same as verify_quote default
_GATE_THRESHOLD = 0.85


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _build_extract_prompt(segment: Segment, memory, depth: str) -> str:
    template = _load_prompt()
    return template.format(
        memory=memory.render() or "(none yet)",
        kind_hint=segment.kind_hint,
        section=segment.section or "(unknown)",
        segment_text=segment.text,
        depth=depth,
    )


def _run_gate(nodes: list[ExtractedNode], segment_text: str) -> tuple[list[ExtractedNode], list[ExtractedNode]]:
    """Apply grounding gate to each node.

    Returns (accepted, failed) where ``accepted`` nodes pass the gate and
    ``failed`` nodes did not.
    """
    accepted: list[ExtractedNode] = []
    failed: list[ExtractedNode] = []
    for node in nodes:
        result = verify_quote(node.source_quote, segment_text, threshold=_GATE_THRESHOLD)
        if result.ok:
            accepted.append(node)
        else:
            failed.append(node)
    return accepted, failed


def _retry_prompt(segment: Segment, failed_quotes: list[str], depth: str, memory) -> str:
    """Build a retry prompt asking the LLM to fix the failed quotes."""
    failed_list = "\n".join(f"- {q!r}" for q in failed_quotes)
    original_prompt = _build_extract_prompt(segment, memory, depth)
    return (
        original_prompt
        + f"\n\n## RETRY — fix these source_quotes\n"
        f"The following source_quotes were NOT found verbatim in the segment text above.\n"
        f"Re-extract these nodes, copying the source_quote character-for-character from the segment:\n"
        f"{failed_list}\n"
        f"Return ONLY the corrected nodes in the same JSON format."
    )


def extract_segment(
    segment: Segment,
    memory,  # RunningMemory
    llm,     # LLMClient or compatible stub with .complete()
    depth: str = "step",
) -> tuple[list[ExtractedNode], int]:
    """Extract nodes from *segment* using the LLM, enforcing the grounding gate.

    Returns a tuple ``(accepted, n_rejected)`` where:
    - ``accepted`` is the list of grounded nodes.
    - ``n_rejected`` is the count of nodes dropped because their
      ``source_quote`` failed ``verify_quote`` after the one retry.

    Algorithm:
    1. Build prompt from ``prompts/extract.md`` + memory + segment.
    2. Call ``llm.complete(...)``.
    3. Parse extraction JSON.
    4. Run grounding gate on every node (``verify_quote``).
    5. For nodes that fail the gate: retry the LLM **once** with a correction
       prompt for those specific failed quotes.
    6. Re-run the gate on the retry results; still-failing nodes are dropped
       and counted in ``n_rejected``.
    """
    # Step 1-2: initial extraction call
    prompt = _build_extract_prompt(segment, memory, depth)
    messages = [{"role": "user", "content": prompt}]
    try:
        raw = llm.complete(messages, temperature=0.2, response_format="json")
    except Exception:
        return [], 0

    # Step 3: parse
    nodes = parse_extraction(raw)
    if not nodes:
        return [], 0

    # Step 4: gate
    accepted, failed = _run_gate(nodes, segment.text)

    if not failed:
        return accepted, 0

    # Step 5: retry once for failed nodes
    failed_quotes = [n.source_quote for n in failed]
    retry_prompt = _retry_prompt(segment, failed_quotes, depth, memory)
    retry_messages = [{"role": "user", "content": retry_prompt}]
    try:
        retry_raw = llm.complete(retry_messages, temperature=0.2, response_format="json")
    except Exception:
        # Drop the failed nodes entirely on LLM error — all originally-failed
        return accepted, len(failed)

    retry_nodes = parse_extraction(retry_raw)
    n_still_failed = len(failed)  # start assuming all failed still fail
    if retry_nodes:
        retry_accepted, retry_failed = _run_gate(retry_nodes, segment.text)
        n_still_failed = len(retry_failed)
        # Only add retry nodes whose source_quote is not already in accepted
        existing_quotes = {n.source_quote for n in accepted}
        for rn in retry_accepted:
            if rn.source_quote not in existing_quotes:
                accepted.append(rn)
                existing_quotes.add(rn.source_quote)

    return accepted, n_still_failed


# ---------------------------------------------------------------------------
# Self-check pass
# ---------------------------------------------------------------------------

_SELF_CHECK_TEMPLATE = """\
You are reviewing extracted mathematical nodes for over-claiming.

## Segment text
```
{segment_text}
```

## Extracted nodes
{nodes_json}

## Task
Identify any nodes whose `text` asserts something that is NOT supported by or goes beyond the segment text.
Return ONLY JSON with this structure:
{{"suspicious_labels": ["<label1>", "<label2>", ...]}}

Return an empty list if all nodes are well-supported: {{"suspicious_labels": []}}
"""


def self_check(
    segment: Segment,
    nodes: list[ExtractedNode],
    llm,
) -> list[ExtractedNode]:
    """Mark nodes that over-claim beyond the segment text as ``"suspicious"``.

    This is a cheap second LLM pass.  On any failure (parse error, LLM error,
    garbled output) all nodes are returned unchanged.
    """
    if not nodes:
        return nodes

    # Build a stable key for each node: label if present, else "(node-{i})"
    idx_to_node = {
        (n.label if n.label else f"(node-{i})"): n
        for i, n in enumerate(nodes)
    }

    nodes_summary = [
        {"label": key, "text": n.text}
        for key, n in idx_to_node.items()
    ]
    prompt = _SELF_CHECK_TEMPLATE.format(
        segment_text=segment.text,
        nodes_json=json.dumps(nodes_summary, ensure_ascii=False, indent=2),
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        raw = llm.complete(messages, temperature=0.2, response_format="json")
    except Exception:
        return nodes

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return nodes

    if not isinstance(data, dict):
        return nodes

    suspicious_labels = data.get("suspicious_labels")
    if not isinstance(suspicious_labels, list):
        return nodes

    label_set = {str(lbl) for lbl in suspicious_labels if lbl}
    for key, node in idx_to_node.items():
        if key in label_set:
            node.status = "suspicious"

    return nodes
