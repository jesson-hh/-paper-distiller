"""Deterministic, LLM-free reading primitives for the proof-graph pipeline.

- segment(text): split a paper's plain text into ordered Segments, flagging
  theorem-statement and proof-block regions. This is the coverage denominator
  for "don't skim" — every segment must later be visited.
- verify_quote(quote, segment_text): the grounding gate (Task 9).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Segment:
    id: int
    kind_hint: str          # "prose" | "theorem" | "proof" | "definition" | "heading"
    section: str | None     # nearest preceding heading label, e.g. "2 Main Result"
    text: str
    char_start: int
    char_end: int
    is_proof_block: bool


# A heading line: "1 Introduction", "2.1 Setup", "Appendix A", etc.
_HEADING_RE = re.compile(r"^\s*(\d+(\.\d+)*\s+\S.*|Appendix\s+\S.*)$")
# Start of a theorem-like statement.
_THEOREM_RE = re.compile(
    r"^\s*(Theorem|Lemma|Proposition|Corollary|Claim|Definition)\b", re.IGNORECASE)
_PROOF_START_RE = re.compile(r"^\s*Proof\b", re.IGNORECASE)
# End-of-proof markers: QED box, "QED", "q.e.d."
_PROOF_END_RE = re.compile(r"(□|\bQED\b|\bq\.?e\.?d\.?\b)", re.IGNORECASE)


def _classify(block: str) -> str:
    head = block.lstrip()
    if _PROOF_START_RE.match(head):
        return "proof"
    if _THEOREM_RE.match(head):
        first = head.split(None, 1)[0].lower()
        return "definition" if first.startswith("defin") else "theorem"
    if _HEADING_RE.match(head.splitlines()[0] if head else ""):
        return "heading"
    return "prose"


def segment(text: str) -> list[Segment]:
    """Split into structural blocks — a new block starts at each heading /
    Theorem-like / Proof line, and at blank-line paragraph breaks. Classify
    each block and record char offsets so downstream code can reconstruct and
    ground to source. This list is the coverage denominator for "don't skim"."""
    if not text or not text.strip():
        return []
    lines = text.splitlines(keepends=True)
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln)

    segments: list[Segment] = []
    cur: list[int] = []
    state = {"sid": 0, "section": None}

    def _is_boundary(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        return bool(_HEADING_RE.match(s) or _THEOREM_RE.match(s)
                    or _PROOF_START_RE.match(s))

    def _flush() -> None:
        if not cur:
            return
        start = offsets[cur[0]]
        end = offsets[cur[-1]] + len(lines[cur[-1]])
        block = text[start:end]
        cur.clear()
        if not block.strip():
            return
        kind = _classify(block)
        if kind == "heading":
            state["section"] = block.strip().splitlines()[0].strip()
        segments.append(Segment(
            id=state["sid"], kind_hint=kind, section=state["section"],
            text=block, char_start=start, char_end=end,
            is_proof_block=bool(_PROOF_START_RE.match(block.lstrip())),
        ))
        state["sid"] += 1

    for i, ln in enumerate(lines):
        if not ln.strip():
            _flush()
            continue
        if cur and _is_boundary(ln):
            _flush()
        cur.append(i)
    _flush()
    return segments
