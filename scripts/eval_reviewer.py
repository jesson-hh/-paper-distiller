"""Manual reviewer quality eval: precision / recall / F1 of PROBLEM detection.

Usage (manual, NOT CI)::

    python scripts/eval_reviewer.py

Requirements:
    - PD_API_KEY, PD_BASE_URL, PD_MODEL env vars (or a .env file in the repo root).
    - ``pip install paper-distiller`` or ``pip install -e ".[dev]"`` in the active venv.

This script is intentionally EXCLUDED from pytest CI.  Proof-judgment quality
cannot be unit-tested — LLM judges are near-chance on proof soundness ("Proof or
Bluff?", USAMO 2025).  Run it manually after every significant change to the
review prompt to track regression.

What it does:
    1. Load a small hand-labeled fixture (FIXTURE below).
    2. Build an in-memory ProofStore, insert fixture nodes (no real papers needed).
    3. Call ``review_node`` against a real LLM for each fixture item.
    4. Compare predicted label against gold_label (ok vs problem).
    5. Print per-item results + aggregate precision / recall / F1.

PROBLEM label set: suspicious, gap, unsupported  (same as reviewer.PROBLEM)
gold_label ∈ {"ok", "problem"}.
"""
from __future__ import annotations

import os
import sys
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Fixture: 5 hand-labeled items.
# Replace / extend with real paper excerpts for a more rigorous eval.
# ---------------------------------------------------------------------------

FIXTURE = [
    {
        "label": "Theorem 1.1",
        "kind": "theorem",
        "statement": "For all x in [0,1], E[X] <= C * sqrt(n).",
        "source_quote": "For all x in [0,1], E[X] <= C * sqrt(n).",
        "techniques": ["Bernstein", "concentration"],
        "gold_label": "ok",
        "note": "Standard, well-grounded theorem statement (a premise, not a derived step).",
    },
    {
        "label": "Step (b)",
        "kind": "proof_step",
        "statement": "Applying Hölder's inequality, we get ||fg||_1 <= ||f||_2 * ||g||_2.",
        "source_quote": "Applying Hölder's inequality, we get ||fg||_1 <= ||f||_2 * ||g||_2.",
        "techniques": ["Hölder"],
        "gold_label": "ok",
        "note": "Correct application of Hölder.",
    },
    {
        "label": "Claim 3.2",
        "kind": "claim",
        "statement": "The sum converges absolutely for all real x.",
        "source_quote": "The sum converges for all real x.",
        "techniques": [],
        "gold_label": "problem",
        "note": "Quote says 'converges' but claim adds 'absolutely' — unsupported leap.",
    },
    {
        "label": "Lemma 4.1",
        "kind": "proof_step",
        "statement": "By Theorem 4.3 (proved below), the error is O(1/n).",
        "source_quote": "By Theorem 4.3 (proved below), the error is O(1/n).",
        "techniques": [],
        "gold_label": "problem",
        "note": "Proof step forward-referencing Theorem 4.3 (not yet proved) — gap.",
    },
    {
        "label": "Corollary 2.1",
        "kind": "proof_step",
        "statement": "The bound follows by combining the previous three lemmas.",
        "source_quote": "The bound follows by combining the previous three lemmas.",
        "techniques": [],
        "gold_label": "problem",
        "note": "Vague 'combining' with no explicit arithmetic — suspicious step.",
    },
]


# ---------------------------------------------------------------------------
# Metric function (pure Python, no LLM — also tested in CI)
# ---------------------------------------------------------------------------

class ConfusionMatrix(NamedTuple):
    tp: int
    fp: int
    tn: int
    fn: int


def compute_metrics(gold: list[str], predicted: list[str]) -> dict:
    """Compute precision / recall / F1 for PROBLEM detection.

    Args:
        gold: list of "ok" or "problem" ground-truth labels.
        predicted: list of "ok" or "problem" model predictions (same length).

    Returns:
        dict with keys: precision, recall, f1, tp, fp, tn, fn, n.

    PROBLEM is the positive class.  A reviewer.PROBLEM label (suspicious / gap /
    unsupported) maps to "problem"; ok / unstated / extracted map to "ok".
    """
    if len(gold) != len(predicted):
        raise ValueError(f"Length mismatch: gold={len(gold)}, predicted={len(predicted)}")

    tp = fp = tn = fn = 0
    for g, p in zip(gold, predicted):
        if g == "problem" and p == "problem":
            tp += 1
        elif g == "ok" and p == "problem":
            fp += 1
        elif g == "ok" and p == "ok":
            tn += 1
        elif g == "problem" and p == "ok":
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n": len(gold),
    }


# ---------------------------------------------------------------------------
# Label normalisation: reviewer labels → binary "problem" / "ok"
# ---------------------------------------------------------------------------

_PROBLEM_LABELS = {"suspicious", "gap", "unsupported"}


def _to_binary(label: str) -> str:
    return "problem" if label in _PROBLEM_LABELS else "ok"


# ---------------------------------------------------------------------------
# Main eval runner
# ---------------------------------------------------------------------------

def run_eval() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("PD_API_KEY")
    base_url = os.getenv("PD_BASE_URL")
    model = os.getenv("PD_MODEL")
    if not (api_key and base_url and model):
        print("ERROR: PD_API_KEY / PD_BASE_URL / PD_MODEL not set.", file=sys.stderr)
        sys.exit(1)

    from paper_distiller.llm.openai_compatible import LLMClient
    from paper_distiller.proofs.store import ProofStore, Node
    from paper_distiller.proofgraph.reviewer import review_node

    llm = LLMClient(api_key=api_key, base_url=base_url, model=model)

    # Build an in-memory store for the fixture nodes
    store = ProofStore(":memory:")

    node_ids: list[int] = []
    for item in FIXTURE:
        nid = store.add_node(Node(
            paper_arxiv_id="eval-fixture",
            kind=item.get("kind", "proof_step"),
            text=item["statement"],
            label=item["label"],
            source_quote=item["source_quote"],
            techniques=item.get("techniques", []),
        ))
        node_ids.append(nid)

    # Run review_node for each
    gold: list[str] = []
    predicted: list[str] = []
    results = []

    print(f"\n{'='*60}")
    print(f"Reviewer eval — {len(FIXTURE)} items — model: {model}")
    print(f"{'='*60}\n")

    for idx, (item, nid) in enumerate(zip(FIXTURE, node_ids)):
        node = store.get_node(nid)
        rr = review_node(store, node, llm)
        gold_bin = item["gold_label"]   # already "ok" or "problem"
        pred_bin = _to_binary(rr.label)
        correct = gold_bin == pred_bin

        print(
            f"[{idx+1}/{len(FIXTURE)}] {item['label']!r:20s}  "
            f"gold={gold_bin:7s}  pred={rr.label:12s} ({pred_bin:7s})  "
            f"conf={rr.confidence:.2f}  {'OK' if correct else 'WRONG'}"
        )
        if not correct:
            print(f"        reason: {rr.reason[:120]}")
            print(f"        note:   {item['note']}")

        gold.append(gold_bin)
        predicted.append(pred_bin)
        results.append({"label": item["label"], "gold": gold_bin, "pred": pred_bin,
                        "reviewer_label": rr.label, "reason": rr.reason})

    store.close()

    metrics = compute_metrics(gold, predicted)
    print(f"\n{'='*60}")
    print(f"  precision = {metrics['precision']:.4f}  "
          f"recall = {metrics['recall']:.4f}  "
          f"F1 = {metrics['f1']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}  "
          f"(n={metrics['n']})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_eval()
