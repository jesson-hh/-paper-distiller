"""CI-safe tests for scripts/eval_reviewer.py.

These tests do NOT call any LLM.  They verify:
  1. The script module can be imported without errors.
  2. The FIXTURE parses as expected (non-empty list of dicts with required keys).
  3. The ``compute_metrics`` function returns correct precision / recall / F1
     on a synthetic confusion matrix.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Make scripts/ importable so we can import eval_reviewer as a module.
_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Import smoke
# ---------------------------------------------------------------------------

def test_eval_reviewer_importable():
    """scripts/eval_reviewer.py imports without errors."""
    mod = importlib.import_module("eval_reviewer")
    assert mod is not None


def test_eval_reviewer_has_expected_symbols():
    """compute_metrics, FIXTURE, and run_eval are all present."""
    mod = importlib.import_module("eval_reviewer")
    assert callable(mod.compute_metrics)
    assert isinstance(mod.FIXTURE, list)
    assert callable(mod.run_eval)


# ---------------------------------------------------------------------------
# Fixture validation
# ---------------------------------------------------------------------------

def test_fixture_non_empty():
    mod = importlib.import_module("eval_reviewer")
    assert len(mod.FIXTURE) >= 3, "Fixture must have at least 3 items for a meaningful eval"


def test_fixture_has_required_keys():
    mod = importlib.import_module("eval_reviewer")
    required = {"label", "statement", "source_quote", "gold_label"}
    for item in mod.FIXTURE:
        missing = required - set(item.keys())
        assert not missing, f"Fixture item {item.get('label')!r} missing keys: {missing}"


def test_fixture_gold_labels_valid():
    mod = importlib.import_module("eval_reviewer")
    valid = {"ok", "problem"}
    for item in mod.FIXTURE:
        assert item["gold_label"] in valid, (
            f"Fixture item {item.get('label')!r} has invalid gold_label {item['gold_label']!r}"
        )


# ---------------------------------------------------------------------------
# compute_metrics — pure function, no LLM
# ---------------------------------------------------------------------------

def test_compute_metrics_perfect_recall():
    from eval_reviewer import compute_metrics
    gold = ["problem", "problem", "ok", "ok"]
    pred = ["problem", "problem", "ok", "ok"]
    m = compute_metrics(gold, pred)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["tn"] == 2
    assert m["fn"] == 0


def test_compute_metrics_all_wrong():
    from eval_reviewer import compute_metrics
    gold = ["problem", "problem", "ok", "ok"]
    pred = ["ok", "ok", "problem", "problem"]
    m = compute_metrics(gold, pred)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["tp"] == 0
    assert m["fp"] == 2
    assert m["tn"] == 0
    assert m["fn"] == 2


def test_compute_metrics_partial():
    from eval_reviewer import compute_metrics
    # 2 correct problems, 1 missed, 1 false alarm
    gold =  ["problem", "problem", "ok",  "problem"]
    pred =  ["problem", "ok",      "problem", "problem"]
    m = compute_metrics(gold, pred)
    # tp=2, fp=1, fn=1, tn=0
    assert m["tp"] == 2
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 0
    # precision = 2/3, recall = 2/3, f1 = 2/3
    assert abs(m["precision"] - 2/3) < 1e-4
    assert abs(m["recall"] - 2/3) < 1e-4
    assert abs(m["f1"] - 2/3) < 1e-4


def test_compute_metrics_no_positives_in_gold():
    from eval_reviewer import compute_metrics
    gold = ["ok", "ok", "ok"]
    pred = ["ok", "ok", "ok"]
    m = compute_metrics(gold, pred)
    # No positive class in gold → precision/recall/F1 are 0.0 by convention
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["tp"] == 0
    assert m["tn"] == 3


def test_compute_metrics_length_mismatch_raises():
    from eval_reviewer import compute_metrics
    import pytest
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_metrics(["ok", "problem"], ["ok"])
