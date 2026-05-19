"""Tests for llm.pricing."""

from __future__ import annotations

import pytest


def test_estimate_cost_qwen_plus():
    from paper_distiller.llm.pricing import estimate_cost_cny

    cost = estimate_cost_cny("qwen-plus", tokens_in=1_000_000, tokens_out=500_000)
    assert cost == pytest.approx(0.8 + 1.0)


def test_estimate_cost_zero_tokens():
    from paper_distiller.llm.pricing import estimate_cost_cny
    assert estimate_cost_cny("qwen-plus", 0, 0) == 0.0


def test_estimate_cost_unknown_model_uses_conservative_default():
    from paper_distiller.llm.pricing import estimate_cost_cny, _warned_models

    _warned_models.discard("never-heard-of-it")
    cost = estimate_cost_cny("never-heard-of-it", 1_000_000, 1_000_000)
    assert cost == pytest.approx(1.0 + 3.0)


def test_env_override_input_rate(monkeypatch):
    from paper_distiller.llm.pricing import estimate_cost_cny

    monkeypatch.setenv("PD_PRICE_IN_CNY_PER_M", "5.0")
    monkeypatch.setenv("PD_PRICE_OUT_CNY_PER_M", "10.0")
    cost = estimate_cost_cny("qwen-plus", 1_000_000, 1_000_000)
    assert cost == pytest.approx(5.0 + 10.0)


def test_known_models_present():
    from paper_distiller.llm.pricing import PRICING_PER_M_TOKENS_CNY

    for name in ("qwen-plus", "qwen-turbo", "qwen-max", "deepseek-chat"):
        assert name in PRICING_PER_M_TOKENS_CNY
        entry = PRICING_PER_M_TOKENS_CNY[name]
        assert {"in", "out"}.issubset(entry.keys())
        assert entry["in"] > 0 and entry["out"] > 0
