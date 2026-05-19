"""Model pricing table + cost estimator.

Rates are CNY per million tokens, approximate (post-2026-Q1 Aliyun and DeepSeek
public pricing). Env vars `PD_PRICE_IN_CNY_PER_M` / `PD_PRICE_OUT_CNY_PER_M`
override both directions for the entire session — useful when a provider
changes pricing or when running against a locally-hosted model.
"""

from __future__ import annotations

import os
import sys

PRICING_PER_M_TOKENS_CNY: dict[str, dict[str, float]] = {
    "qwen-plus": {"in": 0.8, "out": 2.0},
    "qwen-turbo": {"in": 0.3, "out": 0.6},
    "qwen-max": {"in": 20.0, "out": 60.0},
    "deepseek-chat": {"in": 0.5, "out": 1.5},
    "deepseek-reasoner": {"in": 1.0, "out": 4.0},
}

_DEFAULT_RATE = {"in": 1.0, "out": 3.0}
_warned_models: set[str] = set()


def estimate_cost_cny(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return estimated cost in CNY for a number of in/out tokens at `model`'s rate."""
    in_env = os.getenv("PD_PRICE_IN_CNY_PER_M")
    out_env = os.getenv("PD_PRICE_OUT_CNY_PER_M")
    if in_env is not None and out_env is not None:
        in_rate, out_rate = float(in_env), float(out_env)
    else:
        rate = PRICING_PER_M_TOKENS_CNY.get(model)
        if rate is None:
            if model not in _warned_models:
                print(
                    f"[pricing] unknown model {model!r}, using conservative "
                    f"¥{_DEFAULT_RATE['in']}/¥{_DEFAULT_RATE['out']} per M tokens.",
                    file=sys.stderr,
                )
                _warned_models.add(model)
            rate = _DEFAULT_RATE
        in_rate, out_rate = rate["in"], rate["out"]
    return tokens_in / 1_000_000.0 * in_rate + tokens_out / 1_000_000.0 * out_rate
