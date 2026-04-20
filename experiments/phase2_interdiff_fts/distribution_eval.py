"""
Direct distribution-distance evaluation: M6 vs M16 vs real.

Computes three distribution-level metrics that do NOT depend on a
downstream predictor:

  1. MMD^2 (Gaussian kernel): Maximum Mean Discrepancy between real
     and synth panel-window distributions. Lower = closer.
  2. KS statistic per channel: 1D Kolmogorov-Smirnov between real
     and synth marginals for each of (log_ret, log_hc, log_lc, log_oc).
  3. Wasserstein-1 per channel: 1D earth-mover distance.

These measure "how close is the synth distribution to real" in the
strongest sense, independent of whether any particular downstream
model can exploit it.

Usage:
    python distribution_eval.py \
        --panel data/csi800_2015_2024.npz \
        --m6 ckpts/M0_m6_csi800_step20000.samples.npz \
        --m16 ckpts/M0_m16_lev_hinge_w20_step20000.samples.npz \
        --n-trajs 2000
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from scipy import stats

from panel_windows import derive_panel_returns


# ────────────────────────────────────────────────────────────────
# Trajectory sampling
# ────────────────────────────────────────────────────────────────

def extract_real_trajs(
    panel_npz: str, L: int, n_trajs: int,
    time_range: tuple[str, str] | None, rng: np.random.Generator,
) -> np.ndarray:
    """
    Return (n_trajs, L, C) per-stock trajectories from real panel.
    Uniform sample over (stock, start_day) where entire L-window is valid.
    """
    d = np.load(panel_npz, allow_pickle=True)
    panel = d["panel"]
    fields = d["fields"].tolist()
    dates = d["dates"].astype(str)
    returns = derive_panel_returns(panel, fields)  # (N, T-1, C)
    ret_dates = dates[1:]
    if time_range is not None:
        mask = (ret_dates >= time_range[0]) & (ret_dates <= time_range[1])
        returns = returns[:, mask, :]
    N, T, C = returns.shape
    valid = np.isfinite(returns).all(axis=2)

    # For each start, find stocks with full L-window valid
    out = np.empty((n_trajs, L, C), dtype=np.float32)
    count = 0
    tries = 0
    max_tries = n_trajs * 20
    while count < n_trajs and tries < max_tries:
        tries += 1
        s = int(rng.integers(0, T - L + 1))
        ok = np.where(valid[:, s:s + L].all(axis=1))[0]
        if ok.size == 0:
            continue
        stock = int(rng.choice(ok))
        out[count] = returns[stock, s:s + L, :]
        count += 1
    return out[:count]


def extract_synth_trajs(
    samples_npz: str, L: int, n_trajs: int, rng: np.random.Generator,
) -> np.ndarray:
    """
    Return (n_trajs, L, C) per-stock trajectories from synth samples.
    Panels are (n_panels, K, T_syn, C). Sample random (panel, stock, start).
    """
    d = np.load(samples_npz, allow_pickle=True)
    panels = d["panels_denorm"].astype(np.float32)  # (n_p, K, T, C)
    n_p, K, T_syn, C = panels.shape
    out = np.empty((n_trajs, L, C), dtype=np.float32)
    for i in range(n_trajs):
        p = int(rng.integers(0, n_p))
        st = int(rng.integers(0, T_syn - L + 1))
        stock = int(rng.integers(0, K))
        out[i] = panels[p, stock, st:st + L, :]
    return out


# ────────────────────────────────────────────────────────────────
# MMD (Gaussian kernel, median-bandwidth heuristic)
# ────────────────────────────────────────────────────────────────

def _gaussian_kernel_matrix(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """K[i,j] = exp(-||X[i] - Y[j]||^2 / (2 sigma^2))."""
    # ||X - Y||^2 = ||X||^2 + ||Y||^2 - 2 X·Y
    X2 = (X * X).sum(axis=1, keepdims=True)  # (n, 1)
    Y2 = (Y * Y).sum(axis=1, keepdims=True).T  # (1, m)
    cross = X @ Y.T
    d2 = X2 + Y2 - 2 * cross
    d2 = np.clip(d2, a_min=0.0, a_max=None)
    return np.exp(-d2 / (2 * sigma * sigma))


def mmd_squared(X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> tuple[float, float]:
    """
    Unbiased MMD^2 between X (n, d) and Y (m, d) with Gaussian kernel.
    Returns (mmd^2, sigma_used).

    sigma = median heuristic: median of pairwise distances in X ∪ Y.
    """
    if sigma is None:
        # Sub-sample for speed if large
        Z = np.concatenate([X, Y], axis=0)
        idx = np.random.default_rng(0).choice(len(Z), size=min(1000, len(Z)), replace=False)
        Z_sub = Z[idx]
        d2 = ((Z_sub[:, None, :] - Z_sub[None, :, :]) ** 2).sum(axis=-1)
        sigma = float(np.sqrt(np.median(d2[d2 > 0])))
    n, m = len(X), len(Y)
    Kxx = _gaussian_kernel_matrix(X, X, sigma)
    Kyy = _gaussian_kernel_matrix(Y, Y, sigma)
    Kxy = _gaussian_kernel_matrix(X, Y, sigma)
    # Unbiased estimator: exclude diagonal for Kxx and Kyy
    mmd2 = (
        (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
        + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
        - 2 * Kxy.mean()
    )
    return float(mmd2), sigma


# ────────────────────────────────────────────────────────────────
# 1D marginal tests (KS + Wasserstein-1)
# ────────────────────────────────────────────────────────────────

def ks_per_channel(X_real: np.ndarray, X_syn: np.ndarray) -> list[dict]:
    """For each channel c, KS test between real and synth marginals
    (flattened across all trajectories and timesteps)."""
    out = []
    for c in range(X_real.shape[-1]):
        a = X_real[..., c].ravel()
        b = X_syn[..., c].ravel()
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        ks_stat, p = stats.ks_2samp(a, b)
        w1 = stats.wasserstein_distance(a, b)
        out.append({"channel": c, "ks_stat": float(ks_stat),
                    "p_value": float(p),
                    "wasserstein_1": float(w1)})
    return out


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/csi800_2015_2024.npz")
    ap.add_argument("--m6", default="ckpts/M0_m6_csi800_step20000.samples.npz")
    ap.add_argument("--m16", default="ckpts/M0_m16_lev_hinge_w20_step20000.samples.npz")
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--n-trajs", type=int, default=2000)
    ap.add_argument("--train-end", default="2022-12-31")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    print(f"[dist-eval] L={args.L}  n_trajs={args.n_trajs}")

    # Real trajectories from training period (to match what generators saw)
    print(f"[dist-eval] extracting real trajs from {args.panel}")
    real = extract_real_trajs(args.panel, args.L, args.n_trajs,
                               ("2015-01-05", args.train_end), rng)
    print(f"[dist-eval] real: {real.shape}")

    print(f"[dist-eval] extracting M6 synth from {args.m6}")
    m6 = extract_synth_trajs(args.m6, args.L, args.n_trajs, rng)
    print(f"[dist-eval] M6:   {m6.shape}")

    print(f"[dist-eval] extracting M16 synth from {args.m16}")
    m16 = extract_synth_trajs(args.m16, args.L, args.n_trajs, rng)
    print(f"[dist-eval] M16:  {m16.shape}")

    # MMD on full flattened trajectories (L*C = 128 dim)
    print("\n=== MMD^2 (Gaussian kernel, median bandwidth) ===")
    real_flat = real.reshape(real.shape[0], -1)
    m6_flat = m6.reshape(m6.shape[0], -1)
    m16_flat = m16.reshape(m16.shape[0], -1)
    # Sample-size cap for speed
    N = min(1500, len(real_flat), len(m6_flat), len(m16_flat))
    real_s = real_flat[:N]
    m6_s = m6_flat[:N]
    m16_s = m16_flat[:N]

    mmd_real_m6, s1 = mmd_squared(real_s, m6_s)
    mmd_real_m16, s2 = mmd_squared(real_s, m16_s)
    mmd_real_real, s3 = mmd_squared(real_s[:N // 2], real_s[N // 2:])
    print(f"  real vs real (baseline):  {mmd_real_real:+.6f}  (sigma={s3:.3f})")
    print(f"  real vs M6:               {mmd_real_m6:+.6f}  (sigma={s1:.3f})")
    print(f"  real vs M16:              {mmd_real_m16:+.6f}  (sigma={s2:.3f})")
    better = "M6" if mmd_real_m6 < mmd_real_m16 else "M16"
    print(f"  -> closer to real: {better}")
    print(f"     (diff = {abs(mmd_real_m6 - mmd_real_m16):+.6f})")

    # KS + Wasserstein per channel
    print("\n=== KS + Wasserstein-1 per channel (vs real) ===")
    ch_names = ["log_ret", "log_hc", "log_lc", "log_oc"]
    ks6 = ks_per_channel(real, m6)
    ks16 = ks_per_channel(real, m16)
    print(f"{'channel':>9s}   {'KS(M6)':>9s}   {'KS(M16)':>9s}   {'W1(M6)':>9s}   {'W1(M16)':>9s}   winner")
    for c in range(4):
        k6 = ks6[c]["ks_stat"]; k16 = ks16[c]["ks_stat"]
        w6 = ks6[c]["wasserstein_1"]; w16 = ks16[c]["wasserstein_1"]
        win_ks = "M6" if k6 < k16 else "M16"
        win_w = "M6" if w6 < w16 else "M16"
        print(f"{ch_names[c]:>9s}   {k6:>9.5f}   {k16:>9.5f}   {w6:>9.5f}   {w16:>9.5f}   KS:{win_ks}/W:{win_w}")

    # Summary
    ks_m6_wins = sum(1 for c in range(4) if ks6[c]["ks_stat"] < ks16[c]["ks_stat"])
    w_m6_wins = sum(1 for c in range(4) if ks6[c]["wasserstein_1"] < ks16[c]["wasserstein_1"])
    print(f"\n  M6 wins {ks_m6_wins}/4 channels on KS,  {w_m6_wins}/4 on Wasserstein-1")


if __name__ == "__main__":
    main()
