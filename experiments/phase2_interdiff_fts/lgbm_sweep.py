"""
Stronger α-sweep baseline using LightGBM + engineered features.

Same mixing protocol as alpha_sweep.py, but replaces the 138k-param
transformer with a LightGBM model on ~20 engineered features derived
from the 4-channel (log_ret, log_hc, log_lc, log_oc) history.

Features (all computable from both real and synth, since synth has
the same 4 channels):

  Momentum (log-ret lag horizons):
    - ret_1, ret_3, ret_5, ret_10, ret_20 (cumulative past-k log-ret)
  Realized volatility (rolling std of log-ret):
    - rvol_5, rvol_10, rvol_20
  Range (OHLC spread proxies, mean over window):
    - range_hl_5, range_hl_20 (mean of log_hc - log_lc over past k)
    - range_oc_5, range_oc_20 (mean abs log_oc)
  Skew / kurt of recent returns:
    - skew_20, kurt_20 (third/fourth standardized moment of past 20 ret)
  Cross-sectional rank of current-day return (computed at train time):
    - rank_ret_today (quantile in [0, 1] within the panel)
  Sign statistics:
    - n_neg_5: count of negative days in past 5
    - sum_neg_ret_10: sum of max(-ret, 0) over past 10

Label: next-day log_ret.

Evaluation: cross-sectional Spearman rank-IC per day, averaged over
held-out 2023.

Usage:
    python lgbm_sweep.py \
        --panel data/csi800_2015_2024.npz \
        --synth ckpts/M0_m6_csi800_step20000.samples.npz \
        --alphas 0,0.1,0.25,0.5,0.75,0.9 \
        --seeds 0,1,2,3,4 \
        --n-train 100000 \
        --test-year 2023
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

from panel_windows import derive_panel_returns


# ────────────────────────────────────────────────────────────────
# Feature engineering on a (k_stocks, L+1, C) panel window
# ────────────────────────────────────────────────────────────────

def extract_features(window: np.ndarray, L: int) -> np.ndarray:
    """
    window: (k, L+1, C) with channels [log_ret, log_hc, log_lc, log_oc]
    The last time step (index L) is the LABEL day.
    Features are computed on the first L time steps (history).

    Returns: (k, F) feature matrix where F is number of features.
    """
    hist = window[:, :L, :]  # (k, L, 4)
    log_ret = hist[:, :, 0]   # (k, L)
    log_hc = hist[:, :, 1]
    log_lc = hist[:, :, 2]
    log_oc = hist[:, :, 3]

    feats = []
    # Cumulative past-k log returns (momentum)
    for k in (1, 3, 5, 10, 20):
        if k > L:
            feats.append(np.zeros(log_ret.shape[0], dtype=np.float32))
        else:
            feats.append(log_ret[:, -k:].sum(axis=1))
    # Realized volatility
    for k in (5, 10, 20):
        if k > L:
            feats.append(np.zeros(log_ret.shape[0], dtype=np.float32))
        else:
            feats.append(log_ret[:, -k:].std(axis=1))
    # HL range means
    for k in (5, 20):
        if k > L:
            feats.append(np.zeros(log_ret.shape[0], dtype=np.float32))
        else:
            rng = (log_hc[:, -k:] - log_lc[:, -k:])
            feats.append(rng.mean(axis=1))
    # OC absolute range
    for k in (5, 20):
        if k > L:
            feats.append(np.zeros(log_ret.shape[0], dtype=np.float32))
        else:
            feats.append(np.abs(log_oc[:, -k:]).mean(axis=1))
    # Skew / kurt of past 20 returns (standardised)
    k20 = min(20, L)
    r = log_ret[:, -k20:]
    mu = r.mean(axis=1, keepdims=True)
    std = r.std(axis=1, keepdims=True) + 1e-8
    z = (r - mu) / std
    feats.append((z ** 3).mean(axis=1))
    feats.append((z ** 4).mean(axis=1) - 3.0)
    # Cross-sectional rank of most-recent log_ret (within the panel)
    last = log_ret[:, -1]
    rank = last.argsort().argsort().astype(np.float32) / max(len(last) - 1, 1)
    feats.append(rank)
    # Number of negative days in past 5
    k5 = min(5, L)
    feats.append((log_ret[:, -k5:] < 0).sum(axis=1).astype(np.float32))
    # Sum of negative magnitudes in past 10 (downside pressure)
    k10 = min(10, L)
    neg_only = np.clip(-log_ret[:, -k10:], 0.0, None)
    feats.append(neg_only.sum(axis=1))

    return np.stack(feats, axis=1).astype(np.float32)  # (k, F)


FEATURE_NAMES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "rvol_5", "rvol_10", "rvol_20",
    "hlrng_5", "hlrng_20",
    "ocabs_5", "ocabs_20",
    "skew_20", "kurt_20",
    "rank_ret_today",
    "n_neg_5", "sum_neg_ret_10",
]


# ────────────────────────────────────────────────────────────────
# Data assembly
# ────────────────────────────────────────────────────────────────

def load_real_returns(panel_npz: str) -> dict:
    d = np.load(panel_npz, allow_pickle=True)
    returns = derive_panel_returns(d["panel"], d["fields"].tolist())
    ret_dates = d["dates"].astype(str)[1:]
    valid = np.isfinite(returns).all(axis=2)
    returns_clean = np.where(valid[:, :, None], returns, 0.0).astype(np.float32)
    return {"returns": returns_clean, "valid": valid, "dates": ret_dates,
            "codes": d["codes"]}


def build_real_dataset(
    real: dict, N_pick: int, L: int, time_range: tuple[str, str],
    n_samples: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample n_samples panel-windows of (N_pick stocks × L history + 1 label)
    from the real panel restricted to time_range, return flattened
    (samples, features) + (samples,) labels.
    """
    dates = real["dates"]
    in_range = (dates >= time_range[0]) & (dates <= time_range[1])
    idx_range = np.where(in_range)[0]
    if len(idx_range) < L + 1:
        raise ValueError("time range too short")

    returns = real["returns"]
    valid = real["valid"]
    N, T, C = returns.shape
    # For efficiency precompute which (t, stock) have fully-valid [t-L, t+1)
    # window.
    start_min = idx_range[0]
    start_max = idx_range[-1] - L  # last valid start s.t. s+L is also in range

    feats_out = []
    labels_out = []
    target = n_samples
    max_tries = n_samples * 10
    tries = 0
    while len(labels_out) * N_pick < target and tries < max_tries:
        tries += 1
        s = int(rng.integers(start_min, start_max + 1))
        # Check which stocks have fully valid window [s, s+L+1)
        ok = np.where(valid[:, s:s + L + 1].all(axis=1))[0]
        if ok.size < N_pick:
            continue
        picks = rng.choice(ok, N_pick, replace=False)
        win = returns[picks, s:s + L + 1, :]  # (N_pick, L+1, C)
        f = extract_features(win, L)           # (N_pick, F)
        y = win[:, L, 0]                        # (N_pick,)
        feats_out.append(f)
        labels_out.append(y)

    X = np.concatenate(feats_out, axis=0)
    y = np.concatenate(labels_out, axis=0)
    return X.astype(np.float32), y.astype(np.float32)


def build_synth_dataset(
    synth_panels: np.ndarray,  # (n_panels, K, T_syn, C)
    N_pick: int, L: int, n_samples: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_panels, K, T_syn, C = synth_panels.shape
    feats_out, labels_out = [], []
    need_rows = n_samples
    while sum(f.shape[0] for f in feats_out) < need_rows:
        p = int(rng.integers(0, n_panels))
        start = int(rng.integers(0, T_syn - L - 1))
        N_pick_eff = min(N_pick, K)
        if N_pick_eff < K:
            stock_idx = rng.choice(K, N_pick_eff, replace=False)
        else:
            stock_idx = np.arange(K)
        win = synth_panels[p, stock_idx, start:start + L + 1, :]
        f = extract_features(win, L)
        y = win[:, L, 0]
        feats_out.append(f)
        labels_out.append(y)
    X = np.concatenate(feats_out, axis=0)
    y = np.concatenate(labels_out, axis=0)
    return X.astype(np.float32), y.astype(np.float32)


def mix_datasets(
    Xr: np.ndarray, yr: np.ndarray,
    Xs: np.ndarray, ys: np.ndarray,
    alpha: float, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Form a training set with fraction alpha from synth, (1-alpha) from real."""
    n_total = len(yr) + len(ys)
    # Aim for total set = min(len real, len synth) * (something). Just use len(yr)
    n_want = len(yr)
    n_synth = int(round(alpha * n_want))
    n_real = n_want - n_synth
    # Sample with replacement if needed
    if len(yr) >= n_real:
        idx_r = rng.choice(len(yr), n_real, replace=False)
    else:
        idx_r = rng.choice(len(yr), n_real, replace=True)
    if len(ys) >= n_synth:
        idx_s = rng.choice(len(ys), n_synth, replace=False)
    else:
        idx_s = rng.choice(len(ys), n_synth, replace=True)
    X = np.concatenate([Xr[idx_r], Xs[idx_s]], axis=0)
    y = np.concatenate([yr[idx_r], ys[idx_s]], axis=0)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ────────────────────────────────────────────────────────────────
# Test-time evaluation (per-day rank-IC)
# ────────────────────────────────────────────────────────────────

def eval_rank_ic(
    model, real: dict, N_pick: int, L: int,
    test_range: tuple[str, str],
) -> dict:
    returns = real["returns"]
    valid = real["valid"]
    dates = real["dates"]
    in_range = (dates >= test_range[0]) & (dates <= test_range[1])
    idx_range = np.where(in_range)[0]
    # For each test date t, build panel ending at t, predict t+1 return
    ics = []
    for t in idx_range:
        if t - L < 0 or t + 1 >= returns.shape[1]:
            continue
        ok = np.where(valid[:, t - L:t + 1].all(axis=1))[0]
        if ok.size < N_pick:
            continue
        picks = ok[:N_pick]
        win = returns[picks, t - L:t + 1, :]  # uses t=history_end, t+1 as label would be out of win; rebuild
        # Correct: need history [t-L, t) then predict day t. But our convention
        # in training was window of length L+1 where first L rows are history
        # and row L is the label day. So here we also want window[:L] as
        # history ending at day t-1 inclusive, and label is returns[picks, t].
        history_win = returns[picks, t - L:t + 1, :]  # (N_pick, L+1, C), last row = day t
        f = extract_features(history_win, L)         # (N_pick, F)
        y_true = returns[picks, t, 0]                # (N_pick,) actual day-t ret
        pred = model.predict(f)                      # (N_pick,)
        # Spearman rank-IC
        mask = np.isfinite(pred) & np.isfinite(y_true)
        if mask.sum() < 5:
            continue
        p = pred[mask]; a = y_true[mask]
        pr = p.argsort().argsort().astype(float)
        ar = a.argsort().argsort().astype(float)
        pr -= pr.mean(); ar -= ar.mean()
        den = pr.std() * ar.std()
        if den > 0:
            ics.append(float((pr * ar).mean() / den))
    ics = np.array(ics)
    return {
        "ic_mean": float(ics.mean()) if len(ics) else float("nan"),
        "ic_std": float(ics.std()) if len(ics) else float("nan"),
        "ic_ir": float(ics.mean() / (ics.std() + 1e-9)) if len(ics) else float("nan"),
        "n_days": int(len(ics)),
    }


# ────────────────────────────────────────────────────────────────
# Main sweep
# ────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/csi800_2015_2024.npz")
    ap.add_argument("--synth", default="ckpts/M0_m6_csi800_step20000.samples.npz")
    ap.add_argument("--alphas", default="0,0.1,0.25,0.5,0.75,0.9")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--N-pick", type=int, default=32)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--n-train", type=int, default=100000,
                    help="target training set size (samples)")
    ap.add_argument("--train-end", default="2022-12-31")
    ap.add_argument("--test-start", default="2023-01-01")
    ap.add_argument("--test-end", default="2023-12-31")
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-rounds", type=int, default=200)
    ap.add_argument("--out", default="ckpts/alpha_sweep_lgbm.json")
    return ap.parse_args()


def main():
    args = parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"[lgbm-sweep] alphas={alphas}  seeds={seeds}  n_train={args.n_train}")

    real = load_real_returns(args.panel)
    print(f"[lgbm-sweep] real returns shape {real['returns'].shape}")

    synth_panels = np.load(args.synth, allow_pickle=True)["panels_denorm"].astype(np.float32)
    print(f"[lgbm-sweep] synth panels shape {synth_panels.shape}")

    results = []
    # Build real training-pool once per seed (shared across alphas)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        print(f"\n[lgbm-sweep] === seed {seed} — building data pools ===")
        t0 = time.time()
        Xr, yr = build_real_dataset(
            real, args.N_pick, args.L,
            (real["dates"][0], args.train_end), args.n_train, rng,
        )
        Xs, ys = build_synth_dataset(
            synth_panels, args.N_pick, args.L, args.n_train, rng,
        )
        print(f"[lgbm-sweep]   pools: real {Xr.shape}  synth {Xs.shape}  "
              f"({time.time()-t0:.1f}s)")

        for alpha in alphas:
            t0 = time.time()
            X_tr, y_tr = mix_datasets(Xr, yr, Xs, ys, alpha, rng)
            train_set = lgb.Dataset(X_tr, label=y_tr,
                                    feature_name=FEATURE_NAMES)
            model = lgb.train(
                params={
                    "objective": "regression",
                    "learning_rate": args.learning_rate,
                    "num_leaves": args.num_leaves,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 5,
                    "verbose": -1,
                    "seed": seed,
                },
                train_set=train_set,
                num_boost_round=args.num_rounds,
            )
            ic = eval_rank_ic(model, real, args.N_pick, args.L,
                              (args.test_start, args.test_end))
            r = {
                "alpha": alpha, "seed": seed,
                "train_size": int(len(y_tr)),
                "train_time_s": round(time.time() - t0, 1),
                **ic,
            }
            results.append(r)
            print(f"[lgbm-sweep]   alpha={alpha:.2f}  IC={r['ic_mean']:+.5f} "
                  f"+- {r['ic_std']:.5f}  IR={r['ic_ir']:+.3f}  n_days={r['n_days']}")

    out = {"config": vars(args), "results": results}
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[lgbm-sweep] saved -> {args.out}")

    # Per-alpha summary
    print("\n" + "=" * 75)
    print(f"{'alpha':>6s}  {'IC mean':>10s} +/- {'std':>7s}   {'IC_IR':>7s}   {'paired Δ':>9s}")
    print("-" * 75)
    by_seed = {}
    for r in results:
        by_seed.setdefault(r["seed"], {})[r["alpha"]] = r["ic_mean"]
    for a in alphas:
        ics = np.array([by_seed[s][a] for s in sorted(by_seed)])
        paired = np.array([by_seed[s][a] - by_seed[s][0.0]
                           for s in sorted(by_seed)])
        print(f"{a:>6.2f}  {ics.mean():+10.5f} +- {ics.std():7.5f}   "
              f"{ics.mean()/(ics.std()+1e-9):+7.3f}   {paired.mean():+.5f}")


if __name__ == "__main__":
    main()
