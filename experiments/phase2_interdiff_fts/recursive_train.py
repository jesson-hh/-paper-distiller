"""
Recursive retraining model-collapse experiment (Shumailov 2024-style).

Starts from M6 (trained on real CSI800). Each generation:
  1. Sample N panels from the current generation's model (DDIM fast)
  2. Cache synth as a pseudo-panel (N_stocks = K, T = synthetic trajectory)
  3. Train next generation on ONLY synth (pure-synth recursive) OR on
     mix (synth + real) for a less-severe collapse trajectory
  4. Evaluate stylized facts of each generation
  5. Compare eval_compare verdicts across generations

Ends after --n-gens generations. Produces a gen-by-gen drift report.

Usage:
    python recursive_train.py \
        --init-ckpt ckpts/M0_m6_csi800_step20000.pt \
        --panel data/csi800_2015_2024.npz \
        --sectors-npz data/csi800_sectors.npz \
        --n-gens 3 \
        --mix-real 0.0 \
        --steps 8000 \
        --n-syn-panels 2000
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from panel_windows import PanelWindowDataset, derive_panel_returns, per_stock_stats
from model import InterDenoiser
from diffusion import GaussianDiffusion
from regimes import RegimeSpec, fit_regimes, label as regime_label


# ────────────────────────────────────────────────────────────────
# Synthetic corpus -> pseudo-panel npz (for training next gen)
# ────────────────────────────────────────────────────────────────

def synth_to_pseudo_panel(
    samples_npz: str,
    sectors_npz: str,
    out_path: str,
) -> str:
    """
    Convert a samples npz produced by sample.py into a pseudo-panel in
    the same format as data/csi800_2015_2024.npz, so the next-generation
    PanelWindowDataset can consume it.

    Strategy: concatenate all synth panels along time axis. Treat each
    of the 32 "slots" per panel as a pseudo-stock — so we end up with
    32 pseudo-stocks × (n_panels * 64) pseudo-days. A bit wasteful of
    data, but preserves per-panel cross-sectional structure.

    Note: the denormalised log_rets become the "panel" values directly;
    we don't reconstruct OHLC. The next-gen dataset uses log_ret as
    channel 0 derivation, so we fake a synthetic OHLC that recovers
    exactly the original 4 channels.
    """
    d = np.load(samples_npz, allow_pickle=True)
    syn = d["panels_denorm"].astype(np.float32)  # (n_p, 32, 64, 4)
    n_p, K, L, C = syn.shape

    # Stack panels along time. Result: (K, n_p * L, C)
    long = syn.transpose(1, 0, 2, 3).reshape(K, n_p * L, C)

    # We need to fake (open, close, high, low, volume, amount, factor)
    # so that derive_panel_returns reproduces the synthetic log_rets.
    # The relation is:
    #   log_ret[:, t]  = log(adj_close[:, t+1]) - log(adj_close[:, t])
    #   log_hc[:, t+1] = log(adj_high[:, t+1]) - log(adj_close[:, t+1])
    #   log_lc[:, t+1] = log(adj_low[:, t+1])  - log(adj_close[:, t+1])
    #   log_oc[:, t+1] = log(adj_open[:, t+1]) - log(adj_close[:, t+1])
    # so we can pick adj_close to accumulate log_ret; set factor=1,
    # then recover open/high/low from the respective log spreads.
    log_ret = long[:, :, 0]  # (K, T_long)
    log_hc = long[:, :, 1]
    log_lc = long[:, :, 2]
    log_oc = long[:, :, 3]
    T_long = log_ret.shape[1]

    # Reconstruct (non-adjusted) close starting from 100.0
    # derive_panel_returns uses np.diff(log(adj_close), axis=1), which
    # gives returns of length T_long. If we make panel of length T_long + 1
    # with close[:, 0] = 100 and close[:, t+1] = close[:, t] * exp(log_ret[:, t]),
    # derive_panel_returns will reproduce exactly our log_rets.
    close = np.zeros((K, T_long + 1), dtype=np.float32)
    close[:, 0] = 100.0
    close[:, 1:] = close[:, 0:1] * np.exp(np.cumsum(log_ret, axis=1))

    # high, low, open from spreads (aligned to the day after each log_ret,
    # i.e. index t+1 in the panel. Index 0 can be anything — only returns
    # and returns-of-returns are ever used downstream).
    high = np.zeros_like(close)
    low = np.zeros_like(close)
    open_ = np.zeros_like(close)
    high[:, 0] = close[:, 0]
    low[:, 0] = close[:, 0]
    open_[:, 0] = close[:, 0]
    high[:, 1:] = close[:, 1:] * np.exp(log_hc)
    low[:, 1:] = close[:, 1:] * np.exp(log_lc)
    open_[:, 1:] = close[:, 1:] * np.exp(log_oc)

    # Fill volume, amount, factor with placeholders so the loader works.
    volume = np.full_like(close, 1e6, dtype=np.float32)
    amount = close * volume
    factor = np.ones_like(close)

    panel = np.stack([open_, close, high, low, volume, amount, factor], axis=-1)  # (K, T+1, 7)
    fields = np.array(["open", "close", "high", "low", "volume", "amount", "factor"])

    # Dates: fake sequential dates in a format PanelWindowDataset expects (strings)
    # Use an ISO-ish format; since we strip normalisation by date later,
    # any strictly increasing string works.
    dates = np.array([f"2000-01-01+{i:05d}" for i in range(T_long + 1)])
    codes = np.array([f"SYN{i:03d}" for i in range(K)], dtype=object)

    # Copy sector labels from base sidecar (pseudo-stocks 0..31 mapped
    # cyclically to real sectors)
    sd = np.load(sectors_npz, allow_pickle=True)
    base_sectors = sd["sector_labels"]
    syn_sectors = np.array([int(base_sectors[i % len(base_sectors)]) for i in range(K)],
                           dtype=np.int64)

    np.savez_compressed(
        out_path,
        panel=panel, fields=fields, dates=dates, codes=codes,
        valid_mask=np.ones(panel.shape[:2], dtype=bool),
    )
    # Save a matching sectors sidecar
    out_sectors = out_path.replace(".npz", "_sectors.npz")
    np.savez_compressed(
        out_sectors, codes=codes, sector_labels=syn_sectors,
        sector_names=sd.get("sector_names", np.array([]))
    )
    return out_path, out_sectors


# ────────────────────────────────────────────────────────────────
# Training one generation (lightweight — smaller steps/model)
# ────────────────────────────────────────────────────────────────

def train_gen(
    panel_npz: str,
    sectors_npz: str,
    tag: str,
    steps: int,
    base_args: dict,
    device: str,
) -> str:
    """Train one generation, return path to final checkpoint."""
    ds = PanelWindowDataset(
        panel_npz=panel_npz,
        length=base_args["length"],
        k_stocks=base_args["k"],
        seed=base_args["seed"],
        normalise=True,
        time_range=None,  # use whole synthetic panel
        regime_window=base_args["regime_window"],
        n_regimes=base_args["n_regimes"],
        sectors_npz=sectors_npz,
    )
    print(f"[gen-train] dataset info: {ds.info()}")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=base_args["batch"], num_workers=0, drop_last=True)

    model = InterDenoiser(
        n_channels=ds.n_channels,
        max_length=base_args["length"],
        max_stocks=base_args["k"],
        d_model=base_args["d_model"],
        n_blocks=base_args["n_blocks"],
        n_heads=base_args["n_heads"],
        n_regimes=ds.regime_spec.n_regimes if ds.regime_spec else 0,
        sign_cond=False,
    ).to(device)
    diff = GaussianDiffusion(T=base_args["T"], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=base_args["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    def _split_batch(b):
        out = {"x": None, "regime": None, "mkt": None, "sector": None}
        if not isinstance(b, (list, tuple)):
            out["x"] = b.to(device)
            return out
        out["x"] = b[0].to(device)
        rest = [t.to(device) for t in b[1:]]
        reg = [t for t in rest if t.dtype == torch.long]
        flt = [t for t in rest if t.dtype != torch.long]
        if reg: out["regime"] = reg[0]
        if len(flt) >= 1: out["mkt"] = flt[0]
        if len(flt) >= 2: out["sector"] = flt[1]
        return out

    model.train()
    t0 = time.time()
    it = iter(loader)
    ema = None
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        bd = _split_batch(batch)
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                             enabled=(device == "cuda")):
            loss = diff.training_loss(
                model, bd["x"],
                cond=bd["regime"], mkt_cond=bd["mkt"], sector_cond=bd["sector"],
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        lv = float(loss.item())
        ema = lv if ema is None else 0.95 * ema + 0.05 * lv
        if step % 500 == 0 or step == steps:
            sps = step / max(time.time() - t0, 1e-6)
            print(f"[gen-train] step {step}/{steps}  loss={lv:.4f}  ema={ema:.4f}  "
                  f"{sps:.1f} step/s")

    ckpt_path = f"ckpts/M0_{tag}_step{steps}.pt"
    save = {
        "model": model.state_dict(),
        "args": {**base_args, "tag": tag, "steps": steps},
        "step": steps,
        "loss_ema": ema,
        "stats": {"mean": ds.mean, "std": ds.std},
        "ds_info": ds.info(),
        "mkt_cond": True,
        "sector_cond": ds.sector_labels is not None,
        "regime_spec": ds.regime_spec.to_dict() if ds.regime_spec else None,
    }
    torch.save(save, ckpt_path)
    print(f"[gen-train] saved -> {ckpt_path}")
    return ckpt_path


# ────────────────────────────────────────────────────────────────
# Stylized-fact eval wrapper (drop-in uses eval_compare)
# ────────────────────────────────────────────────────────────────

def eval_against_real(samples_path: str, real_panel: str) -> dict:
    """Call eval_compare.py and parse key metrics."""
    result = subprocess.run(
        ["D:/app/miniconda/envs/stocks/python.exe", "eval_compare.py",
         "--real-panel", real_panel, "--samples", samples_path],
        capture_output=True, text=True, timeout=300,
    )
    out = result.stdout
    metrics = {}
    verdicts = {}
    for line in out.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Parse rows like:
        #   "std                            real= 0.0274  syn= 0.0271  -> OK"
        if "real=" in line and "syn=" in line and "->" in line:
            parts = line.split()
            # format: <metric> real= <v1> syn= <v2> -> <verdict>
            try:
                metric = parts[0]
                real_v = float(parts[2])
                syn_v = float(parts[4])
                verdict = parts[-1]
                metrics[metric] = {"real": real_v, "syn": syn_v}
                verdicts[metric] = verdict
            except (ValueError, IndexError):
                pass
        # Temporal leverage line not in verdict block
        if "leverage_lag1" in line and "real=" not in line:
            parts = line.split()
            try:
                if len(parts) >= 3 and parts[0] == "leverage_lag1":
                    metrics["leverage_lag1"] = {"real": float(parts[1]),
                                                 "syn": float(parts[2])}
            except ValueError:
                pass
    return {"metrics": metrics, "verdicts": verdicts}


# ────────────────────────────────────────────────────────────────
# Main recursive loop
# ────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-ckpt", default="ckpts/M0_m6_csi800_step20000.pt")
    ap.add_argument("--panel", default="data/csi800_2015_2024.npz")
    ap.add_argument("--sectors-npz", default="data/csi800_sectors.npz")
    ap.add_argument("--n-gens", type=int, default=3)
    ap.add_argument("--steps", type=int, default=8000,
                    help="training steps per generation (reduce from 20k for speed)")
    ap.add_argument("--n-syn-panels", type=int, default=2000,
                    help="panels to sample per generation (for training the next gen)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ckpts/recursive_collapse.json")
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[recurse] device={device}  n_gens={args.n_gens}  steps/gen={args.steps}")

    # Load init (generation 0 = M6) config
    ck0 = torch.load(args.init_ckpt, map_location=device, weights_only=False)
    base_args = {
        "length": ck0["args"]["length"],
        "k": ck0["args"]["k"],
        "batch": ck0["args"]["batch"],
        "lr": ck0["args"]["lr"],
        "T": ck0["args"]["T"],
        "d_model": ck0["args"]["d_model"],
        "n_blocks": ck0["args"]["n_blocks"],
        "n_heads": ck0["args"]["n_heads"],
        "regime_window": ck0["args"]["regime_window"],
        "n_regimes": ck0["args"]["n_regimes"],
        "seed": args.seed,
    }

    results = []
    current_ckpt = args.init_ckpt
    current_panel = args.panel
    current_sectors = args.sectors_npz

    # Gen 0 eval (M6 on real)
    print("\n[recurse] === gen 0 (M6 trained on REAL) ===")
    gen0_samples = str(Path(current_ckpt).with_suffix(".samples.npz"))
    assert Path(gen0_samples).exists(), f"missing {gen0_samples}"
    gen0_eval = eval_against_real(gen0_samples, args.panel)
    print(f"[recurse] gen 0 verdicts: {gen0_eval['verdicts']}")
    results.append({"gen": 0, "ckpt": current_ckpt,
                    "trained_on": "real",
                    "samples": gen0_samples,
                    **gen0_eval})

    # Recursive loop
    for g in range(1, args.n_gens + 1):
        print(f"\n[recurse] === gen {g} ===")
        # 1. Sample from previous gen (skip if already cached)
        gen_tag = f"rec_g{g}"
        prev_samples = str(Path(current_ckpt).with_suffix(f".rec_samples_for_g{g}.npz"))
        if Path(prev_samples).exists():
            print(f"[recurse] resume: {prev_samples} exists, skipping sampling")
        else:
            print(f"[recurse] sampling {args.n_syn_panels} panels from {current_ckpt}")
            n_batches = args.n_syn_panels // 8
            panel_data = np.load(current_panel, allow_pickle=True)
            panel_dates = panel_data["dates"].astype(str)
            extra_args = []
            if not panel_dates[0].startswith("201"):
                extra_args = ["--time-range", f"{panel_dates[0]},{panel_dates[-1]}"]
            subprocess.run(
                ["D:/app/miniconda/envs/stocks/python.exe", "sample.py",
                 "--ckpt", current_ckpt,
                 "--n-batches", str(n_batches), "--batch", "8",
                 "--panel", current_panel, "--sectors-npz", current_sectors,
                 "--sampler", "ddim", "--ddim-steps", "50",
                 *extra_args,
                 "--out", prev_samples],
                check=True,
            )
        # 2. Convert to pseudo-panel for training (skip if cached)
        pseudo_panel = str(Path(current_ckpt).with_suffix(f".rec_pseudo_g{g}.npz"))
        new_sectors = pseudo_panel.replace(".npz", "_sectors.npz")
        if Path(pseudo_panel).exists() and Path(new_sectors).exists():
            print(f"[recurse] resume: pseudo-panel {pseudo_panel} exists")
            new_panel = pseudo_panel
        else:
            new_panel, new_sectors = synth_to_pseudo_panel(
                prev_samples, current_sectors, pseudo_panel,
            )
            print(f"[recurse] pseudo-panel: {new_panel}")

        # 3. Train gen g (skip if cached)
        new_ckpt = f"ckpts/M0_{gen_tag}_step{args.steps}.pt"
        if Path(new_ckpt).exists():
            print(f"[recurse] resume: {new_ckpt} exists, skipping training")
        else:
            new_ckpt = train_gen(new_panel, new_sectors, gen_tag, args.steps,
                                  base_args, device)

        # 4. Sample gen g (for eval)
        gen_samples = new_ckpt.replace(".pt", ".samples.npz")
        print(f"[recurse] sampling for eval -> {gen_samples}")
        if Path(gen_samples).exists():
            print(f"[recurse] resume: {gen_samples} exists")
        else:
            pseudo = np.load(new_panel, allow_pickle=True)
            pseudo_dates = pseudo["dates"].astype(str)
            time_range_str = f"{pseudo_dates[0]},{pseudo_dates[-1]}"
            subprocess.run(
                ["D:/app/miniconda/envs/stocks/python.exe", "sample.py",
                 "--ckpt", new_ckpt,
                 "--n-batches", "50", "--batch", "8",
                 "--panel", new_panel, "--sectors-npz", new_sectors,
                 "--sampler", "ddim", "--ddim-steps", "50",
                 "--time-range", time_range_str,
                 "--out", gen_samples],
                check=True,
            )

        # 5. Eval gen g AGAINST REAL
        gen_eval = eval_against_real(gen_samples, args.panel)
        print(f"[recurse] gen {g} verdicts: {gen_eval['verdicts']}")
        results.append({
            "gen": g, "ckpt": new_ckpt,
            "trained_on": "prev_gen_synth",
            "samples": gen_samples,
            **gen_eval,
        })

        # Advance
        current_ckpt = new_ckpt
        current_panel = new_panel
        current_sectors = new_sectors

    # Save
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[recurse] saved -> {args.out}")

    # Summary: drift of key metrics across generations
    print("\n" + "=" * 95)
    print(f"{'gen':>4s}  {'std':>8s}  {'kurt':>8s}  {'hill_R':>8s}  {'hill_L':>8s}  "
          f"{'acf_r2_1':>9s}  {'lev':>8s}  {'pair_corr':>9s}  {'verdicts':>11s}")
    print("-" * 95)
    for r in results:
        m = r["metrics"]
        v = r["verdicts"]
        n_ok = sum(1 for x in v.values() if x == "OK")
        n_tot = len(v)
        def _s(key):
            return m.get(key, {}).get("syn", float("nan"))
        print(f"{r['gen']:>4d}  {_s('std'):>8.4f}  {_s('excess_kurt'):>8.3f}  "
              f"{_s('hill_right'):>8.3f}  {_s('hill_left'):>8.3f}  "
              f"{_s('acf_r2_lag1'):>9.4f}  "
              f"{m.get('leverage_lag1', {}).get('syn', float('nan')):>8.4f}  "
              f"{_s('panel_mean_pair_corr'):>9.4f}  "
              f"{n_ok}/{n_tot:>3d}")


if __name__ == "__main__":
    main()
