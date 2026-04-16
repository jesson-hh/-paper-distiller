"""
Sample synthetic panel windows from a trained M0 InterDiff baseline.

The output npz mirrors the per-stock window file format so the existing
stylized_facts.py can consume it directly.

    python sample.py --ckpt ckpts/M0_main_step8000.pt --n-batches 50 --batch 8

This produces ~ batch * n_batches * k panels = e.g. 6400 stock-trajectories.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

from model import InterDenoiser
from diffusion import GaussianDiffusion
from panel_windows import PanelWindowDataset
from regimes import RegimeSpec


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-batches", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--panel", default="data/csi300_2015_2024.npz",
                    help="source panel for borrowing regime label sequences (M1)")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["args"]
    print(f"[sample] ckpt step={ck.get('step')} loss_ema={ck.get('loss_ema'):.4f}")

    regime_spec = None
    if "regime_spec" in ck:
        regime_spec = RegimeSpec.from_dict(ck["regime_spec"])
        print(f"[sample] regime: K={regime_spec.n_regimes} window={regime_spec.window}")

    use_mkt_cond = ck.get("mkt_cond", False)
    print(f"[sample] mkt_cond={use_mkt_cond}")

    model = InterDenoiser(
        n_channels=4,
        max_length=cfg["length"],
        max_stocks=cfg["k"],
        d_model=cfg["d_model"],
        n_blocks=cfg["n_blocks"],
        n_heads=cfg["n_heads"],
        n_regimes=regime_spec.n_regimes if regime_spec is not None else 0,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    diff = GaussianDiffusion(T=cfg["T"], device=device)

    L = cfg["length"]
    K = cfg["k"]
    C = 4

    # --- prepare conditioning sources ---
    ds_source = None
    if regime_spec is not None or use_mkt_cond:
        ds_source = PanelWindowDataset(
            panel_npz=args.panel,
            length=L,
            k_stocks=K,
            seed=args.seed,
            normalise=True,
            time_range=("2015-01-05", cfg.get("train_end", "2022-12-31")),
            regime_window=regime_spec.window if regime_spec else 0,
            n_regimes=regime_spec.n_regimes if regime_spec else 0,
        )
        source_iter = iter(ds_source)

    all_samples = []
    for bi in range(args.n_batches):
        cond_batch = None
        mkt_batch = None

        if ds_source is not None:
            cond_list = []
            mkt_list = []
            for _ in range(args.batch):
                item = next(source_iter)
                if isinstance(item, (list, tuple)):
                    if len(item) == 3:
                        _, lab, mkt = item
                        cond_list.append(lab)
                        mkt_list.append(mkt)
                    elif len(item) == 2:
                        if item[1].dtype == torch.long:
                            cond_list.append(item[1])
                        else:
                            mkt_list.append(item[1])
                else:
                    pass
            if cond_list:
                cond_batch = torch.stack(cond_list, dim=0).to(device)
            if mkt_list and use_mkt_cond:
                mkt_batch = torch.stack(mkt_list, dim=0).to(device)

        x = diff.sample(model, shape=(args.batch, K, L, C),
                        cond=cond_batch, mkt_cond=mkt_batch)
        all_samples.append(x.cpu().numpy())
        print(f"[sample] batch {bi+1}/{args.n_batches}")

    samples = np.concatenate(all_samples, axis=0)
    print(f"[sample] panel batch tensor: {samples.shape}")

    flat = samples.reshape(-1, L, C)
    print(f"[sample] flattened (stock-trajectory, L, C): {flat.shape}")

    mean = ck["stats"]["mean"]
    std = ck["stats"]["std"]
    avg_mean = mean.mean(axis=0)
    avg_std = std.mean(axis=0)
    flat_denorm = flat * avg_std[None, None, :] + avg_mean[None, None, :]
    print("[sample] de-normalised using avg per-stock stats")

    panels_denorm = samples * avg_std[None, None, None, :] + avg_mean[None, None, None, :]

    out = Path(args.out) if args.out else Path(args.ckpt).with_suffix(".samples.npz")
    np.savez_compressed(
        out,
        windows=flat,
        windows_denorm=flat_denorm,
        panels=samples,
        panels_denorm=panels_denorm,
        ret_fields=np.array(["log_ret", "log_hc", "log_lc", "log_oc"]),
        ckpt=str(args.ckpt),
    )
    print(f"[sample] saved -> {out.resolve()}")
    print(f"[sample] windows stats:  mean={flat_denorm[..., 0].mean():.5f}  "
          f"std={flat_denorm[..., 0].std():.5f}  "
          f"min={flat_denorm[..., 0].min():.4f}  "
          f"max={flat_denorm[..., 0].max():.4f}")


if __name__ == "__main__":
    main()
