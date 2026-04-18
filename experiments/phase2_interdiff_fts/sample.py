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
from edm_diffusion import StudentTEDM
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
                    help="source panel for borrowing regime/mkt/sector sequences")
    ap.add_argument("--sectors-npz", default="data/csi300_sectors.npz",
                    help="sector labels sidecar; used if ckpt has sector_cond=True")
    ap.add_argument("--sampler", choices=["ddpm", "ddim", "edm"], default="ddpm",
                    help="ddpm = full T-step ancestral sampling; "
                         "ddim = deterministic fast DDPM sampler; "
                         "edm = Karras 2022 Heun 2nd-order (used for t-EDM ckpts).")
    ap.add_argument("--ddim-steps", type=int, default=50,
                    help="number of DDIM steps when --sampler=ddim")
    ap.add_argument("--edm-steps", type=int, default=18,
                    help="number of Heun steps when --sampler=edm (t-EDM)")
    ap.add_argument("--ddim-eta", type=float, default=0.0,
                    help="DDIM stochasticity; 0 = deterministic, 1 = DDPM-like")
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="classifier-free guidance scale; 1.0 = no guidance, "
                         ">1 amplifies conditioning (typical 1.5-7.5)")
    ap.add_argument("--time-range", default=None,
                    help="override date window for conditioning source, "
                         "format 'YYYY-MM-DD,YYYY-MM-DD'; useful for "
                         "pseudo-panels with non-ISO dates")
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
    use_sector_cond = ck.get("sector_cond", False)
    cfg_drop_train = ck.get("cfg_drop", 0.0)
    print(f"[sample] mkt_cond={use_mkt_cond}  sector_cond={use_sector_cond}  "
          f"cfg_drop(train)={cfg_drop_train}")
    print(f"[sample] sampler={args.sampler}  "
          f"{'steps=' + str(args.ddim_steps) + ' eta=' + str(args.ddim_eta) if args.sampler == 'ddim' else 'T=' + str(ck['args']['T'])}  "
          f"guidance={args.guidance}")
    if args.guidance != 1.0 and cfg_drop_train == 0.0:
        print("[sample] WARNING: guidance != 1 but ckpt was not trained with cfg_drop>0; "
              "unconditional branch is not well-learned, results may be unstable.")

    model = InterDenoiser(
        n_channels=4,
        max_length=cfg["length"],
        max_stocks=cfg["k"],
        d_model=cfg["d_model"],
        n_blocks=cfg["n_blocks"],
        n_heads=cfg["n_heads"],
        n_regimes=regime_spec.n_regimes if regime_spec is not None else 0,
        sign_cond=ck.get("sign_cond", False),
        lev_cond=ck.get("lev_cond", False),
        lev_window=ck.get("lev_window", 5),
        lev_mode=ck.get("lev_mode", "both"),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    # Auto-detect EDM ckpts
    is_edm = ck.get("edm", False)
    if is_edm:
        edm_cfg = ck.get("edm_config", {})
        diff = StudentTEDM(
            sigma_min=edm_cfg.get("sigma_min", 0.002),
            sigma_max=edm_cfg.get("sigma_max", 80.0),
            sigma_data=edm_cfg.get("sigma_data", 1.0),
            nu=edm_cfg.get("nu", 6.0),
            P_mean=edm_cfg.get("P_mean", -1.2),
            P_std=edm_cfg.get("P_std", 1.2),
            device=device,
        )
        # If user didn't explicitly pick a sampler, default to edm for edm ckpts
        if args.sampler == "ddpm":
            args.sampler = "edm"
        print(f"[sample] diffusion = t-EDM  nu={edm_cfg.get('nu', 6.0)}")
    else:
        diff = GaussianDiffusion(T=cfg["T"], device=device)

    L = cfg["length"]
    K = cfg["k"]
    C = 4

    # --- prepare conditioning sources ---
    def _unpack_item(item):
        """Same convention as train._split_batch: split by dtype."""
        out = {"regime": None, "mkt": None, "sector": None}
        if not isinstance(item, (list, tuple)):
            return out
        rest = list(item[1:])
        for t in rest:
            if t.dtype == torch.long and out["regime"] is None:
                out["regime"] = t
            elif t.dtype != torch.long:
                if out["mkt"] is None:
                    out["mkt"] = t
                elif out["sector"] is None:
                    out["sector"] = t
        return out

    ds_source = None
    if regime_spec is not None or use_mkt_cond or use_sector_cond:
        if args.time_range:
            lo, hi = args.time_range.split(",")
            tr = (lo.strip(), hi.strip())
        else:
            tr = ("2015-01-05", cfg.get("train_end", "2022-12-31"))
        ds_source = PanelWindowDataset(
            panel_npz=args.panel,
            length=L,
            k_stocks=K,
            seed=args.seed,
            normalise=True,
            time_range=tr,
            regime_window=regime_spec.window if regime_spec else 0,
            n_regimes=regime_spec.n_regimes if regime_spec else 0,
            sectors_npz=args.sectors_npz if use_sector_cond else None,
        )
        source_iter = iter(ds_source)

    all_samples = []
    for bi in range(args.n_batches):
        cond_batch = None
        mkt_batch = None
        sector_batch = None

        if ds_source is not None:
            cond_list, mkt_list, sec_list = [], [], []
            for _ in range(args.batch):
                parts = _unpack_item(next(source_iter))
                if parts["regime"] is not None:
                    cond_list.append(parts["regime"])
                if parts["mkt"] is not None:
                    mkt_list.append(parts["mkt"])
                if parts["sector"] is not None:
                    sec_list.append(parts["sector"])
            if cond_list:
                cond_batch = torch.stack(cond_list, dim=0).to(device)
            if mkt_list and use_mkt_cond:
                mkt_batch = torch.stack(mkt_list, dim=0).to(device)
            if sec_list and use_sector_cond:
                sector_batch = torch.stack(sec_list, dim=0).to(device)

        if args.sampler == "edm":
            x = diff.sample_heun(
                model, shape=(args.batch, K, L, C),
                steps=args.edm_steps,
                cond=cond_batch, mkt_cond=mkt_batch, sector_cond=sector_batch,
                guidance=args.guidance,
            )
        elif args.sampler == "ddim":
            x = diff.sample_ddim(
                model, shape=(args.batch, K, L, C),
                steps=args.ddim_steps, eta=args.ddim_eta,
                cond=cond_batch, mkt_cond=mkt_batch, sector_cond=sector_batch,
                guidance=args.guidance,
            )
        else:
            x = diff.sample(
                model, shape=(args.batch, K, L, C),
                cond=cond_batch, mkt_cond=mkt_batch, sector_cond=sector_batch,
                guidance=args.guidance,
            )
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
