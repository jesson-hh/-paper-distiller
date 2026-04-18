"""
M0 InterDiff baseline trainer (unconditional).

Memory safeguards
-----------------
* `--max-mem-gb`  early-exit if peak CUDA memory exceeds this hard cap.
* `--reserve-gb`  refuses to start if free GPU memory is below this.
* `--mem-check-every`  re-measure peak mem and bail if it crept up.
* Pre-flight: alloc model + one batch + one fwd/bwd, report peak,
  refuse to launch if peak > 0.7 * free.
"""
from __future__ import annotations
import argparse
import gc
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from panel_windows import PanelWindowDataset
from model import InterDenoiser, count_params
from diffusion import GaussianDiffusion
from edm_diffusion import StudentTEDM

try:
    import psutil
    _PROC = psutil.Process()
except ImportError:
    psutil = None
    _PROC = None


def _free_gb() -> float:
    return torch.cuda.mem_get_info(0)[0] / 1e9


def _total_gb() -> float:
    return torch.cuda.mem_get_info(0)[1] / 1e9


def _rss_gb() -> float:
    if _PROC is None:
        return 0.0
    return _PROC.memory_info().rss / 1e9


def _sys_avail_gb() -> float:
    if psutil is None:
        return float("inf")
    return psutil.virtual_memory().available / 1e9


def _check_cpu_ram(max_rss_gb: float, reserve_ram_gb: float, where: str):
    if psutil is None:
        return
    rss = _rss_gb()
    avail = _sys_avail_gb()
    if rss > max_rss_gb:
        raise RuntimeError(
            f"{where}: process RSS {rss:.2f}GB > cap {max_rss_gb}GB — aborting"
        )
    if avail < reserve_ram_gb:
        raise RuntimeError(
            f"{where}: system available RAM {avail:.2f}GB < reserve "
            f"{reserve_ram_gb}GB — aborting to protect host"
        )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/csi300_2015_2024.npz")
    ap.add_argument("--length", type=int, default=32)
    ap.add_argument("--k", type=int, default=16, help="stocks per panel")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--T", type=int, default=200, help="diffusion steps")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-blocks", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--tag", default="cpu_smoke")
    ap.add_argument("--train-end", default="2022-12-31")
    ap.add_argument("--max-mem-gb", type=float, default=20.0,
                    help="hard cap on peak CUDA memory; abort if exceeded")
    ap.add_argument("--reserve-gb", type=float, default=4.0,
                    help="refuse to start if free GPU mem < this")
    ap.add_argument("--max-cpu-gb", type=float, default=8.0,
                    help="hard cap on process RSS (CPU RAM); abort if exceeded")
    ap.add_argument("--reserve-ram-gb", type=float, default=16.0,
                    help="abort if system available RAM drops below this")
    ap.add_argument("--mem-check-every", type=int, default=100)
    ap.add_argument("--cudnn-benchmark", action="store_true", default=True)
    ap.add_argument("--regime-window", type=int, default=0,
                    help="rolling vol window (days) for regime labels; 0=off")
    ap.add_argument("--n-regimes", type=int, default=0,
                    help="number of volatility regimes; 0=off")
    ap.add_argument("--aux-market-weight", type=float, default=0.0,
                    help="weight on market-factor MSE (common mode) in training loss")
    ap.add_argument("--sectors-npz", default=None,
                    help="path to sector labels sidecar npz (e.g. data/csi300_sectors.npz); "
                         "enables per-stock sector factor conditioning")
    ap.add_argument("--bf16", action="store_true",
                    help="train with bf16 mixed precision (forward/backward in bf16, "
                         "optimizer state in fp32); requires Ampere+ GPU")
    ap.add_argument("--cfg-drop", type=float, default=0.0,
                    help="probability of dropping all conditioning per batch during training, "
                         "enabling classifier-free guidance at inference; 0.0 = disabled, "
                         "0.1 is the standard CFG drop rate")
    ap.add_argument("--sign-cond", action="store_true",
                    help="add sign-aware (asymmetric) conditioning branches: negative-clipped "
                         "mkt and sector factors get their own projections. Breaks the linear "
                         "symmetry of additive conditioning so the model can learn leverage "
                         "effect (down-moves -> elevated volatility).")
    ap.add_argument("--edm", action="store_true",
                    help="use Student-t EDM diffusion (Karras 2022 + Pandey 2024 "
                         "heavy-tailed prior) instead of DDPM Gaussian. Better tail "
                         "coverage for kurtosis-heavy financial data.")
    ap.add_argument("--edm-nu", type=float, default=6.0,
                    help="Student-t degrees of freedom (nu). Smaller = heavier tails. "
                         "nu=6 -> kurt 3.0; nu=5 -> 6.0; nu=inf -> Gaussian.")
    ap.add_argument("--edm-sigma-min", type=float, default=0.002)
    ap.add_argument("--edm-sigma-max", type=float, default=80.0)
    ap.add_argument("--edm-sigma-data", type=float, default=1.0,
                    help="preconditioning scale; 1.0 for per-stock z-scored data")
    ap.add_argument("--edm-P-mean", type=float, default=-1.2)
    ap.add_argument("--edm-P-std", type=float, default=1.2)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device = {device}")
    if psutil is not None:
        print(f"[train] cpu ram: rss={_rss_gb():.2f}GB "
              f"avail={_sys_avail_gb():.2f}GB "
              f"(cap={args.max_cpu_gb} reserve={args.reserve_ram_gb})")
        if _sys_avail_gb() < args.reserve_ram_gb:
            raise RuntimeError(
                f"system available RAM {_sys_avail_gb():.2f}GB "
                f"< reserve {args.reserve_ram_gb}GB; refusing to start"
            )
    else:
        print("[train] psutil unavailable — CPU RAM safeguards disabled")
    if device == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        free, total = _free_gb(), _total_gb()
        print(f"[train] gpu total={total:.2f}GB free={free:.2f}GB "
              f"(reserve={args.reserve_gb} cap={args.max_mem_gb})")
        if free < args.reserve_gb:
            raise RuntimeError(
                f"free GPU memory {free:.2f}GB < reserve {args.reserve_gb}GB; refusing to start"
            )

    ds = PanelWindowDataset(
        panel_npz=args.panel,
        length=args.length,
        k_stocks=args.k,
        seed=args.seed,
        normalise=True,
        time_range=("2015-01-05", args.train_end),
        regime_window=args.regime_window,
        n_regimes=args.n_regimes,
        sectors_npz=args.sectors_npz,
    )
    print(f"[train] dataset info: {ds.info()}")
    use_regimes = ds.regime_spec is not None
    if use_regimes:
        from regimes import summary as regime_summary
        rs = regime_summary(ds.regime_labels, ds.regime_spec.n_regimes)
        print(f"[train] regimes: K={ds.regime_spec.n_regimes} "
              f"window={ds.regime_spec.window} entropy={rs['entropy']:.4f} "
              f"frac={np.round(rs['label_frac'],3).tolist()}")
    _check_cpu_ram(args.max_cpu_gb, args.reserve_ram_gb, "after dataset load")
    if psutil is not None:
        print(f"[train] cpu ram after ds: rss={_rss_gb():.2f}GB "
              f"avail={_sys_avail_gb():.2f}GB")

    loader = DataLoader(
        ds, batch_size=args.batch, num_workers=0, drop_last=True
    )

    model = InterDenoiser(
        n_channels=ds.n_channels,
        max_length=args.length,
        max_stocks=args.k,
        d_model=args.d_model,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_regimes=ds.regime_spec.n_regimes if use_regimes else 0,
        sign_cond=args.sign_cond,
    ).to(device)
    n_params = count_params(model)
    print(f"[train] model params = {n_params:,}")

    if args.edm:
        diff = StudentTEDM(
            sigma_min=args.edm_sigma_min,
            sigma_max=args.edm_sigma_max,
            sigma_data=args.edm_sigma_data,
            nu=args.edm_nu,
            P_mean=args.edm_P_mean,
            P_std=args.edm_P_std,
            device=device,
        )
        print(f"[train] diffusion = t-EDM  nu={args.edm_nu}  "
              f"sigma range [{args.edm_sigma_min}, {args.edm_sigma_max}]  "
              f"sigma_data={args.edm_sigma_data}")
    else:
        diff = GaussianDiffusion(T=args.T, device=device)
        print(f"[train] diffusion = DDPM  T={args.T}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    def _split_batch(b):
        """
        Unpack a batch into a dict with keys: x, regime, mkt, sector.
        Each value is a tensor on `device` or None.

        The PanelWindowDataset yields tuples whose order is always:
          (window, [regime_labels], [mkt], [sector])
        where regime_labels has int64 dtype and mkt/sector are float32.
        We split by dtype so the result is unambiguous for any combo.
        """
        out = {"x": None, "regime": None, "mkt": None, "sector": None}
        if not isinstance(b, (list, tuple)):
            out["x"] = b.to(device)
            return out
        out["x"] = b[0].to(device)
        rest = [t.to(device) for t in b[1:]]
        regime_items = [t for t in rest if t.dtype == torch.long]
        float_items = [t for t in rest if t.dtype != torch.long]
        if regime_items:
            out["regime"] = regime_items[0]
        if len(float_items) >= 1:
            out["mkt"] = float_items[0]
        if len(float_items) >= 2:
            out["sector"] = float_items[1]
        return out

    if device == "cuda":
        try:
            it_pre = iter(loader)
            pb = _split_batch(next(it_pre))
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                loss_pre = diff.training_loss(
                    model, pb["x"],
                    cond=pb["regime"], mkt_cond=pb["mkt"], sector_cond=pb["sector"],
                    aux_market_weight=args.aux_market_weight,
                    cfg_drop=args.cfg_drop,
                )
            loss_pre.backward()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            preflight_peak = torch.cuda.max_memory_allocated() / 1e9
            free_now = _free_gb()
            print(f"[train] preflight: peak={preflight_peak:.2f}GB free={free_now:.2f}GB")
            if preflight_peak > args.max_mem_gb:
                raise RuntimeError(
                    f"preflight peak {preflight_peak:.2f}GB > cap {args.max_mem_gb}GB; "
                    f"reduce --batch / --k / --d-model / --n-blocks"
                )
            del pb, loss_pre, it_pre
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                f"OOM at preflight; reduce config. orig: {str(e)[:200]}"
            )

    ema_loss = None
    t0 = time.time()
    model.train()
    losses_log = []

    it = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        bd = _split_batch(batch)

        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(args.bf16 and device == "cuda")):
            loss = diff.training_loss(
                model, bd["x"],
                cond=bd["regime"], mkt_cond=bd["mkt"], sector_cond=bd["sector"],
                aux_market_weight=args.aux_market_weight,
                cfg_drop=args.cfg_drop,
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        lv = float(loss.item())
        ema_loss = lv if ema_loss is None else 0.95 * ema_loss + 0.05 * lv
        losses_log.append(lv)

        if step % args.log_every == 0 or step == 1:
            sps = step / max(time.time() - t0, 1e-6)
            mem_str = ""
            if device == "cuda":
                peak = torch.cuda.max_memory_allocated() / 1e9
                mem_str = f"  peak={peak:.2f}GB"
            print(
                f"[train] step {step:5d}/{args.steps}  "
                f"loss={lv:.4f}  ema={ema_loss:.4f}  "
                f"lr={opt.param_groups[0]['lr']:.2e}  "
                f"{sps:.1f} step/s{mem_str}"
            )

        if device == "cuda" and step % args.mem_check_every == 0:
            peak = torch.cuda.max_memory_allocated() / 1e9
            if peak > args.max_mem_gb:
                raise RuntimeError(
                    f"peak GPU memory {peak:.2f}GB > cap {args.max_mem_gb}GB at step {step}; "
                    f"aborting to protect host"
                )

        if step % args.mem_check_every == 0:
            _check_cpu_ram(args.max_cpu_gb, args.reserve_ram_gb,
                           f"step {step}")

        if step % args.ckpt_every == 0 or step == args.steps:
            ckpt_dir = Path("ckpts")
            ckpt_dir.mkdir(exist_ok=True)
            ckpt_path = ckpt_dir / f"M0_{args.tag}_step{step}.pt"
            save_obj = {
                "model": model.state_dict(),
                "args": vars(args),
                "step": step,
                "loss_ema": ema_loss,
                "stats": {"mean": ds.mean, "std": ds.std},
                "ds_info": ds.info(),
                "mkt_cond": True,
                "sector_cond": ds.sector_labels is not None,
                "cfg_drop": args.cfg_drop,
                "bf16": args.bf16,
                "sign_cond": args.sign_cond,
                "edm": args.edm,
                "edm_config": {
                    "nu": args.edm_nu,
                    "sigma_min": args.edm_sigma_min,
                    "sigma_max": args.edm_sigma_max,
                    "sigma_data": args.edm_sigma_data,
                    "P_mean": args.edm_P_mean,
                    "P_std": args.edm_P_std,
                } if args.edm else None,
            }
            if use_regimes:
                save_obj["regime_spec"] = ds.regime_spec.to_dict()
            torch.save(save_obj, ckpt_path)
            print(f"[train] ckpt -> {ckpt_path}")

    np.save(Path("ckpts") / f"M0_{args.tag}_losses.npy", np.array(losses_log))
    print("[train] done.")


if __name__ == "__main__":
    main()
