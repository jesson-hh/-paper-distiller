"""
Minimal DDPM utilities: beta schedules, noising, training loss,
and an ancestral sampler.

Defaults to the original Ho et al. linear schedule, which is stable for
T as small as 500 — important for CPU experiments. Cosine kept around
for reference but disabled because at small T it produces β values that
get clamped near 1, blowing up the sampler.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def linear_beta_schedule(
    T: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(min=1e-5, max=0.999)


class GaussianDiffusion:
    def __init__(self, T: int = 500, device: str = "cpu", schedule: str = "linear"):
        self.T = T
        if schedule == "linear":
            betas = linear_beta_schedule(T).to(device)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(T).to(device)
        else:
            raise ValueError(f"unknown schedule {schedule}")
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_cum = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum = torch.sqrt(self.alpha_cum)
        self.sqrt_one_minus_alpha_cum = torch.sqrt(1.0 - self.alpha_cum)
        self.device = device

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        sa = self.sqrt_alpha_cum[t].view(-1, *([1] * (x0.ndim - 1)))
        sb = self.sqrt_one_minus_alpha_cum[t].view(-1, *([1] * (x0.ndim - 1)))
        return sa * x0 + sb * noise

    def training_loss(
        self,
        model,
        x0: torch.Tensor,
        cond: torch.Tensor | None = None,
        mkt_cond: torch.Tensor | None = None,
        aux_market_weight: float = 0.0,
    ) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        eps = model(xt, t, cond=cond, mkt_cond=mkt_cond)
        base = F.mse_loss(eps, noise)
        if aux_market_weight <= 0.0:
            return base
        mkt_true = noise.mean(dim=1, keepdim=True)
        mkt_pred = eps.mean(dim=1, keepdim=True)
        res_true = noise - mkt_true
        res_pred = eps - mkt_pred
        loss_mkt = F.mse_loss(mkt_pred, mkt_true)
        loss_res = F.mse_loss(res_pred, res_true)
        return loss_res + aux_market_weight * loss_mkt

    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        progress: bool = False,
        clip_denoised: bool = True,
        clip_range: float = 5.0,
        cond: torch.Tensor | None = None,
        mkt_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=self.device)
        iters = range(self.T - 1, -1, -1)
        if progress:
            try:
                from tqdm import tqdm
                iters = tqdm(iters, desc="sample")
            except ImportError:
                pass
        for i in iters:
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            eps = model(x, t, cond=cond, mkt_cond=mkt_cond)
            beta = self.betas[i]
            alpha = self.alphas[i]
            ac = self.alpha_cum[i]
            sa_cum = self.sqrt_alpha_cum[i]
            sb_cum = self.sqrt_one_minus_alpha_cum[i]

            x0_pred = (x - sb_cum * eps) / sa_cum
            if clip_denoised:
                x0_pred = x0_pred.clamp(-clip_range, clip_range)

            if i > 0:
                ac_prev = self.alpha_cum[i - 1]
                coef_x0 = (torch.sqrt(ac_prev) * beta) / (1 - ac)
                coef_xt = (torch.sqrt(alpha) * (1 - ac_prev)) / (1 - ac)
                mean = coef_x0 * x0_pred + coef_xt * x
                var = beta * (1 - ac_prev) / (1 - ac)
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var) * noise
            else:
                x = x0_pred
        return x
