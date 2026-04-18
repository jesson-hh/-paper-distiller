"""
Student-t EDM (t-EDM) diffusion.

Based on:
- Karras et al. 2022 ("Elucidating the Design Space of Diffusion-Based
  Generative Models", NeurIPS) -- standard EDM preconditioning + sigma
  schedule + Heun 2nd-order sampler.
- Pandey et al. 2024 ("Heavy-Tailed Diffusion Models", arxiv 2410.14171)
  -- Student-t noise prior for better heavy-tail coverage, nu controls
  tail thickness (nu -> infty recovers Gaussian EDM).

Motivation for financial data: real CSI800 excess_kurt ~ 3.55, but our
Gaussian DDPM (M6) achieves only ~3.0. Student-t noise with nu ~ 5-7
should better match the fat tails. Student-t kurtosis = 6 / (nu - 4)
for nu > 4: nu=5 -> kurt=6, nu=6 -> kurt=3, nu=8 -> kurt=1.5.

Preconditioning uses standard EDM formulas (the paper notes coefficients
"additionally depend on nu" but keeps the Gaussian-EDM functional form
as a good approximation for moderate nu).

Interface mirrors GaussianDiffusion:
    edm = StudentTEDM(nu=6, sigma_min=0.002, sigma_max=80, device="cuda")
    loss = edm.training_loss(model, x0, cond=..., mkt_cond=..., sector_cond=...)
    samples = edm.sample_heun(model, shape=(B, N, L, C), steps=18, ...)
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


class StudentTEDM:
    """
    t-EDM diffusion model. All public methods share API with
    GaussianDiffusion so the train/sample loops need minimal change.

    Args:
        sigma_min: smallest noise scale in the rho-schedule
        sigma_max: largest noise scale
        rho: rho-schedule exponent (Karras default = 7)
        sigma_data: scale for preconditioning. For our z-scored returns,
                    0.5 is the standard default; we use 1.0 since inputs
                    are per-stock z-scored (std=1).
        nu: Student-t degrees of freedom. nu = infty -> Gaussian EDM.
        P_mean, P_std: log-normal params for training sigma sampling
                       (Karras defaults: -1.2, 1.2).
        device: torch device
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 1.0,
        nu: float = 6.0,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        device: str = "cpu",
        aux_lev_weight: float = 0.0,
        aux_lev_target: float = 0.0,
        aux_lev_low_sigma_only: bool = True,
    ):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.sigma_data = float(sigma_data)
        self.nu = float(nu)
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)
        self.device = device
        self.aux_lev_weight = float(aux_lev_weight)
        self.aux_lev_target = float(aux_lev_target)
        self.aux_lev_low_sigma_only = bool(aux_lev_low_sigma_only)
        # For convenience
        self._log_sigma_min = math.log(self.sigma_min)
        self._log_sigma_max = math.log(self.sigma_max)

    # ---------- Student-t noise ----------

    def sample_student_t(self, shape, device=None):
        """
        Multivariate Student-t draws with dof nu, identity covariance.
        Method: y / sqrt(chi2/nu) where y ~ N(0, I), chi2 ~ Chi^2(nu).
        chi2 is scalar per sample (shared across the d dims of that
        sample) — this is the multivariate t_d(0, I, nu) definition.
        """
        device = device or self.device
        y = torch.randn(shape, device=device)
        B = shape[0]
        chi2 = torch.distributions.Chi2(
            torch.tensor(self.nu, device=device)
        ).sample((B,))
        # Broadcast chi2 (shape (B,)) against y (shape (B, ...))
        view = (B,) + (1,) * (y.ndim - 1)
        chi2 = chi2.view(view)
        return y / torch.sqrt(chi2 / self.nu)

    # ---------- EDM preconditioning ----------

    def c_skip(self, sigma):
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_in(self, sigma):
        return 1.0 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_noise(self, sigma):
        """
        Time-embedding input. Following EDM, use 0.25 * log(sigma).

        Our InterDenoiser's sinusoidal embedding expects a meaningful
        range; 0.25 * log(sigma) in [-1.55, 1.10] is too compact for
        the default frequencies. We scale by TIME_SCALE to spread it
        across the sinusoidal bands.
        """
        TIME_SCALE = 250.0  # makes range ~[-400, 275], similar to discrete DDPM t
        return TIME_SCALE * torch.log(sigma)

    # ---------- Training ----------

    def sample_train_sigma(self, B, device):
        """Log-normal sigma sampling as in Karras EDM."""
        rnd = torch.randn(B, device=device)
        return torch.exp(rnd * self.P_std + self.P_mean)

    def training_loss(
        self,
        model,
        x0: torch.Tensor,
        cond: torch.Tensor | None = None,
        mkt_cond: torch.Tensor | None = None,
        sector_cond: torch.Tensor | None = None,
        cfg_drop: float = 0.0,
        aux_market_weight: float = 0.0,  # kept for API compat, unused here
    ) -> torch.Tensor:
        B = x0.shape[0]
        device = x0.device

        # Classifier-free guidance: drop all conditioning with prob cfg_drop
        if cfg_drop > 0.0 and torch.rand((), device=device).item() < cfg_drop:
            cond = None
            mkt_cond = None
            sector_cond = None

        # Sample sigma per batch element
        sigma = self.sample_train_sigma(B, device)
        view = (B,) + (1,) * (x0.ndim - 1)
        sigma_v = sigma.view(view)

        # Student-t noise scaled by sigma
        noise = self.sample_student_t(x0.shape, device) * sigma_v
        xt = x0 + noise

        # EDM preconditioning
        c_in_v = self.c_in(sigma_v)
        c_out_v = self.c_out(sigma_v)
        c_skip_v = self.c_skip(sigma_v)
        c_noise_v = self.c_noise(sigma)  # (B,) for time embedding

        # Target for F_theta:  F = (x0 - c_skip * xt) / c_out
        target_F = (x0 - c_skip_v * xt) / c_out_v

        # Model outputs F_theta
        F_theta = model(
            c_in_v * xt, c_noise_v,
            cond=cond, mkt_cond=mkt_cond, sector_cond=sector_cond,
        )

        # EDM loss: weight = 1 / c_out^2, but since target_F = (x0 - c_skip*xt) / c_out
        # and D_theta - x0 = c_out * (F_theta - target_F), the weighted MSE
        # (1/c_out^2) * ||D_theta - x0||^2 equals ||F_theta - target_F||^2.
        main_loss = F.mse_loss(F_theta, target_F)

        # Optional auxiliary leverage loss: directly supervise the
        # corr(r_t, r_{t+1}^2) statistic on the predicted x0. Useful for
        # nudging A-share positive leverage (target ~ +0.013) since
        # the noise prediction MSE alone doesn't fix the sign.
        if self.aux_lev_weight > 0.0:
            x0_pred = c_skip_v * xt + c_out_v * F_theta  # (B, N, L, C)
            # Optionally restrict the aux loss to low-sigma examples in the
            # batch: at high sigma, x0_pred is mostly noise, so the leverage
            # estimate is meaningless. Mask with sigma < sigma_data.
            if self.aux_lev_low_sigma_only:
                mask = (sigma < self.sigma_data).view(B)
                if mask.any():
                    x0_pred_for_aux = x0_pred[mask]
                else:
                    x0_pred_for_aux = x0_pred  # fall back if no low-sigma in batch
            else:
                x0_pred_for_aux = x0_pred

            r = x0_pred_for_aux[..., 0]  # (B', N, L) log_ret
            # Pool everything: pair (r_t, r_{t+1}^2) across all (B', N) trajectories
            a = r[..., :-1].reshape(-1)
            b = (r[..., 1:] ** 2).reshape(-1)
            a_c = a - a.mean()
            b_c = b - b.mean()
            num = (a_c * b_c).mean()
            den = a_c.std() * b_c.std() + 1e-8
            lev_pred = num / den

            aux_loss = (lev_pred - self.aux_lev_target) ** 2
            return main_loss + self.aux_lev_weight * aux_loss
        return main_loss

    # ---------- Sampling ----------

    def denoise(
        self,
        model,
        x: torch.Tensor,
        sigma: torch.Tensor,  # (B,) or scalar
        cond: torch.Tensor | None = None,
        mkt_cond: torch.Tensor | None = None,
        sector_cond: torch.Tensor | None = None,
        guidance: float = 1.0,
    ) -> torch.Tensor:
        """Full denoiser D_theta(x, sigma) -> estimate of x_0."""
        B = x.shape[0]
        if sigma.ndim == 0:
            sigma = sigma.expand(B)
        view = (B,) + (1,) * (x.ndim - 1)
        sigma_v = sigma.view(view)
        c_in_v = self.c_in(sigma_v)
        c_out_v = self.c_out(sigma_v)
        c_skip_v = self.c_skip(sigma_v)
        c_noise_v = self.c_noise(sigma)

        cond_on = (cond is not None or mkt_cond is not None or sector_cond is not None)
        if guidance != 1.0 and cond_on:
            F_c = model(c_in_v * x, c_noise_v, cond=cond, mkt_cond=mkt_cond, sector_cond=sector_cond)
            F_u = model(c_in_v * x, c_noise_v, cond=None, mkt_cond=None, sector_cond=None)
            F_theta = F_u + guidance * (F_c - F_u)
        else:
            F_theta = model(c_in_v * x, c_noise_v, cond=cond, mkt_cond=mkt_cond, sector_cond=sector_cond)

        return c_skip_v * x + c_out_v * F_theta

    @torch.no_grad()
    def sample_heun(
        self,
        model,
        shape,
        steps: int = 18,
        cond: torch.Tensor | None = None,
        mkt_cond: torch.Tensor | None = None,
        sector_cond: torch.Tensor | None = None,
        guidance: float = 1.0,
        clip_range: float | None = 5.0,
        progress: bool = False,
    ) -> torch.Tensor:
        """
        EDM's deterministic Heun 2nd-order sampler.

        rho-schedule:
            sigma_i = (sigma_max^(1/rho) + i/(N-1) *
                       (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
            sigma_N = 0 (final cleaned sample)

        Typical steps=18 gives quality comparable to DDPM T=500.
        """
        device = self.device
        idx = torch.arange(steps, device=device, dtype=torch.float32)
        sigmas = (
            self.sigma_max ** (1.0 / self.rho)
            + idx / max(steps - 1, 1) *
              (self.sigma_min ** (1.0 / self.rho) - self.sigma_max ** (1.0 / self.rho))
        ) ** self.rho
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])  # sigma_N = 0

        # Initialize from Student-t, scaled by sigma_max
        x = self.sample_student_t(shape, device) * sigmas[0]

        iters = range(steps)
        if progress:
            try:
                from tqdm import tqdm
                iters = tqdm(iters, desc="edm-heun")
            except ImportError:
                pass

        for i in iters:
            s_cur = sigmas[i]
            s_next = sigmas[i + 1]

            # Euler step
            denoised = self.denoise(
                model, x, s_cur, cond=cond, mkt_cond=mkt_cond,
                sector_cond=sector_cond, guidance=guidance,
            )
            if clip_range is not None:
                denoised = denoised.clamp(-clip_range, clip_range)
            d_cur = (x - denoised) / s_cur
            x_next = x + (s_next - s_cur) * d_cur

            # 2nd-order correction (skip at last step where s_next = 0)
            if s_next > 0:
                denoised_next = self.denoise(
                    model, x_next, s_next, cond=cond, mkt_cond=mkt_cond,
                    sector_cond=sector_cond, guidance=guidance,
                )
                if clip_range is not None:
                    denoised_next = denoised_next.clamp(-clip_range, clip_range)
                d_next = (x_next - denoised_next) / s_next
                x_next = x + (s_next - s_cur) * 0.5 * (d_cur + d_next)

            x = x_next

        return x


def count_params(m) -> int:
    return sum(p.numel() for p in m.parameters())
