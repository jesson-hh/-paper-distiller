"""
Tiny InterDiff-style hierarchical denoiser.

Input  : x  (B, N, L, C)   noised panel window
         t  (B,)            diffusion timestep
Output : eps (B, N, L, C)  predicted noise

Architecture
------------
1. linear input projection C -> d_model
2. learnable temporal positional embedding   (1, 1, L, d)
3. learnable stock positional embedding      (1, N, 1, d)   (set-style)
4. sinusoidal time embedding -> 2-layer MLP -> film bias added per token
5. K stacked InterBlocks, each:
       a. intra-stock self-attention along L  (per-stock temporal mixing)
       b. inter-stock self-attention along N  (per-time cross-section)
       c. feedforward
   each sub-layer is pre-LN with residual.
6. output linear d_model -> C
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class MHSA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.drop(self.out(out))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class InterBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.intra = MHSA(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.inter = MHSA(d_model, n_heads, dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, mult=ff_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, L, D = x.shape

        # intra-stock: attend along L, batch over (B, N)
        h = self.ln1(x).reshape(B * N, L, D)
        h = self.intra(h).reshape(B, N, L, D)
        x = x + h

        # inter-stock: attend along N, batch over (B, L)
        h = self.ln2(x).permute(0, 2, 1, 3).reshape(B * L, N, D)
        h = self.inter(h).reshape(B, L, N, D).permute(0, 2, 1, 3)
        x = x + h

        # feedforward (point-wise)
        x = x + self.ff(self.ln3(x))
        return x


def _rolling_causal_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    """
    Causal rolling mean: out[t] = mean(x[max(0, t-window+1) : t+1]).

    x: (B, L)
    returns: (B, L), same shape, using left-zero-padding so very-early positions
             are averaged over fewer "real" points (approximate bias in first
             `window` steps is negligible for L >> window).
    """
    B, L = x.shape
    x_pad = F.pad(x.unsqueeze(1), (window - 1, 0), mode="constant", value=0.0)
    pooled = F.avg_pool1d(x_pad, kernel_size=window, stride=1)  # (B, 1, L)
    return pooled.squeeze(1)


def compute_rsv_from_mkt(mkt_cond: torch.Tensor, window: int = 5) -> torch.Tensor:
    """
    HAR-RV-L style asymmetric conditioning from the market factor series.

    Given mkt_cond: (B, L) market factor (equal-weight mean log_ret across
    the sampled stocks in each window), compute two rolling statistics:
      rsv_pos[t] = mean over past `window` days of ReLU(+r)^2    -- upside realized variance
      rsv_neg[t] = mean over past `window` days of ReLU(-r)^2    -- downside realized variance

    Return stacked (B, L, 2). Used as asymmetric conditioning for leverage
    effect -- past-negative-returns signal is explicit and pre-aggregated.
    """
    r = mkt_cond
    r_pos = F.relu(r)
    r_neg = F.relu(-r)
    rsv_pos = _rolling_causal_mean(r_pos * r_pos, window)
    rsv_neg = _rolling_causal_mean(r_neg * r_neg, window)
    return torch.stack([rsv_pos, rsv_neg], dim=-1)  # (B, L, 2)


class InterDenoiser(nn.Module):
    def __init__(
        self,
        n_channels: int,
        max_length: int,
        max_stocks: int,
        d_model: int = 64,
        n_blocks: int = 3,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
        n_regimes: int = 0,
        sign_cond: bool = False,
        lev_cond: bool = False,
        lev_window: int = 5,
        lev_mode: str = "both",
    ):
        super().__init__()
        self.sign_cond = sign_cond
        self.lev_cond = lev_cond
        self.lev_window = lev_window
        assert lev_mode in ("both", "pos_only", "neg_only"), \
            f"lev_mode must be both/pos_only/neg_only, got {lev_mode!r}"
        self.lev_mode = lev_mode
        self.in_proj = nn.Linear(n_channels, d_model)
        self.t_pos = nn.Parameter(torch.zeros(1, 1, max_length, d_model))
        self.s_pos = nn.Parameter(torch.zeros(1, max_stocks, 1, d_model))
        nn.init.trunc_normal_(self.t_pos, std=0.02)
        nn.init.trunc_normal_(self.s_pos, std=0.02)

        self.n_regimes = n_regimes
        if n_regimes > 0:
            self.regime_embed = nn.Embedding(n_regimes, d_model)
            nn.init.trunc_normal_(self.regime_embed.weight, std=0.02)

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.blocks = nn.ModuleList(
            [InterBlock(d_model, n_heads, ff_mult, dropout) for _ in range(n_blocks)]
        )
        self.ln_out = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, n_channels)
        self.d_model = d_model

        # Market factor conditioning: (B, L) -> (B, 1, L, d_model)
        self.mkt_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Per-stock sector factor conditioning: (B, N, L) -> (B, N, L, d_model)
        self.sector_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Sign-aware (asymmetric) conditioning: extra branches that see ONLY
        # the negative-clipped factors. Breaks the linear symmetry of the
        # additive-projection scheme so the model can learn leverage effect
        # (corr(r_t, r_{t+1}^2) > 0 when r_t < 0).
        if self.sign_cond:
            self.mkt_neg_proj = nn.Sequential(
                nn.Linear(1, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.sector_neg_proj = nn.Sequential(
                nn.Linear(1, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

        # Leverage conditioning: (B, L, 2) past realized semi-variance
        # [rsv_pos, rsv_neg], computed deterministically from mkt_cond at
        # forward time. Explicit time-lagged asymmetric signal (HAR-RV-L).
        if self.lev_cond:
            self.lev_proj = nn.Sequential(
                nn.Linear(2, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
        mkt_cond: torch.Tensor | None = None,
        sector_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, L, C = x.shape
        h = self.in_proj(x)
        h = h + self.t_pos[:, :, :L, :] + self.s_pos[:, :N, :, :]

        if cond is not None and self.n_regimes > 0:
            h = h + self.regime_embed(cond)  # (B, N, L, d)

        if mkt_cond is not None:
            # mkt_cond: (B, L) -> (B, 1, L, 1) -> project -> (B, 1, L, d) -> broadcast
            me = self.mkt_proj(mkt_cond[:, None, :, None])  # (B, 1, L, d_model)
            h = h + me  # broadcast across N stocks
            if self.sign_cond:
                # Negative-clipped: ReLU(-m_t); non-zero only on down-moves
                neg = torch.clamp(-mkt_cond, min=0.0)
                h = h + self.mkt_neg_proj(neg[:, None, :, None])
            if self.lev_cond:
                # Realized semi-variance (past window days). Shape (B, L, 2).
                # For lev_mode="pos_only" zero out rsv_neg channel (and vice
                # versa for "neg_only") so the model is forced to learn
                # leverage response from a single direction's semi-variance.
                lev = compute_rsv_from_mkt(mkt_cond, self.lev_window)
                if self.lev_mode == "pos_only":
                    zero = torch.zeros_like(lev[..., 1])
                    lev = torch.stack([lev[..., 0], zero], dim=-1)
                elif self.lev_mode == "neg_only":
                    zero = torch.zeros_like(lev[..., 0])
                    lev = torch.stack([zero, lev[..., 1]], dim=-1)
                h = h + self.lev_proj(lev[:, None, :, :])

        if sector_cond is not None:
            # sector_cond: (B, N, L) — per-stock sector factor
            se = self.sector_proj(sector_cond[:, :, :, None])  # (B, N, L, d_model)
            h = h + se
            if self.sign_cond:
                neg_s = torch.clamp(-sector_cond, min=0.0)
                h = h + self.sector_neg_proj(neg_s[:, :, :, None])

        te = sinusoidal_time_embedding(t, self.d_model)
        te = self.time_mlp(te)[:, None, None, :]
        h = h + te

        for blk in self.blocks:
            h = blk(h)

        return self.out(self.ln_out(h))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
