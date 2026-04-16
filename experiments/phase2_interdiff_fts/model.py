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
    ):
        super().__init__()
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

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
        mkt_cond: torch.Tensor | None = None,
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

        te = sinusoidal_time_embedding(t, self.d_model)
        te = self.time_mlp(te)[:, None, None, :]
        h = h + te

        for blk in self.blocks:
            h = blk(h)

        return self.out(self.ln_out(h))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
