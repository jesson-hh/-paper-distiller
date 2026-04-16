"""
Online panel-window sampler for InterDiff-style training.

Each iteration yields a tensor of shape (k_stocks, length, C) where the
k stocks are sampled uniformly from those that have *fully valid* data
across [start, start+length).

Channels are derived from the raw panel at construction time:
    log_ret  = diff(log(close * factor))
    log_hc   = log(high*factor) - log(close*factor)        (intraday range)
    log_lc   = log(low*factor)  - log(close*factor)
    log_oc   = log(open*factor) - log(close*factor)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from regimes import RegimeSpec, fit_regimes, label as regime_label


def _safe_log(x: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=np.float32)
    pos = (x > 0) & np.isfinite(x)
    out[pos] = np.log(x[pos])
    return out


def derive_panel_returns(panel: np.ndarray, fields: list[str]) -> np.ndarray:
    f = {n: i for i, n in enumerate(fields)}
    factor = panel[:, :, f["factor"]] if "factor" in f else np.ones_like(panel[:, :, 0])
    adj_close = panel[:, :, f["close"]] * factor
    adj_high = panel[:, :, f["high"]] * factor
    adj_low = panel[:, :, f["low"]] * factor
    adj_open = panel[:, :, f["open"]] * factor

    la = _safe_log(adj_close)
    log_ret = np.diff(la, axis=1)

    c1 = adj_close[:, 1:]
    log_hc = (_safe_log(adj_high[:, 1:]) - _safe_log(c1)).astype(np.float32)
    log_lc = (_safe_log(adj_low[:, 1:]) - _safe_log(c1)).astype(np.float32)
    log_oc = (_safe_log(adj_open[:, 1:]) - _safe_log(c1)).astype(np.float32)

    return np.stack([log_ret, log_hc, log_lc, log_oc], axis=-1).astype(np.float32)


def per_stock_stats(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(returns)
    masked = np.where(finite, returns, 0.0)
    counts = finite.sum(axis=1).clip(min=1)
    mean = masked.sum(axis=1) / counts
    var = ((np.where(finite, returns - mean[:, None, :], 0.0)) ** 2).sum(axis=1) / counts
    std = np.sqrt(var) + 1e-6
    return mean, std


class PanelWindowDataset(IterableDataset):
    """
    Yields normalised panel windows.

    Item shape: (k_stocks, length, C)   torch.float32

    The dataset is *infinite* — sampling stops only when the training
    loop breaks. Training-step count is controlled by the caller.
    """

    def __init__(
        self,
        panel_npz: str | Path,
        length: int = 32,
        k_stocks: int = 16,
        seed: int = 0,
        normalise: bool = True,
        time_range: tuple[str, str] | None = None,
        regime_window: int = 0,
        n_regimes: int = 0,
        regime_spec: RegimeSpec | None = None,
    ):
        d = np.load(panel_npz, allow_pickle=True)
        panel = d["panel"]
        fields = d["fields"].tolist()
        dates = d["dates"].astype(str)

        returns = derive_panel_returns(panel, fields)
        ret_dates = dates[1:]
        if time_range is not None:
            lo, hi = time_range
            mask = (ret_dates >= lo) & (ret_dates <= hi)
            if not mask.any():
                raise ValueError(f"empty time_range {time_range}")
            returns = returns[:, mask, :]
            ret_dates = ret_dates[mask]

        if normalise:
            mean, std = per_stock_stats(returns)
            returns = (returns - mean[:, None, :]) / std[:, None, :]
            self.mean = mean
            self.std = std
        else:
            self.mean = np.zeros((returns.shape[0], returns.shape[2]), dtype=np.float32)
            self.std = np.ones_like(self.mean)

        self.returns = np.where(np.isfinite(returns), returns, 0.0).astype(np.float32)
        self.valid = np.isfinite(returns).all(axis=2)

        self.length = length
        self.k_stocks = k_stocks
        self.dates = ret_dates
        self.codes = d["codes"]

        N, T = self.valid.shape
        if T < length:
            raise ValueError(f"series too short: T={T} < length={length}")

        cum = np.zeros((N, T + 1), dtype=np.int32)
        cum[:, 1:] = np.cumsum(self.valid.astype(np.int32), axis=1)
        full = (cum[:, length:] - cum[:, :T - length + 1]) == length
        self._stock_full_mask_per_start = full

        valid_starts = full.any(axis=0)
        self._valid_starts = np.where(valid_starts)[0]
        if self._valid_starts.size == 0:
            raise ValueError("no fully-valid windows of requested length")

        self._rng = np.random.default_rng(seed)
        self.n_channels = self.returns.shape[2]

        use_regimes = (n_regimes > 0 and regime_window > 0) or regime_spec is not None
        if use_regimes:
            r_for_fit = self.returns[..., 0]
            if regime_spec is None:
                regime_spec = fit_regimes(r_for_fit, window=regime_window, n_regimes=n_regimes)
            self.regime_spec = regime_spec
            self.regime_labels = regime_label(r_for_fit, regime_spec)
        else:
            self.regime_spec = None
            self.regime_labels = None

    def __iter__(self):
        rng = self._rng
        L = self.length
        k = self.k_stocks
        while True:
            s = int(rng.choice(self._valid_starts))
            ok_stocks = np.where(self._stock_full_mask_per_start[:, s])[0]
            if ok_stocks.size < k:
                continue
            picks = rng.choice(ok_stocks, size=k, replace=False)
            window = self.returns[picks, s : s + L, :]
            # Market factor: equal-weight mean of log_ret across sampled stocks
            mkt = window[:, :, 0].mean(axis=0).astype(np.float32)  # (L,)
            if self.regime_labels is not None:
                lab = self.regime_labels[picks, s : s + L]
                yield torch.from_numpy(window), torch.from_numpy(lab), torch.from_numpy(mkt)
            else:
                yield torch.from_numpy(window), torch.from_numpy(mkt)

    def info(self) -> dict:
        return {
            "n_stocks": int(self.returns.shape[0]),
            "T": int(self.returns.shape[1]),
            "n_channels": int(self.n_channels),
            "length": int(self.length),
            "k_stocks": int(self.k_stocks),
            "n_valid_starts": int(self._valid_starts.size),
            "mean_eligible_per_start": float(
                self._stock_full_mask_per_start[:, self._valid_starts].sum(0).mean()
            ),
        }
