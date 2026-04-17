---
title: "A-Share Positive Leverage Effect — Sign-Flipped vs Classical"
category: "open-problems"
slug: "a-share-positive-leverage"
tags: ["leverage-effect", "a-share", "market-microstructure", "empirical", "open-problem"]
refs: ["Black 1976"]
links: ["m7-m8-modernization-and-leverage", "factor-conditional-interdiff-m4-m5", "fts-interdiff-fusion"]
created: "2026-04-17T23:40:00"
updated: "2026-04-17T23:40:00"
---

# A-Share Positive Leverage Effect

> **Open problem**: In CSI800 (2015–2024), $\text{corr}(r_t, r_{t+1}^2) = +0.013$. This is the OPPOSITE sign of the classical leverage effect (Black 1976) observed in US/European equities, where the correlation is negative.
> **Discovered via**: M4-M8 synthetic generation ablations, where all our models produced the classical negative leverage (-0.003 to -0.011) while real CSI800 data shows positive leverage.
> **Relation**: [[m7-m8-modernization-and-leverage]] is the experimental write-up. [[factor-conditional-interdiff-m4-m5]] is where the gap first surfaced.

## The numbers

Leverage statistic $\rho = \text{corr}(r_t, r_{t+1}^2)$ computed on L=64 bootstrap windows:

| dataset | window | $\rho$ | sign vs classical |
|---|---|---|---|
| CSI800 real | 64-day | **+0.0135** | **FLIPPED** |
| Our M4-M8 synthetic | 64-day | -0.003 ~ -0.011 | classical |
| US equity (textbook) | daily | -0.02 ~ -0.05 | classical |

All our diffusion models, regardless of architecture or conditioning (M4 market-only, M5 market+sector, M6 CSI800 scaled, M7 with CFG, M8 with sign-aware branches), produce the **classical negative leverage**. Real A-share CSI800 is **consistently positive**.

This isn't a sampler artifact — DDPM-500 and DDIM-50 give the same sign. Isn't a CFG artifact — guidance sweeps {1.0, 1.5, 3.0, 5.0} all keep negative. Isn't capacity — sign-aware conditioning smoke-tests proved the model CAN express arbitrary asymmetry.

## The classical leverage effect (why negative is "expected")

Fischer Black (1976): "the return of the stock is negatively correlated with changes in the volatility of the return." Mechanism: negative returns reduce equity value relative to debt, raising financial leverage, raising perceived risk, raising implied volatility. This generates:

$$\text{corr}(r_t, \sigma_{t+1}) < 0 \implies \text{corr}(r_t, r_{t+1}^2) < 0$$

Robust in US/European equity markets for decades. Modeled by EGARCH (Nelson 1991), GJR-GARCH, stochastic volatility with leverage parameter.

## Why A-share might flip it

Several structural features of Chinese A-share markets that don't apply to US equities:

### 1. Daily price limits (涨停 / 跌停, ±10%)

Extreme down-moves are **capped**, which:
- Suppresses the magnitude of $r_t$ when $r_t < 0$ is extreme
- Also delays negative information (sustained 跌停 for multiple days)
- Breaks the sharp "big down → big next-day vol" empirical coupling

Conversely, 涨停 (up-limit) triggers well-documented "一字板" patterns — once locked at limit, queue of buy orders creates informational overhang that bursts into high-volume volatility the next day.

### 2. Retail dominance + momentum chasing

A-share has much higher retail participation than developed markets. Retail traders are more prone to:
- Chasing rallies (追涨) — positive returns attract buyers, amplifying next-day vol
- Holding losers (割肉困难) — negative returns lead to passive holding, muting vol

This asymmetry tilts $\text{corr}(r_t > 0, |r_{t+1}|)$ upward.

### 3. T+1 settlement

Buy today, sell tomorrow. Means:
- Intraday reversal after buying is impossible — positive-return days accumulate unsold positions
- Overnight gap on day t+1 carries unresolved information from t+1 → higher overnight variance after positive days

Classical leverage in US is driven partly by intraday hedging that A-share T+1 prevents.

### 4. IPO/limit-up lottery speculation

New listings and limit-up stocks attract speculative buying. After a limit-up (positive $r_t$), the next day often has dramatic opening gaps ("炒新"现象), inflating $r_{t+1}^2$.

### 5. Index-level vs single-stock

Classical leverage is often strongest at the index level (common-factor volatility). CSI800 = CSI300 + CSI500, dominated by large caps. The sign flip is more pronounced at the single-stock level (where retail speculation dominates) than the pure index. Our leverage statistic is aggregated across stock-trajectories, which may tilt it further positive.

## Why our diffusion models can't learn it

Our training loss is MSE on noise prediction. This penalizes pointwise reconstruction error but has **no term that targets second-moment lagged correlations**. The model converges to whatever minimizes local noise MSE, and among the many denoisers with similar MSE, the one the optimizer reaches first tends to have classical (negative) leverage — probably because:

- The denoiser is smooth, GELU-based → slight monotone dependence of $|r_{t+1}|$ on $|r_t|$ (not signed)
- Gaussian noise injection in forward process is symmetric → reconstruction naturally symmetric
- Sign-aware branches (M8) add capacity but gradient signal doesn't select asymmetry direction

So we produce the "generic" leverage sign, which happens to be classical.

## Open questions

1. **Is this universal to A-share or CSI800-specific?**
   Check: CSI300 (pure large cap), CSI500 (mid cap), CSI1000 (small cap), STAR50 (科创板).
   Prediction: smaller cap → stronger positive leverage (retail dominance).

2. **What time scale does it hold on?**
   Check: intraday (1-min data), daily (current), weekly, monthly leverage.
   Prediction: positive at daily (our result), might flip to classical at longer horizons (institutional hedging takes over).

3. **Is it stationary?**
   Check: rolling 252-day leverage estimate 2015-2024.
   Prediction: changes around regulatory events — the 2015 crash, 2016 fuse-breaker, 2020 COVID, each might have a regime shift.

4. **Can a minimal augmentation fix it without breaking other metrics?**
   Candidates:
   - Explicit `leverage_lag1` auxiliary loss with small weight $w$
   - Asymmetric noise schedule: $\beta(t, r<0) \ne \beta(t, r>0)$
   - Two-stage: GJR-GARCH on market factor (captures leverage sign) → InterDiff conditional on factor trajectory

5. **Does the synthetic "wrong leverage" hurt downstream tasks?**
   Answered by the upcoming α-sweep: if models trained on synthetic + real data outperform real-only, the leverage mismatch is cosmetic. If they underperform, leverage is load-bearing.

## Experimental budget to close this

- **Sanity check** (1 hour): Run leverage statistic on CSI300, CSI500, CSI1000 real data separately. Confirm positive sign holds across subsets.
- **Time-varying check** (2 hours): 252-day rolling leverage 2015-2024 on CSI800. Overlay regulatory event dates.
- **Aux-loss attempt** (1 day): Train M9 with explicit leverage aux loss, weight sweep. Validate all 7 verdicts remain OK.
- **Two-stage attempt** (1 week): GARCH market factor + InterDiff residual. More work but principled fix.

## Related

- [[m7-m8-modernization-and-leverage]] — experimental writeup that surfaced this
- [[factor-conditional-interdiff-m4-m5]] — where the persistent leverage gap was first noted as "leverage asymmetry"
- Black, F. (1976). "Studies of stock price volatility changes." ASA Proceedings.
