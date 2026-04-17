---
title: "M7/M8 — Modernization (bf16+DDIM+CFG) and the Leverage Puzzle"
category: "articles"
slug: "m7-m8-modernization-and-leverage"
tags: ["diffusion", "finance", "bf16", "ddim", "classifier-free-guidance", "leverage-effect", "ablation", "csi800"]
refs: ["experiments/phase2_interdiff_fts"]
links: ["factor-conditional-interdiff-m4-m5", "fts-interdiff-fusion", "factor-conditional-denoising", "a-share-positive-leverage"]
created: "2026-04-17T23:30:00"
updated: "2026-04-17T23:30:00"
---

# M7/M8 — Modernization + Leverage Puzzle

> **One-line**: Added bf16 + DDIM + CFG (M7), confirmed 3 real wins; tried sign-aware conditioning (M8) to fix leverage — failed, but uncovered that **A-share CSI800 has POSITIVE leverage (+0.013)** while all our models produce classical NEGATIVE leverage (-0.007 ~ -0.011).
> **Relation**: Continuation of [[factor-conditional-interdiff-m4-m5]]. Root cause analysis in [[a-share-positive-leverage]].

## Modernization layer (M7)

After M6 on CSI800 hit all 7 verdict metrics green except leverage, we audited our diffusion engine and found it was stuck on 2020 tech. Added three modernizations:

### 1. bf16 mixed precision training (`--bf16`)

Wrap `diff.training_loss` and the preflight forward/backward in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. Model weights and optimizer state stay in fp32; only computation is bf16.

**Result on RTX 5090, CSI800 config**:

| metric | fp32 (M6) | **bf16 (M7)** | Δ |
|---|---|---|---|
| step/s | 22.5 | **32.0** | +42% |
| peak GPU | 2.43 GB | **1.46 GB** | -40% |
| ema loss | 0.1325 | 0.1337 | +1% |
| wall time (20k steps) | 15 min | **10.5 min** | -30% |

The 42% speedup is less than the theoretical 2x because the model is small enough that the attention kernels aren't compute-bound. **Free 40% memory headroom** is the real win — opens space for bigger models or longer windows on the same hardware.

### 2. DDIM deterministic sampler (`--sampler ddim --ddim-steps 50`)

Song 2021 skip-step formula. Added `sample_ddim` method to `GaussianDiffusion` that selects a subset of `T` diffusion timesteps (linspace) and uses:

$$x_{t_{k-1}} = \sqrt{\bar\alpha_{k-1}}\, \hat x_0 + \sqrt{1-\bar\alpha_{k-1}-\sigma_k^2}\,\epsilon_\theta(x_{t_k}, t_k) + \sigma_k z$$

With `eta=0` (fully deterministic), `eta=1` reduces to DDPM ancestral sampling on the selected subsequence.

**Benchmark on batch=8, k=32, L=64, RTX 5090**:

| Sampler | Steps | Guidance | Time (ms) | Speedup |
|---|---|---|---|---|
| DDPM | 500 | 1.0 | 2675 | 1.0x |
| **DDIM** | 25 | 1.0 | **149** | **18.0x** |
| DDIM | 50 | 1.0 | 265 | 10.1x |
| DDIM | 100 | 1.0 | 533 | 5.0x |
| DDIM | 25 | 3.0 (CFG) | 253 | 10.6x |
| DDIM | 50 | 3.0 (CFG) | 494 | 5.4x |

CFG doubles cost because of two forward passes per step. Stylized-fact metrics on DDIM-50 are indistinguishable from DDPM-500 in our tests.

### 3. Classifier-free guidance (`--cfg-drop 0.1` in train, `--guidance G` at sample)

Training modification: with probability `cfg_drop` per batch, set all conditioning to `None`. At inference:

$$\tilde\epsilon = \epsilon_\theta(x_t, \emptyset) + G \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

Guidance scale `G=1.0` is pure conditional (no overhead). `G>1` amplifies conditioning. `sample.py` warns if ckpt had `cfg_drop=0` but user requests `G != 1.0` (the unconditional branch wouldn't be properly learned).

### M7 guidance sweep on CSI800

| metric | REAL | g=1.0 | g=1.5 | g=3.0 | g=5.0 |
|---|---|---|---|---|---|
| std | 0.0274 | 0.0267 | 0.0295 | 0.0278 | 0.0365 |
| excess_kurt | 3.55 | 2.90 | 2.72 | 1.96 | 1.91 |
| hill_right / left | 3.46 / 3.00 | 3.84 / 3.19 | 3.93 / 3.29 | 4.75 / 3.87 | 3.86 / 4.66 |
| acf_r² lag1 | 0.074 | 0.069 | 0.069 | 0.067 | 0.052 |
| **leverage_lag1** | **+0.013** | **-0.011** | **-0.003** | **-0.011** | **-0.010** |
| panel_mean_pair_corr | 0.349 | 0.376 | 0.314 | 0.315 | 0.154 |
| panel_max_eig_frac | 0.398 | 0.418 | 0.362 | 0.362 | 0.226 |

**Observations**:
1. `g=1.5` gives the best leverage (-0.003, halfway to real +0.013) but still negative
2. `g>=3.0` **destroys** other metrics: kurt collapses (1.96), panel correlation fails (0.15)
3. `g=1.0` is worst for leverage of any setting — suggests `cfg_drop` during training slightly hurt compared to M6 without cfg_drop (M6 was -0.007)

**Conclusion**: CFG can tune a tradeoff knob but is **not a fix** for leverage. Classical CFG amplifies whatever conditional-vs-unconditional signal the model learned; if the model didn't learn the right asymmetric structure, amplifying it is useless or harmful.

## M8 — Sign-aware conditioning (failed leverage fix)

### Design

Hypothesis: leverage requires asymmetric response to factor sign. Our `mkt_proj: Linear(1, d) -> GELU -> Linear(d, d)` is nearly linear — `out(-m) ≈ -out(m)`. Break the symmetry by adding an explicit branch that sees only negative-clipped factor:

```python
if sign_cond:
    self.mkt_neg_proj = Sequential(Linear(1,d), GELU(), Linear(d,d))
    self.sector_neg_proj = Sequential(Linear(1,d), GELU(), Linear(d,d))

# forward:
if mkt_cond is not None:
    h += self.mkt_proj(mkt_cond[:, None, :, None])
    if self.sign_cond:
        h += self.mkt_neg_proj(relu(-mkt_cond)[:, None, :, None])

if sector_cond is not None:
    h += self.sector_proj(sector_cond[:, :, :, None])
    if self.sign_cond:
        h += self.sector_neg_proj(relu(-sector_cond)[:, :, :, None])
```

+8.6k params on d_model=64 test, ~34k on d_model=128 production.

Smoke test verified genuine asymmetry:
```
|model(-mkt) - model(+mkt)|.mean() = 0.153  (vs ~0 before)
```

### Training and evaluation

M8 config = M7 + `--sign-cond` − `--cfg-drop` (isolated the sign-cond effect).

| metric | Real | M6 | M7 g=1.0 | **M8 DDIM** | **M8 DDPM** |
|---|---|---|---|---|---|
| ema loss | — | 0.1325 | 0.1337 | 0.1332 | — |
| leverage_lag1 | **+0.013** | -0.007 | -0.011 | -0.010 | **-0.007** |
| excess_kurt | 3.55 | 3.03 | 2.90 | 2.95 | 3.01 |
| panel_mean_pair_corr | 0.349 | 0.347 | 0.376 | 0.368 | 0.349 |
| all other verdicts | — | 7/7 OK | 7/7 OK | 7/7 OK | 7/7 OK |

**Sign-cond did not fix leverage**. M8 DDPM leverage (-0.007) matches M6 (-0.007); M8 DDIM (-0.010) is marginally worse than DDPM.

### Why sign-cond failed

The capacity exists (smoke test proves the model is asymmetric), but the training signal doesn't exploit it. Our MSE-on-noise loss has no gradient term directly pushing leverage in either direction — the model converges to whatever minimizes noise prediction MSE, which happens to land at slight classical-negative leverage across M4-M8 (consistent -0.003 to -0.011).

Giving the model capacity without a matching training signal is like adding a muscle with no innervation.

## The real finding: A-share positive leverage

The deeper story, separated out to [[a-share-positive-leverage]]:

| dataset | leverage_lag1 sign |
|---|---|
| US equities (Black 1976 classical) | **negative** (down → vol) |
| Our CSI800 real | **+0.013 (positive)** |
| All our models (M4-M8) | -0.003 to -0.011 (classical) |

**Our models are producing physically-natural classical leverage. Real A-share violates it.** This reframes the "leverage bug" as an A-share microstructure quirk that our data-agnostic diffusion doesn't capture.

A-share-specific mechanisms that could flip the sign:
- ±10% 涨跌停 (daily price limits) cap downside, but up-limits trigger next-day limit-up cascades with high volume/vol
- Heavy retail participation + momentum chasing amplifies post-rally vol
- T+1 settlement prevents intraday reversal after rallies → overnight gap vol

None of these are learnable from pure returns data; they're structural features of the market our denoiser can't infer.

## Decision: accept the gap, move on

Option A (explicit leverage auxiliary loss): risky, prone to `aux_market_weight`-style tradeoffs where pushing one metric distorts others.

Option B (accept): **chosen**. We have 7/7 OK verdicts and 15/16 indicators in the right range. Leverage is a second-order effect (|0.013| is small even in real data) whose direction is A-share-specific rather than universal. Moving to α-sweep to validate the downstream value of the current synthetic quality.

Option C (two-stage GARCH + InterDiff): expensive engineering. Reserved for if α-sweep shows synthetic data is materially worse than real when leverage is used.

## Artifacts

```
experiments/phase2_interdiff_fts/
├── diffusion.py       # GaussianDiffusion + sample_ddim + CFG path
├── model.py           # + sign_cond flag + mkt_neg_proj + sector_neg_proj
├── train.py           # + --bf16 --cfg-drop --sign-cond
├── sample.py          # + --sampler ddim --ddim-steps --ddim-eta --guidance
└── ckpts/
    ├── M0_m7_modern_step20000.pt      # bf16 + cfg_drop=0.1
    ├── M0_m7_modern_step20000.g{1.0,1.5,3.0,5.0}.samples.npz
    ├── M0_m8_signcond_step20000.pt    # bf16 + sign_cond
    └── M0_m8_signcond_step20000.{ddim,ddpm}.samples.npz
```

## Next

- [[alpha-sweep-csi800-m6]] — downstream IC sweep on model-collapse phase transition
- If α-sweep shows collapse, revisit leverage via explicit aux loss (Option A) or two-stage GARCH (Option C)
