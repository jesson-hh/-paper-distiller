---
title: "Recursive Collapse Experiment on CSI800 (M6 → Gen3)"
category: "articles"
slug: "recursive-collapse-csi800"
tags: ["model-collapse", "recursive-training", "diffusion", "csi800", "m6", "empirical"]
refs: ["Shumailov 2024", "experiments/phase2_interdiff_fts/recursive_train.py"]
links: ["lgbm-alpha-sweep-phase-transition", "alpha-sweep-csi800-m6", "factor-conditional-interdiff-m4-m5", "phase-transition-alpha-star-empirical"]
created: "2026-04-18T01:30:00"
updated: "2026-04-18T01:30:00"
---

# Recursive Collapse Experiment

> **Setup**: Classical Shumailov-2024-style self-consuming loop. Gen0 = M6 (trained on real CSI800). Each subsequent gen trains a **fresh** InterDenoiser ONLY on the previous generation's synthetic samples — no real data injection. Track stylized-fact and cross-section drift across generations to test whether M6's residual biases amplify recursively.
> **Relation**: The "paranoid" complement to [[lgbm-alpha-sweep-phase-transition]]'s one-step α-sweep. Measures actual collapse trajectory, not just one-step augmentation utility.

## Pipeline

```
Gen 0 (baseline):
    M6 = InterDenoiser_big, trained on REAL CSI800 panel, 20k steps
    Gen0 samples: sample.py on M6 real-trained ckpt, 50×8 = 400 panels

For g in [1, 2, 3]:
    1. Sample N=2000 panels from Gen(g-1)   (DDIM 50 steps, fast)
    2. Stitch panels into pseudo-panel data/rec_g{g}.npz:
         - 32 pseudo-stocks
         - n_panels × 64 pseudo-days concatenated
         - faked OHLC from log_ret+log_hc+log_lc+log_oc via cumprod
           (exact reconstruction — no information loss)
    3. Train Gen(g) = fresh InterDenoiser on rec_g{g}.npz
         - Same architecture (d_model=128, 6 blocks, 8 heads)
         - 8k steps (reduced from 20k — synthetic is cleaner signal,
           less noise to average out)
         - bf16 enabled
    4. Sample from Gen(g): 50×8 = 400 panels
    5. Eval Gen(g) against REAL CSI800 via eval_compare
       → stylized_facts verdict (7 metrics) + leverage sign
```

## Why this matters

Classical model-collapse theory (Shumailov 2024) predicts that recursive training on synthetic data eventually loses tail coverage — kurtosis collapses, modes disappear, distribution shrinks toward its mean. Our α-sweep tested **one-step** augmentation quality, which is a much weaker condition. The recursive loop tests whether M6's specific biases (slight kurt gap, wrong-sign leverage, market_factor_var gap) compound generation-by-generation.

If the pipeline collapses, we learn:
- M6 is not "closed under self-training" — its biases amplify
- The one-step α-sweep was optimistic

If it doesn't collapse:
- M6 is robust to self-consumption
- Factor-conditional conditioning (market + sector factors bootstrapped from each gen) provides enough structural anchoring to prevent classical collapse

## Results

### Per-generation summary (eval against real CSI800)

| gen | trained on | std | kurt | hill_R | hill_L | acf_r2_1 | **leverage** | **pair_corr** | **verdicts** |
|---|---|---|---|---|---|---|---|---|---|
| **0** (M6) | real | 0.0283 | 3.03 | 3.83 | 3.16 | 0.066 | -0.007 | **0.347** | **7/7 OK** |
| 1 | M6 synth | 0.0260 | 2.71 | 4.00 | 3.24 | 0.061 | **-0.016** | 0.394 (+13%) | 6/7 (pair MEH) |
| 2 | gen1 synth | 0.0246 | 2.87 | 3.86 | 3.14 | 0.078 | **-0.029** | **0.429 (+23%)** | 5/7 (pair FAIL, max_eig MEH) |
| 3 | gen2 synth | 0.0247 | 2.85 | 3.89 | 3.15 | 0.089 | -0.029 | **0.483 (+38%)** | **5/7** (pair FAIL, max_eig FAIL) |

Real reference: kurt=3.55, hill_R=3.46, hill_L=3.00, leverage=+0.013, pair_corr=0.349.

### Drift signatures

**What collapsed monotonically:**

1. **panel_mean_pair_corr blows up** — 0.347 → 0.394 → 0.429 → 0.483, about **+10%-per-generation** with no sign of stopping. Exceeds MEH at gen 1, FAIL at gen 2. By gen 3, the synthetic pair correlation is **38% above real**. This is the dominant collapse signature.
2. **panel_max_eig_frac inflates** in lockstep (0.427 → 0.487 across gens, from ~0.40 real). Common-mode structure is getting stronger, not weaker.
3. **leverage doubles and redoubles** — -0.007 → -0.016 → -0.029 → -0.029. The wrong-sign bias from [[a-share-positive-leverage]] is amplifying 2× per gen until it saturates. By gen 2, leverage is **4.3× further from real** than at gen 0.
4. **Verdicts drop 7 → 6 → 5 → 5.**

**What did NOT collapse classically:**

1. **Kurtosis stabilises** at 2.7-2.9, not driving toward Gaussian (real is 3.55). No Shumailov-style "tail coverage loss".
2. **Hill indices oscillate** weakly (3.83 → 4.00 → 3.86 → 3.89) — no monotonic thinning of tails.
3. **std shrinks only slightly** (0.028 → 0.025, -10%). Not the dramatic variance collapse predicted by vanilla i.i.d. recursive-training theory.
4. **acf_r²_lag1 actually grows** across generations (0.066 → 0.061 → 0.078 → 0.089). More vol-clustering, not less — opposite of what pure recursive collapse would predict.

### The non-classical collapse pattern

Classical model-collapse theory (Shumailov, Dohmatob-etal) assumes i.i.d. or mild autoregressive generators without strong conditioning, and predicts **distribution narrowing** — tails get thinner, modes merge, variance contracts.

Our generator has **explicit market + sector factor conditioning**, and the collapse signature is different: **common-mode amplification**, not variance narrowing. Each generation:
1. Samples from the prior gen using its bootstrapped market and sector factors
2. The prior gen's factor output is slightly over-amplified (M6 had a small positive delta on pair_corr baseline)
3. When gen g+1 is trained on those outputs, it learns the slightly-too-strong factor and amplifies again
4. Compounds geometrically: ~1.10× per gen on pair_corr → 1.38× by gen 3

This is a **conditioning-specific collapse mode**: the factor conditioning we introduced in [[factor-conditional-denoising]] to SOLVE the cross-section gap is the exact mechanism that causes **amplification-style collapse** under recursive self-training. The 1-step α-sweep (see [[lgbm-alpha-sweep-phase-transition]]) doesn't catch this because α-mixing always anchors a fraction of real data; recursive collapse has zero real-data anchoring.

### Leverage: bias amplification, then saturation

Leverage goes -0.007 → -0.016 → -0.029 → -0.029. The first two generations roughly double each time; gen 3 saturates. Hypothesis:
- M6's wrong-sign leverage is small but systematic
- Gen 1 learns from M6 output where the sign is *even more* wrong (because sampling lands slightly toward typical M6 outputs, losing what little correct-sign variance was left)
- After 2-3 generations, leverage is locked at the maximum the sign-aware architecture can express without more capacity — the -0.029 ceiling is probably related to M8's [[m7-m8-modernization-and-leverage|sign-aware smoke test]] showing ~0.15 asymmetry magnitude

### What this means for downstream use

- **One-step augmentation**: Safe across all α (confirmed by [[lgbm-alpha-sweep-phase-transition]])
- **Recursive self-training**: Only safe for 0 generations (use M6 directly). By gen 1 already has pair_corr MEH; by gen 2 it's FAIL. **Do not retrain from our own synthetic without real-data anchoring.**
- **Mitigation**: If recursive training is required (e.g. continual self-improvement), must re-anchor with a real-data fraction each generation. α-sweep data suggests α=0.5 mix prevents common-mode amplification by keeping 50% real every step.

## Methodological notes

### Factor conditioning in recursive mode

When we train Gen(g) on Gen(g-1) synthetic, the `PanelWindowDataset` computes new market factor and per-stock sector factor FROM THE SYNTHETIC PANEL. So:
- Gen 1 market factor = mean across the 32 synth stocks in each window
- Sector labels for synthetic stocks are cyclically copied from real CSI800 sectors
- Gen 1's "market factor" is therefore an artifact of Gen 0's synthesis

This means Gen 1's conditioning distribution is a degraded echo of Gen 0's. Each subsequent gen narrows this further. If any specific factor shape (e.g. high-vol regimes) is under-represented in Gen 0's output, Gen 1 sees even less of it, and so on. Combined with the bias amplification observed above, this explains why common-mode strength grows geometrically.

### Pseudo-panel construction fidelity

The synth→pseudo-panel reconstruction is exact: we set close[t+1] = close[t] * exp(log_ret[t]) and back-compute open/high/low from the original log_hc/log_lc/log_oc channels. When PanelWindowDataset derives log returns from this pseudo-panel, it reproduces the original 4 channels byte-identically. Zero information loss in the round-trip.

### 32 pseudo-stocks is a bottleneck

Real CSI800 has 1324 stocks; the pseudo-panel has only 32. This limits:
- Cross-section diversity per window (k_stocks=32 panels cover 100% of available stocks)
- Regime diversity across time (2000 panels × 64 days = 128k pseudo-days, but all with the same 32 cyclical "stock identities")

This likely contributes to the common-mode amplification — with only 32 distinct stock "identities" in the training panel, the cross-section degenerates faster than it would with 1000+ distinct stocks.

## Open questions

1. **Does α-mixing prevent recursive collapse?** Run the recursive loop with each gen trained on (1-α)·real + α·prev-synth, α=0.5. If pair_corr stays near real, we've confirmed α-anchoring is the fix.
2. **Why does kurt not collapse?** In classical theory it would. Possibly because factor conditioning supplies enough effective signal that the marginals don't need to be learned from scratch each gen.
3. **Where does leverage saturate?** -0.029 is very close to M8's max-expressible asymmetry. Is this a hard architectural ceiling, or a soft one set by training dynamics?
4. **Is there a "runaway" at gen > 3?** By gen 3 pair_corr is still growing (+10% per gen). Does it plateau at some level or diverge?

## Artifacts

```
experiments/phase2_interdiff_fts/
├── recursive_train.py                        # driver (3 gens, 8k steps each)
└── ckpts/
    ├── M0_rec_g1_step8000.pt                 # gen 1 (trained on M6 synth)
    ├── M0_rec_g1_step8000.samples.npz
    ├── M0_rec_g2_step8000.pt                 # gen 2 (trained on gen1 synth)
    ├── M0_rec_g2_step8000.samples.npz
    ├── M0_rec_g3_step8000.pt                 # gen 3 (trained on gen2 synth)
    ├── M0_rec_g3_step8000.samples.npz
    ├── recursive_collapse.json               # per-gen metrics dict
    └── recursive_collapse.log                # full training log
```

## Methodological notes

### Factor conditioning in recursive mode

When we train Gen(g) on Gen(g-1) synthetic, the `PanelWindowDataset` computes new market factor and per-stock sector factor FROM THE SYNTHETIC PANEL. So:
- Gen 1 market factor = mean across the 32 synth stocks in each window
- Sector labels for synthetic stocks are cyclically copied from real CSI800 sectors
- Gen 1's "market factor" is therefore an artifact of Gen 0's synthesis

This means Gen 1's conditioning distribution is a degraded echo of Gen 0's. Each subsequent gen narrows this further. If any specific factor shape (e.g. high-vol regimes) is under-represented in Gen 0's output, Gen 1 sees even less of it, and so on.

### Pseudo-panel construction fidelity

The synth→pseudo-panel reconstruction is exact: we set close[t+1] = close[t] * exp(log_ret[t]) and back-compute open/high/low from the original log_hc/log_lc/log_oc channels. When PanelWindowDataset derives log returns from this pseudo-panel, it reproduces the original 4 channels byte-identically. Zero information loss in the round-trip.

### 32 pseudo-stocks is a bottleneck

Real CSI800 has 1324 stocks; the pseudo-panel has only 32. This limits:
- Cross-section diversity per window (k_stocks=32 panels cover 100% of available stocks)
- Regime diversity across time (2000 panels × 64 days = 128k pseudo-days, but all with the same 32 cyclical "stock identities")

This could itself cause collapse-like behavior (not true model collapse, just low-data overfitting). A more powerful test would keep the pseudo-panel as wide as the real one by interleaving multiple generations' samples, but then the boundary between "gen g" and "gen g-1" blurs. Current setup is the cleanest definition of single-source-per-gen.

## Artifacts

```
experiments/phase2_interdiff_fts/
├── recursive_train.py                        # driver
└── ckpts/
    ├── M0_rec_g1_step8000.pt
    ├── M0_rec_g1_step8000.samples.npz
    ├── M0_rec_g2_step8000.pt
    ├── M0_rec_g2_step8000.samples.npz
    ├── M0_rec_g3_step8000.pt
    ├── M0_rec_g3_step8000.samples.npz
    ├── recursive_collapse.json               # per-gen metrics
    └── recursive_collapse.log                # full training log
```
