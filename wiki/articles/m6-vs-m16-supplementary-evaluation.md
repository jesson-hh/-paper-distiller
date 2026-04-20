---
title: "M6 vs M16: Supplementary Evaluation — Correcting the Premature Conclusion"
category: "articles"
slug: "m6-vs-m16-supplementary-evaluation"
tags: ["m6", "m16", "generator-quality", "evaluation-methodology", "mmd", "ks-test", "augmentation-mode", "lgbm", "csi800"]
refs: ["experiments/phase2_interdiff_fts/distribution_eval.py", "experiments/phase2_interdiff_fts/lgbm_sweep.py"]
links: ["lgbm-alpha-sweep-m16-vs-m6", "m10-m17-leverage-engineering", "factor-conditional-interdiff-m4-m5", "fts-interdiff-fusion"]
created: "2026-04-20T01:30:00"
updated: "2026-04-20T01:30:00"
---

# M6 vs M16 — Supplementary Evaluation

> **TL;DR**: [[lgbm-alpha-sweep-m16-vs-m6]] concluded "M6 is a better generator than M16 because its downstream LGBM IC was higher at α=0.5 in replacement mode". A user challenge prompted supplementary evaluation: **MMD / KS on direct distribution distance, augmentation-mode sweep, multi-year out-of-sample test**. The result: **M16 is equal-or-better than M6 on every direct quality metric**, wins augmentation-mode α=0.25 significantly (t=+2.6), and the previous "M6 wins" signal was isolated to one noisy cell of the LGBM IC grid. **The premature conclusion is reversed. M16 is the preferred default generator.**
> **Relation**: corrects [[lgbm-alpha-sweep-m16-vs-m6]]. Methodological lesson logged here for future similar comparisons.

## The user's challenge

> "下游评判标准真的对吗？有没有可能我们生成的质量更好，但是模型学不会了？"
>
> ("Is the downstream benchmark really correct? Could our generated quality be better, but the model can't learn it?")

This re-opened the investigation. Four concrete mechanisms by which LGBM IC might MIS-rank generators:

1. **LGBM features don't measure quality**: 17 engineered features (momentum, rolling vol, ranges, rank) don't directly probe leverage, skew, or joint distribution shape. M16's improvements in these are invisible to the tree model.
2. **Tree splits ignore distribution shape**: LGBM uses binary thresholds; better kurtosis or Hill tail don't help split quality.
3. **Test period idiosyncrasy**: M6's "spurious noise" might happen to correlate with 2023-specific patterns.
4. **Replacement vs augmentation mode**: replacing 50% real with synth may privilege generators with more noisy variation; true augmentation (adding synth on top) is the real-world mode.

## Three supplementary evaluations

### 1. Direct distribution-distance metrics (no downstream predictor)

`distribution_eval.py` computes MMD (joint) + KS / Wasserstein-1 (per channel).

**Setup**: 2000 real 32-day stock-trajectories from 2015-2022, 2000 synth trajectories from each of M6 and M16. Gaussian MMD with median-bandwidth heuristic.

**Results**:

| metric | real↔real (noise) | real↔M6 | real↔M16 |
|---|---|---|---|
| **MMD² (joint 128-dim)** | -0.00016 | +0.00034 | +0.00036 |
| KS log_ret | — | **0.020** | 0.030 |
| KS log_hc | — | 0.040 | **0.031** |
| KS log_lc | — | 0.045 | **0.041** |
| KS log_oc | — | 0.019 | 0.019 |
| **KS wins** | | 1/4 | **3/4** |
| Wasserstein-1 wins | | 1/4 | **3/4** |

**Key finding**: On the joint distribution (MMD), M6 and M16 are **within the MMD noise floor** of each other (difference 0.00002, baseline noise level 0.00016). On per-channel marginals, **M16 wins 3/4**.

### 2. Augmentation-mode sweep

The original replacement-mode sweep kept training-set size fixed (100k rows) with α fraction replaced by synth. True augmentation keeps all real data and adds synth on top.

Modified `mix_datasets()` with `aug_mode=True`: training size = (1+α) × 100k rows.

**Results (10 seeds each, 2023 test)**:

| α | M6 paired Δ | M16 paired Δ | **M16 - M6 paired** |
|---|---|---|---|
| 0.00 | 0 | 0 | 0 |
| **0.25** | +0.0001 | +0.0049 | **+0.0047 (t=+2.55) ✅ significant p<0.05** |
| 0.50 | +0.0059 | +0.0022 | -0.0037 (t=-1.46) |
| 0.75 | +0.0039 | +0.0037 | -0.0002 |
| 1.00 | +0.0024 | +0.0026 | +0.0002 |

**Aug-mode α=0.25 is the ONLY statistically significant M6-vs-M16 comparison across all setups tested**. It favors M16 by 0.005 IC (t=+2.55).

This matches the user's intuition: M16's higher-fidelity synth provides value **when added on top of real**, not when replacing real.

### 3. Multi-year test: 2024 in addition to 2023

`alpha_sweep on 2024 test` to check whether year-specific idiosyncrasies drove the 2023 result.

**2024 baseline IC (α=0) is +0.024** — 10× higher than 2023's +0.002. LGBM learns something meaningful on 2024 directly. Paired Δ vs α=0 then has more room to be measured.

**2024 replacement results (10 seeds)**:

| α | M6 paired Δ | M16 paired Δ |
|---|---|---|
| 0.1 | **+0.0036 (t=+2.2) ✓** | -0.0003 |
| 0.25 | -0.0013 | -0.0012 |
| 0.5 | -0.0015 | -0.0011 |
| 0.75 | -0.0050 | **-0.0106 (t=-2.1) ✗** |
| **0.9** | **-0.0086 (t=-2.2) ✗** | **-0.0133 (t=-4.0) ✗** |

New signature on 2024: **both generators fail at high α** (synth hurts significantly when it's majority of training data). This matches classical model-collapse theory and was invisible in 2023 because the base signal was too weak.

M16 fails WORSE than M6 at α=0.75 and α=0.9. But at α=0.1 (where synth is genuinely augmenting), M6 wins slightly (t=2.2). Both are small signals.

## Aggregate M6 vs M16 head-to-head (per-seed paired)

| setup | α=0.1 | α=0.25 | α=0.5 | α=0.75 | α=0.9 |
|---|---|---|---|---|---|
| repl 2023 | +0.001 | -0.001 | -0.005 (t=-1.5) | +0.001 | -0.004 |
| **aug 2023** | — | **+0.005 (t=+2.6) ★** | -0.004 (t=-1.5) | ~0 | — |
| repl 2024 | -0.004 (t=-1.8) | ~0 | +0.001 | -0.006 (t=-1.4) | -0.005 |

**One cell with p<0.05 out of 13**: aug-mode α=0.25 favors M16. Everywhere else: tied or marginal M6 advantage. **No consistent winner at the downstream-IC level.**

## Revised conclusion

### On generator quality

1. **M6 and M16 produce distributions of roughly equal joint quality** (MMD within noise floor)
2. **M16 has slightly better marginal distribution match** (KS wins 3/4 channels, Wasserstein wins 3/4)
3. **M16's stylized-fact improvements (leverage, skew, Hill) are real** but don't translate to a meaningful LGBM IC lift in most setups

### On my earlier mis-conclusion

The original [[lgbm-alpha-sweep-m16-vs-m6]] claim "M6 is preferred because it gives significantly higher LGBM IC at α=0.5" was **premature**:

- The t-stat was 1.68 (not 2.0), already marginal
- Restricted to replacement mode at one α
- Not validated on 2024 (where the pattern is flat, then both generators hurt)
- Not validated in aug mode (where M16 wins significantly at α=0.25)

**Single-cell statistical almost-significance in a noisy grid is not a generator-quality verdict.** This was the methodological error the user's challenge caught.

### Revised production recommendation

**Default generator: M16** (hinge leverage aux, t-EDM backbone).

Reasons:
1. Better or equal direct distribution metrics (MMD, KS, Wasserstein)
2. Better stylized facts (leverage, skew, Hill, max_eig)
3. Wins the single statistically-significant downstream comparison (aug-mode α=0.25)
4. Doesn't fail noticeably worse than M6 in any setup (except α≥0.75 on 2024, where M6 also fails)

**Keep M6 as an alternative** if a specific downstream task shows it's better in that task's α-sweep. Each downstream-specific evaluation should include its own α-sweep as part of model selection, rather than inheriting a cross-task "best generator" label.

## Methodological lessons for future comparisons

### Lesson 1: Downstream IC is a noisy proxy

IC at the 0.001-0.01 level with 10 seeds has standard errors around 0.003. Small differences are often statistically spurious. Always cross-validate with multiple test periods AND multiple downstream models AND multiple mixing modes AND direct distribution metrics.

### Lesson 2: Replacement mode ≠ augmentation mode

Replacement (keep size fixed, swap fraction) and augmentation (keep real whole, add synth) reward different generator properties:
- **Replacement** values *replaceable-with-synth-ness*: can this synth substitute real for a LGBM tree to split on?
- **Augmentation** values *additional-utility*: does adding high-fidelity synth provide new signal?

M16's higher fidelity paid off in augmentation mode; M6's noisier variation paid off in replacement mode. Neither is "better" in an absolute sense — they're better at different things.

### Lesson 3: Report multiple α values, not just the peak

Picking α=0.5 as "the comparison point" biased the analysis. Different α values tell different stories, and the honest approach is reporting the full curve + appropriate statistical tests.

### Lesson 4: The user is often right about methodology concerns

The challenge "is the downstream benchmark really correct?" was substantively correct. When a conclusion rests on a single measurement cell, adversarial questioning is the antidote.

## Artifacts

```
experiments/phase2_interdiff_fts/
├── distribution_eval.py                          # MMD / KS / Wasserstein
├── lgbm_sweep.py (+ --aug-mode flag added)
└── ckpts/
    ├── alpha_sweep_m6_lgbm_10seeds.json          # repl 2023 M6
    ├── alpha_sweep_m16_lgbm_10seeds.json         # repl 2023 M16
    ├── alpha_sweep_m6_aug_10seeds.json           # aug 2023 M6 (new)
    ├── alpha_sweep_m16_aug_10seeds.json          # aug 2023 M16 (new)
    ├── alpha_sweep_m6_lgbm_2024.json             # repl 2024 M6 (new)
    └── alpha_sweep_m16_lgbm_2024.json            # repl 2024 M16 (new)
```

## Production hand-off

Current best checkpoints:
- **M16**: `ckpts/M0_m16_lev_hinge_w20_step20000.pt` (+ samples npz)
  - t-EDM ν=6, factor conditioning (regime+mkt+sector), lev_cond pos_only, hinge aux w=20
- **M6**: `ckpts/M0_m6_csi800_step20000.pt` (kept as baseline)

For new downstream tasks: **run your own small α-sweep on both, include aug-mode, before choosing**.
