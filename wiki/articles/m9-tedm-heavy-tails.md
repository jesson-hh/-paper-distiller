---
title: "M9 — Student-t EDM for Heavy Tails (Pandey 2024 + Karras 2022)"
category: "articles"
slug: "m9-tedm-heavy-tails"
tags: ["diffusion", "edm", "student-t", "heavy-tails", "kurtosis", "karras", "pandey", "csi800", "m9"]
refs: ["arxiv:2410.14171", "arxiv:2206.00364", "experiments/phase2_interdiff_fts/edm_diffusion.py"]
links: ["factor-conditional-interdiff-m4-m5", "m7-m8-modernization-and-leverage", "a-share-positive-leverage", "fts-interdiff-fusion"]
created: "2026-04-18T02:30:00"
updated: "2026-04-18T02:30:00"
---

# M9 — Student-t EDM

> **TL;DR**: Replaced Gaussian DDPM with **Student-t EDM (t-EDM)** from Pandey et al. 2024, combined with Karras 2022 preconditioning and Heun 2nd-order sampler. Kept factor conditioning (market + sector) from M4-M6 intact. **Kurtosis 3.03 → 3.14, Hill indices all closer to real, std more accurate — tails improved as targeted**. Side effect: leverage gap deepens from -0.007 → -0.012 due to symmetric heavy-tailed noise amplifying both directions equally. 7/7 verdict still fully green.

## Motivation

After M4-M8, all 7 verdicts green but 3 stylized facts persistently slightly off target:

| metric | real | M6 (Gaussian DDPM) | gap |
|---|---|---|---|
| excess_kurt | 3.55 | 3.03 | **-15%** |
| hill_right | 3.46 | 3.85 | +11% (tails too thin) |
| hill_left | 3.00 | 3.16 | +5% |

All three point to **tail-coverage loss**: Gaussian noise is mathematically lighter-tailed than financial return distributions. The denoiser produces samples whose extreme events are less extreme than real data's.

NVIDIA's EDM (Karras 2022) provides a cleaner diffusion framework, and Pandey et al. 2024 extended it to **t-EDM** — Student-t noise prior with degrees-of-freedom ν controlling tail thickness. For ν > 4, excess kurtosis of Student-t is 6 / (ν - 4): ν=5 → kurt 6.0, ν=6 → kurt 3.0, ν=8 → kurt 1.5.

We targeted ν=6 to approximately match real's 3.55.

## Implementation

`edm_diffusion.py` implements `StudentTEDM` with the same public API as `GaussianDiffusion` (training_loss, sample), so train.py / sample.py only need minimal wiring.

### Key formulas

**Forward noising**: $q(x_t \mid x_0) = t_d(x_0, \sigma_t^2 I_d, \nu)$ with $\sigma_t = t$ (σ is the time variable, not discrete index).

**EDM preconditioning** (same as Gaussian EDM):
$$c_{\text{skip}} = \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2},\quad
c_{\text{out}} = \frac{\sigma\, \sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}},\quad
c_{\text{in}} = \frac{1}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}$$

Denoiser: $D_\theta(x, \sigma) = c_{\text{skip}}\, x + c_{\text{out}}\, F_\theta(c_{\text{in}}\, x, c_{\text{noise}}(\sigma))$

**Loss**: $\mathcal{L} = \mathbb{E}_{\sigma \sim \text{LogN}(P_m, P_s)}\, \mathbb{E}_{n \sim t_d(0, \sigma^2 I, \nu)}\, \|F_\theta(\cdot) - (x_0 - c_{\text{skip}}\, x_t)/c_{\text{out}}\|^2$

**Sampler**: rho-schedule + Heun 2nd-order (from Karras 2022), unchanged for t-EDM.

### Student-t noise sampling

```python
y = torch.randn(shape)                     # Gaussian
chi2 = Chi2(nu).sample((B,))              # per-sample chi-square
noise = y / torch.sqrt(chi2 / nu)         # multivariate t_d(0, I, nu)
```

Per-sample chi² (shared across all dims of that sample) matches multivariate $t_d(0, I_d, \nu)$ definition — not element-wise chi².

### Time-embedding trick

Our InterDenoiser has a sinusoidal embedding designed for discrete $t \in [0, 500]$. EDM's $c_{\text{noise}}(\sigma) = 0.25 \log \sigma$ gives a very compact range $[-1.55, 1.10]$ — too small for the default sinusoidal frequencies to differentiate. Fix: scale by **TIME_SCALE = 250** before feeding, which spreads the log-sigma range across the sinusoidal bands. No model-architecture changes needed.

## Training

Config identical to M7 except diffusion engine:
- CSI800, 1324 stocks, length=64, k=32, batch=16, 20000 steps, bf16
- Factor conditioning: regime + market + sector (from M4-M6)
- EDM: ν=6, σ_min=0.002, σ_max=80, σ_data=1.0, ρ=7, P_mean=-1.2, P_std=1.2

31.6 step/s on RTX 5090 (matches M7), peak 1.46 GB. Total 20k steps ~10.5 min.

**Note**: EDM loss and DDPM loss are **not directly comparable** (different scale — F_θ space vs ε space). M9 ema=0.53 vs M7 ema=0.13 doesn't mean anything about relative quality.

## Results

### Sample generation
Heun 18 steps on 50×8 = 400 panels. Sampling time per panel: ~50 ms (vs DDIM 50 steps at ~265 ms for batch-8). **5× faster than DDIM, 50× faster than full DDPM**.

### Stylized-fact verdict (vs real CSI800)

| metric | real | M6 DDPM | **M9 t-EDM** | M9 vs M6 |
|---|---|---|---|---|
| std | 0.0274 | 0.0283 | **0.0276** | ⬆ more accurate |
| **excess_kurt** | **3.55** | 3.03 | **3.14** | ✅ **+0.11 toward real** |
| **hill_right** | **3.46** | 3.85 | **3.75** | ✅ **-0.10 toward real** |
| **hill_left** | **3.00** | 3.16 | **3.09** | ✅ **-0.07 toward real** |
| skew | +0.11 | -0.10 | -0.14 | — slight |
| acf_r² lag1 | 0.074 | 0.066 | 0.060 | — slight |
| acf_r² lag5 | 0.010 | 0.016 | 0.018 | — slight |
| **acf_r² lag10** | **-0.005** | -0.001 | **-0.003** | ⬆ closer |
| acf_r² lag20 | -0.025 | -0.019 | -0.019 | — tie |
| **leverage_lag1** | +0.013 | -0.007 | **-0.012** | ❌ **doubles in wrong direction** |
| pair_corr | 0.349 | 0.347 | 0.367 | — slight over |
| max_eig_frac | 0.398 | 0.394 | 0.408 | — slight over |
| **Verdict total** | — | 7/7 | **7/7** | unchanged |

### Core findings

**Primary target (tails) improved as hypothesized**:
- Kurtosis gap -15% → -12%
- Both Hill indices moved measurably toward real
- std became more accurate

**Unintended side effect on leverage**:
- Leverage_lag1 deepened from -0.007 (M6) to -0.012 (M9) — 2× worse
- Mechanism: Student-t noise is **symmetric heavy-tailed**, so both large positive and large negative extreme values are more common. For already-wrong-sign leverage (see [[a-share-positive-leverage]]), symmetric amplification worsens both tails' contribution to the correlation in the wrong direction.
- Consistent with M8's finding: symmetric architectural changes can't fix directional asymmetry.

**Cross-section slight inflation**:
- pair_corr +5% above real (was +0% in M6), max_eig_frac +2.5%
- Possibly Heun sampler artifact (recursive-collapse experiments also showed common-mode sensitivity to sampler changes)

## The ν tradeoff

t-EDM exposes a single-parameter tradeoff: **tail thickness vs leverage stability**.

| ν | expected Student-t kurt | guess for leverage effect |
|---|---|---|
| 4 | ∞ (undefined) | catastrophic |
| **5** | 6.0 | worst leverage |
| 6 | 3.0 | **M9's choice** |
| 7 | 2.0 | less tail gain |
| 8 | 1.5 | approaches Gaussian |
| ∞ | 0 | = DDPM (our M6) |

Future experiments could sweep ν ∈ {5, 6, 7, 8} to locate where kurt hits real ~3.55 without leverage blowing up. Since real leverage is already positive and ours is negative regardless of noise shape, no ν value in this family solves leverage; only asymmetric interventions (not attempted here) would.

## Combining with other upgrades

M9 stacks on top of M4-M6 factor conditioning unchanged. The recipe "factor-conditional InterDiff + t-EDM noise" is the current best model:

- **Cross-section**: solved by factor conditioning (M4-M6), kept
- **Tails**: improved by Student-t noise (M9)
- **Leverage sign**: still wrong (open problem; [[a-share-positive-leverage]])
- **Sampling speed**: 5× faster than DDIM, 50× faster than DDPM via Heun 18-step

The M8 sign-aware conditioning did not help; leave it off. The M7 CFG helped slightly at g=1.5 and could be tried with M9 if needed.

## When to prefer M9 over M6

- **Use M9** if downstream task is sensitive to tail coverage (risk models, VaR, stress testing)
- **Use M6** if downstream is sensitive to leverage sign or needs pair_corr exact
- **α-sweep on M9** would tell us if the better tails translate to better downstream IC — not yet run

## Artifacts

```
experiments/phase2_interdiff_fts/
├── edm_diffusion.py                      # StudentTEDM class
├── train.py                              # +--edm --edm-nu ... flags
├── sample.py                             # auto-detects EDM, uses Heun
└── ckpts/
    ├── M0_m9_tedm_nu6_step20000.pt
    ├── M0_m9_tedm_nu6_step20000.samples.npz
    └── m9_tedm_nu6_train.log
```

## Next

1. **ν-sweep** {5, 7, 8} to map kurt vs leverage tradeoff curve
2. **M9 α-sweep** — does M9's better tails help downstream LGBM IC beyond M6's?
3. **Revisit leverage**: Student-t didn't solve the asymmetry. See [[a-share-positive-leverage]] for candidate fixes (aux loss, GJR-GARCH 2-stage)
4. **Recursive collapse on M9** — does heavier-tailed noise prevent or accelerate common-mode amplification?

## References

- Karras, T., Aittala, M., Aila, T., Laine, S. (2022). "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS. [arxiv:2206.00364](https://arxiv.org/abs/2206.00364)
- Pandey, K., Rudner, T., et al. (2024). "Heavy-Tailed Diffusion Models", ICLR. [arxiv:2410.14171](https://arxiv.org/abs/2410.14171)
