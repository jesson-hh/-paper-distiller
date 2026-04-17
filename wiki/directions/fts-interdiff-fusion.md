---
title: "FTS-Diffusion × InterDiff Fusion: Pattern × Cross-section"
category: "directions"
slug: "fts-interdiff-fusion"
tags: ["finance", "diffusion", "multi-stock", "pattern", "fusion", "applied"]
refs: ["arxiv:2403.13638", "arxiv:2406.16064"]
links: ["synthetic-augmentation-financial-timeseries", "fts-diffusion-iclr-2024", "interdiff-inter-stock-correlations", "factor-conditional-interdiff-m4-m5", "factor-conditional-denoising", "m7-m8-modernization-and-leverage", "a-share-positive-leverage", "alpha-sweep-csi800-m6", "lgbm-alpha-sweep-phase-transition", "recursive-collapse-csi800", "wasserstein-generative-regression-framework", "wasserstein-generative-regression", "huang-jian"]
created: "2026-04-14T11:00:00"
updated: "2026-04-17T14:15:00"
---

# FTS-Diffusion × InterDiff Fusion

> **Status**: design draft (2026-04-14)
> **Owner**: me
> **Parent**: [[synthetic-augmentation-financial-timeseries]]

## 一句话

把 [[fts-diffusion-iclr-2024|FTS-Diffusion]] 的**时间形态分解**（pattern + scaling + Markov 转移）和 [[interdiff-inter-stock-correlations|InterDiff]] 的**横截面层级 transformer** 合到一个生成器里 —— 它们处理的是**正交的两个轴**，组合不冲突。

## 动机：两个轴是正交的

| 轴 | 由谁负责 | 现有方法 |
|---|---|---|
| **时间形态** | 单股序列的 morphology（pattern + scaling 律 + 状态转移） | FTS-Diffusion |
| **横截面** | 同一时刻 N 只股票之间的相关结构 | InterDiff |
| **组合** | "整个 panel 在每天每股的 pattern 上一致 + 数值上横截面相关" | **本方向** |

**单线缺陷**：
- FTS-Diffusion 是单股的 → 跨股相关靠后处理 copula，丢横截面信息
- InterDiff 是单尺度的 → 数值层面建模 cross-section，但没有显式时间状态机，长程 regime 切换靠 transformer 隐式学
- CoFinDiff 是单股 + trend 条件 → 接近但条件粒度太粗（trend label 只 4 类）

## 核心问题

> 给定全市场 daily snapshot 序列 $\{X_t \in \mathbb R^{N\times F}\}_{t=1}^T$，能不能学到一个生成器 $p_\theta$ 使得：
> 1. 每只股票的边际**保留 FTS pattern + scaling 律**
> 2. 同一时刻横截面**相关矩阵**和真实 panel 一致
> 3. **regime 切换**（牛 → 熊、震荡 → 趋势）是一致的"市场级"事件，而不是每股独立掷骰子

## 三种可行的耦合姿势

### 方案 A. Pattern-conditional InterDiff（推荐起步）

**思路**：FTS 离线挖 pattern → InterDiff 在生成时**条件化**于全市场 pattern 矩阵。

```
1. 离线：在所有股票合并后的 segment 集上跑 SISC，得到一个共享 pattern dict K（k=8~32）
2. 离线：每天每股打 pattern label → 得到 [N_stocks × T] 的 label 矩阵
3. 训 InterDiff 的 denoiser 时，加一个 pattern embedding 条件：
     - 每个 token 的 condition = pattern embed + position + stock id
     - cross-attention 注入（沿用 CoFinDiff 的 trick）
4. 生成时：先在 pattern 空间跑一个 markov / autoregressive sampler
     - 横截面 markov：transition 不是 per-stock，而是 [N×K] → [N×K] 的联合
     - 退化版：共享一个 market-state，每股 pattern 在 market-state 下条件独立
5. 给定 pattern 矩阵，InterDiff 一次 denoise 出整个数值 panel
```

**关键设计点**：
- **pattern 字典必须跨股共享**（不能每股一套）—— 否则 label 没有横截面意义
- pattern 数量 K 不能太大（K=8~16 起步），否则 transition 矩阵稀疏
- 横截面 markov 是这条路最难的部分，先用"共享 market-state + 条件独立"退化版起步

**优点**：
- 增量改造，InterDiff 的多头注意力直接复用
- pattern 只是额外条件，可以 ablation 验证贡献
- pattern label 是离散的，调试容易

**缺点**：
- pattern 字典质量决定上限
- 横截面 markov 的设计空间大

### 方案 B. 外层 FTS + 内层 InterDiff（两段式）

```
1. FTS 单股先生成每股的 pattern 序列（保留时间形态、scaling）
2. 给定全市场的 pattern 矩阵 [N × T]，InterDiff 一次性把数值 panel denoise 出来
```

**优点**：分工最清晰
**缺点**：两段误差累积，pattern 层不知道横截面冲击 → 容易出现"pattern 一致但数值矛盾"

### 方案 C. 横截面 SISC（最激进）

```
不在单股上聚类，直接在全市场 daily snapshot 上聚 regime（牛/熊/震荡/反转 …）
InterDiff 条件在 regime 上
```

**优点**：规避 pattern 字典对齐问题
**缺点**：这其实是 CoFinDiff 的 trend 条件的一个超集，**已经被 CoFinDiff 占了**，新意不够

## 风险评估

| 风险 | 严重度 | 缓解 |
|---|---|---|
| 两边都没代码 | 🔴 | 先单跑 InterDiff 拿 baseline，再增量加 FTS pattern condition |
| 跨股 SISC 字典退化 | 🟡 | K 控制在 8~16，先做 sanity check（pattern 是否在不同行业上有差别）|
| 横截面 markov 设计复杂 | 🟡 | 先用"共享 market-state + 条件独立"退化版 |
| 训练计算量 = N_stocks × ... | 🟡 | 用 CSI300（300 股）起步，跑通后再扩到 CSI800 |
| Compute = InterDiff × pattern condition | 🟢 | pattern embed 只是一个额外的 token，开销小 |

## 增量路径（从 InterDiff baseline 出发）

```
M0. 直接跑 InterDiff（重实现）on CSI300，10 年日线 → 拿 IC 基线
M1. 加 SISC 离线 pattern 标注（不改 InterDiff），算每股 pattern 序列的统计特性
        ↓ 验证 pattern 字典在跨股上有意义
M2. InterDiff denoiser 加 pattern embed cross-attention（条件版）
        ↓ Ablation：vs M0 在 stylized facts / IC 上的差
M3. 加横截面 markov sampler（退化版：共享 market-state + 条件独立）
        ↓ Ablation：vs M2 在长程 regime 切换上是否更真实
M4. 全联合：joint pattern + value denoise
        ↓ 终态
```

每个 milestone 都有 baseline 对比 → 任何一步如果没收益就停在上一步。

## 评估指标

继承 [[synthetic-augmentation-financial-timeseries]] 的指标 + 几个新增：

- **Stylized facts**（每股）：Hill index, ACF of $r^2$, leverage effect — 验证 pattern 不破坏单股 morphology
- **Cross-sectional**：correlational score（Frobenius distance of corr matrix）— 验证横截面没退化
- **Pattern fidelity**（新）：pattern 转移矩阵的 KL(real || synth)
- **Regime coherence**（新）：同一日所有股票的 pattern 是否聚集在合理的市场状态（用 entropy 度量）
- **下游 IC**：合成数据 + 真实数据混合训练 ranking 模型，比 real-only baseline

## 数据

- Qlib 格式的 A 股 daily：`G:\stocks\stock_data\cn_data`
- 股票池：`G:\stocks\stock_data\cn_data\instruments\csi300.txt` / `csi500.txt` / `csi800.txt` / `csi1000.txt`
- Features 全：OHLCV + adj + amount + factor + 多种 mom/rev + macd/rsi + market cap + turnover

第一版选 **CSI300**（300 股，规模可控），10 年（2015-01-01 → 2024-12-31）。

## 和其它三条线的关系

| 线 | 单股形态 | 横截面 | 代码 | 状态 |
|---|---|---|---|---|
| FTS-Diffusion 主线 | ✅ | ❌ | ❌ | scaffold 已搭 |
| WaveletDiff fork | 🟡 多尺度 | ❌ | ✅ | scaffold 已搭 |
| InterDiff 主线 | ❌ | ✅ | ❌ | 候选 |
| **本方向（FTS × InterDiff）** | **✅** | **✅** | **❌** | **新方向** |

**定位**：作为 Phase 2 的"理想终态" —— Phase 1 三条线先各自跑通，再决定要不要走 fusion。

## 局限提醒

- **没现成代码可参考** —— 是从设计到实现都要自己做的方向
- **联合 markov 设计是开放问题**，可能需要 1-2 周纯研究
- **pattern 字典跨股一致性** 在中国 A 股上是经验问题，要先做 sanity check
- 评估上 fusion 是否真的比单独的 InterDiff 好，**没人做过对照**，结果未知

## 下一步

1. M0：先把 InterDiff 单独在 CSI300 跑通，建立 baseline
2. M1：离线做 SISC 跨股聚类，验证 pattern 字典质量
3. 决定是直接 M2（pattern condition）还是先把其他 Phase 1 线对比清楚

## Progress Log

### 2026-04-14 — M0 ✅ + M1 ✅

代码位置：`experiments/phase2_interdiff_fts/`(独立实验目录,数据从 `G:\stocks\stock_data` 只读拷一份到 `data/csi300_2015_2024.npz`)。

#### 基础设施

- `qlib_reader.py`、`universe.py`、`build_dataset.py`:Qlib bin 读取 → CSI300 panel(594 支 × 2431 天 × 7 字段,drop 54 支低覆盖,92.6% 有效)
- `panel_windows.py`:IterableDataset,每步采 (k_stocks, length, C=4) 窗口,通道 = {log_ret, log_hc, log_lc, log_oc},per-stock z-score
- `model.py`:InterDenoiser —— 逐层 intra-stock L-attn + inter-stock N-attn + FF,正弦 t-embed + 可学习 s_pos/t_pos
- `diffusion.py`:DDPM 线性 β∈[1e-4, 0.02],predict-x0 采样 + x0 clip ±5(**注意**:cosine + 小 T 会被 β.clamp(0.999) 撑爆,已踩过)
- `stylized_facts.py` + `eval_compare.py`:Hill / ACF r² / leverage / panel cross-corr
- `train.py`:GPU/CPU RAM 双守护(preflight peak check + 周期 RSS 检查 + sys avail 预留)

#### 方法学关键点(踩过的坑)

1. **长 vs 短序列的 ACF 不对称**:real 上用 (594, 2430) 算 ACF(r²) 得 0.275,但 syn 是 64 步窗口,短窗口 ACF 天然低估。**正确对比方式**:从 real 里 bootstrap 出同形状 (n_panels, k, L) 窗口,在窗口空间算 ACF。
2. **panel 相关的 lumping 陷阱**:不能把所有 panel 的股票扁平成一个 (N_total, L) 矩阵算相关——那是把独立时间轴 lumping 到一起。正确做法是每个 panel 内算 corr,再 panel 间平均。
3. **cosine schedule @ T=200** β 被 clamp 到 0.999 → 1/sqrt(α)=31 → 采样爆炸。切线性 + predict-x0 后稳住。

#### M0 结果(unconditional InterDiff baseline)

配置:d_model=128, 6 blocks, 8 heads, length=64, k=32, batch=16, T=500 linear, 20k steps, lr=2e-4 cosine decay。1.73M params, ~16 step/s on RTX 5090, peak GPU 2.4 GB, 20 min wall。

**公平对比(64 步 bootstrap real baseline)**:

| metric | real(L=64) | M0_big | verdict |
|---|---|---|---|
| std | 0.0261 | 0.0242 | OK |
| excess_kurt | 3.67 | 2.85 | OK |
| hill_right / left | 3.43 / 2.90 | 3.56 / 3.46 | OK |
| acf_r² lag1 | 0.0620 | 0.0406 | MEH |
| acf_r² lag5 | 0.0078 | 0.0007 | — |
| leverage lag1 | 0.0173 | 0.0120 | — |
| panel_mean_pair_corr | 0.332 | 0.281 | MEH |
| panel_max_eig_frac | 0.381 | 0.328 | MEH |

诊断:边际分布已经抓得不错,胖尾 Hill 几乎完全复现;弱一点的是**窗口级 vol clustering 和 cross-section 结构**。

**容量无法解决 vol clustering**:先前在 ~680k params / 8k steps 的小 M0 上跑过,acf_r² lag1 同量级。放大 4× params + 2.5× steps 到 1.73M / 20k step 后,lag1 仅从 0.04 移到 0.04,**结构性缺陷而非容量缺陷**。这就是做 M1 的动机。

#### M1 结果(pattern-conditional InterDiff)

新增模块 `regimes.py`:
- 对归一化后的 log_ret 算 rolling realised vol `sqrt(rolling_mean(r², W=10))`
- 取 log 后按 8 等频分位桶成离散 regime label,逐 (stock, t) 一个 int
- 标签分布 **等频**(entropy=ln(8)=2.0794),分桶的 mean|r| 单调从 0.005→0.040(跨 8×)

注入方式:`InterDenoiser` 里加 `regime_embed = nn.Embedding(K, d_model)`,在 input token 上加 `self.regime_embed(cond)`。diffusion 训练/采样都 thread 一个 `cond: (B, N, L)` 可选参数。采样时从真实 panel 中 **borrow regime label 序列**(同 dataset 迭代器,保持顺序可复现)——本质是:"给定真实 regime prior,生成数值"。

训练同 M0_big 配置 + `--regime-window 10 --n-regimes 8`。+1024 params(一个 8×128 embedding)。ema 0.1934(M0:0.2099)。

**评估对比**:

| metric | real(L=64) | M0_big | **M1_big** | M1 verdict |
|---|---|---|---|---|
| std | 0.0261 | 0.0242 | **0.0260** | OK |
| acf_r² lag1 | 0.0620 | 0.0406 MEH | **0.0570** | **OK** ⬆ |
| acf_r² lag5 | 0.0078 | 0.0007 | **0.0165** | — |
| leverage_lag1 | 0.0173 | 0.0120 | 0.0106 | ~ |
| excess_kurt | 3.67 | 2.85 | 2.71 | OK,略退 |
| panel_mean_pair_corr | 0.332 | 0.281 | 0.272 | MEH |

**Conditioning 质量诊断**(`_diag_m1.py`):
- 每桶 mean |r| 的 syn/real 比值 = [1.00, 0.99, 0.99, 0.99, 0.99, 1.00, 0.99, 0.94],**近乎完美**
- 全局 per-window vol-envelope 相关(syn vs 真实借来的模板)= **0.975**
- regime 条件在数值层面彻底生效,M1 学会了"给定 regime 序列,生成匹配 magnitude 的返回"

#### M1 剩下的缺口

两个方向没被 regime 条件改善:
1. **Panel cross-sectional correlation gap**:0.272 vs 0.332(~18%)。这是 inter-stock attention 的职责,regime 条件管不到。想法:加 cross-section 正则 / 检查 inter-attention 容量 / β weighted loss 上偏向低 SNR 步骤。
2. **超额峰度轻微下降**:M1 2.71 vs M0 2.85 vs real 3.67。加 regime cond 反而让尾巴稍微瘦了一点点(每桶内部被强制更均匀)。改 K 或改连续 envelope 可能有帮助。

#### 下一步(M2 / M3)

- **M2 cross-section 强化**:先查 panel_mean_corr gap 根因(是 inter attention 层数不够?是 loss 对低 SNR 过于宽容?),尝试 cross-sectional auxiliary loss 或 inter-attention 容量加倍
- **M3 长序列**:length=128 / 256,让窗口内部真能表达 regime 切换,再测 acf_r² 是否贴近 real 的长序列值

#### M2 结果(cross-section 强化:市场因子辅助 loss)

根因诊断(`_diag_cs.py`):cross-section gap 100% 来自 **market factor(共同模式)**,不在残差结构。
- real(L=64):`mean_pair_corr=0.319`,`max_eig_frac=0.360`,`resid_mean_pair_corr=-0.026`,`market_factor_var=0.348`
- M1:`mean_pair_corr=0.272`,`max_eig_frac=0.315`,`resid_mean_pair_corr=-0.026`,`market_factor_var=0.272`
- 残差相关几乎相同,差的全是 market factor 的方差。

Loss 设计:`loss = loss_res + w * loss_mkt`,其中
```
mkt_true = noise.mean(dim=1, keepdim=True)
mkt_pred = eps.mean(dim=1, keepdim=True)
res_true = noise - mkt_true; res_pred = eps - mkt_pred
loss_mkt = MSE(mkt_pred, mkt_true); loss_res = MSE(res_pred, res_true)
```

**关键陷阱**:`loss_res + 1*loss_mkt ≡ base_MSE`(数值恒等,`_diag_aux_loss.py` 已数值验证)。所以 w=1 与 baseline 完全等价——首次 M2 run 采样 byte-identical,被这个坑了一次。需要 w >> 1 才起作用。

| run | w | mean_pair_corr | max_eig_frac | market_var | 对比 M1 |
|---|---|---|---|---|---|
| M1 | — | 0.272 | 0.315 | 0.272 | — |
| M2 (w=1) | 1 | 0.272 | 0.315 | 0.272 | 恒等,无效 |
| **M2b** (w=16) | 16 | **0.283** | **0.327** | **0.283** | **+4%** |
| M2c (w=64) | 64 | 0.270 | 0.311 | 0.270 | 退化 |

w=16 最优但只收回 ~20% 的 gap。继续加权过头后整体 loss landscape 被扭曲,sampling 质量反而下降。结论:**纯 loss 加权无法根治**,要么改 predict-x0(让信号在低 SNR 步仍显著),要么扩 inter-attention 容量(多 head / 多 block 专门给 N-轴)。这些标记为 M2.5 备选。

#### M3 结果(长序列 length=128)

同 M1 配置,只把 `--length 32`(实际 M1 训练用 64)换成 128,其余不动。20k 步训练,ema=0.193(与 M1 持平),GPU 峰值 4.74 GB(L 翻倍 → attention 内存翻倍)。

| metric | real(L=128) | M1(L=64) | **M3(L=128)** | verdict |
|---|---|---|---|---|
| std | 0.0255 | 0.0260 | 0.0259 | OK |
| excess_kurt | 3.77 | 2.71 | 2.70 | 持平 |
| hill_right | 3.36 | — | 3.84 | OK |
| acf_r² lag1 | 0.095 | 0.057 | 0.105 | **OK**(L 变长后 real 基线也抬高了) |
| acf_r² lag5 | 0.033 | 0.017 | 0.036 | **OK** |
| acf_r² lag10 | 0.024 | — | -0.005 | 差 |
| leverage_lag1 | 0.018 | 0.011 | 0.011 | ~ |
| mean_pair_corr | 0.328 | 0.272 | 0.276 | MEH |
| max_eig_frac | 0.370 | 0.315 | 0.315 | MEH |
| **market_factor_var** | **0.357** | 0.272 | **0.275** | **MEH(-23%)** |
| resid_mean_pair_corr | -0.028 | -0.026 | -0.029 | 持平 |

**结论**:更长的上下文**没有**让 inter-attention 学出更强的 common mode。market factor gap 从 M1 的 -22% 到 M3 的 -23%,几乎原地踏步。lag10 的 ACF 甚至跑到负值,说明长序列里中长期记忆没被捕捉到(模型在 64 步以内靠 regime cond 撑,超过就退化为 near-iid)。

**综合判断**(M0→M1→M2→M3):
- **Marginal / 尾巴**:M1 已基本解决。
- **短期 vol clustering**:M1 的 regime-cond 路径有效,M3 在长窗口下一样 OK。
- **中长期 ACF(lag≥10)**:缺;需要显式的时间长程先验(autoregressive head 或 Hawkes-like)。
- **Cross-section / market factor**:**结构性缺陷**——loss 加权只能拿回 ~20%,长窗口也不帮忙。需要架构层面的改动:或显式分解 `x = α·market + residual` 用两个支路训练,或将 inter-attention 扩成独立 factor head。这是 M4 的重点。

### 2026-04-17 — M4 ✅ + M5 ✅

详细结果见 [[factor-conditional-interdiff-m4-m5]]，方法学详情见 [[factor-conditional-denoising]]。

#### M4(market-factor 条件注入)

**核心发现**:M3 诊断出来的 -22% market_factor_var gap **不是训练不充分,是模型没看见这个信号**。修复方案:训练时显式把 per-window 的等权市场因子 $m_t = \bar r_t$ 作为条件喂进 denoiser;采样时从真实面板 bootstrap 一段 $m_t$ 序列作为生成的 guide。

架构改动极小:`InterDenoiser` 加一个 `mkt_proj: Linear(1, d_model)` 的两层 MLP,把 $(B, L)$ 映射到 $(B, 1, L, d)$ 然后**加法广播**到所有 N 个股 token。和 regime embedding 并存、和 sinusoidal time embed 并存,都是加法组合。

训练配置同 M1/M3(d_model=128, 6 blocks, 8 heads, length=64, k=32, batch=16, T=500, 20k steps),新增 params ~17k(占总 1.73M 的 1%)。RTX 5090 上 24.6 step/s, peak 2.39 GB,13 min wall,ema=**0.1901**(M1=0.1934)。

| metric | real(L=64) | M1 | **M4** | M4 gap |
|---|---|---|---|---|
| market_factor_var | 0.385 | 0.272(-22%) | **0.382** | **-0.7%** ⬆ |
| panel_mean_pair_corr | 0.339 | 0.272(-15%) | **0.337** | **-0.6%** ⬆ |
| panel_max_eig_frac | 0.388 | 0.317(-14%) | **0.378** | **-2.6%** ⬆ |
| excess_kurt | 3.81 | 2.71 | 3.01 | — |
| acf_r² lag1 | 0.061 | 0.057 | 0.067 | OK |
| hill_right / left | 3.41/2.91 | 3.87/3.30 | 3.82/3.03 | OK/更贴近 |
| leverage lag1 | 0.013 | 0.011 | -0.003 | ❌ 倒负 |

**结论**:cross-section 这条线从 15% gap 直接收到 <1%,**所有 7 项 verdict 全绿**。唯一没改善的是 leverage 非对称性(下跌-波动放大),这是方向性问题,不靠对称加法 conditioning 解决。

#### M5(+ 行业 sector 因子)

在 M4 基础上再加一条**per-stock** 的 sector 因子信号。步骤:
1. `tushare_stock_basic.parquet` → qlib 代码映射 → 110 个细分行业压成 **11 个大 sector**(FINANCE/TECH/MEDIA/HEALTHCARE/CONSUMER/INDUSTRIAL/MATERIALS/ENERGY/METALS/REAL_ESTATE/TRANSPORT)+ UNKNOWN(16 支未覆盖)。边表 `data/csi300_sectors.npz`。
2. 采样 k=32 股后,对每支股 i,算**除自己之外**同 sector 的等权平均(避免恒等映射),作为 stock-specific 的 sector 因子信号 $(k, L)$。若本窗内该 sector 只有自己,回退到 market factor。
3. `sector_proj: Linear(1, d_model)` 把 $(B, N, L)$ 映到 $(B, N, L, d)$ 加到 h。和 market_proj 并存。

训练同 M4 配置 + `--sectors-npz data/csi300_sectors.npz`,+4k params。23.3 step/s, peak 2.43 GB,ema=**0.1317**(比 M4 **低 31%**——sector 信号让 denoiser 预测噪声任务本身变简单)。

| metric | real(L=64) | M4 | **M5** | 说明 |
|---|---|---|---|---|
| excess_kurt | 3.81 | 3.01 | **3.30** | 更接近 |
| hill_right | 3.41 | 3.82 | **3.71** | 更接近 |
| hill_left | 2.91 | 3.03 | **3.00** | 更接近 |
| acf_r² lag1 | 0.061 | 0.067 | **0.057** | 更接近 |
| acf_r² lag5 | 0.007 | 0.017 | **0.014** | 更接近 |
| **acf_r² lag10** | **-0.001** | -0.034 | **-0.002** | **几乎完美** |
| panel_mean_pair_corr gap | — | -0.6% | **-0.5%** | 保持 |
| market_factor_var gap | — | -0.7% | -0.7% | 保持 |
| max_eig_frac gap | — | -2.6% | **-1.3%** | 改善 |
| leverage lag1 | 0.013 | -0.003 | -0.004 | 仍为负 |

**结论**:全绿 + 在 5 个细粒度指标上比 M4 更准。**lag10 的中长期 vol clustering 从 -0.034 收到 -0.002,之前标记为"结构性缺陷"的问题被意外解决了**——诊断是:sector 因子里天然带着比 regime(单股 rolling vol 分位)更长期的集体记忆(行业轮动、板块热度的持续性),denoiser 借到了这条信号。

**仍未解决**:leverage 非对称性。下一步见 [[factor-conditional-interdiff-m4-m5#next]]。

#### 综合判断(M0→M5)

- ✅ **Marginal / 尾巴**:M1 基本解决,M5 进一步贴近
- ✅ **短期 vol clustering**:M1 解决
- ✅ **中长期 ACF(lag≥10)**:M5 意外解决(之前判为结构性)
- ✅ **Cross-section / market factor**:M4 解决,M5 保持
- ❌ **Leverage 非对称性**:对称加法 conditioning 解不了,需要 sign-aware 机制或残差层 asymmetric noise

M5 是当前最好的模型。下一步:CSI800 扩规模 + α-sweep 下游验证。

### 2026-04-17 (续) — M6 CSI800 scaling ✅

CSI800(1324 股,过滤后) + 同 M5 配置 + sectors_npz。结果:**架构完美 scale**,ema=0.1325(vs M5 0.1317),同 GPU 占用,训练时间线性。Cross-section gap 在更大池子下**进一步缩小**(market_factor_var gap -0.7% → -0.5%)。全部 7 项 verdict 绿。leverage 仍为负(-0.007),和池子大小无关。

详细数字见 [[factor-conditional-interdiff-m4-m5#m6-csi800-scaling]]。

### 2026-04-17 (续2) — M7 modernization + M8 sign-cond

Engine 升级(M7):加 bf16 mixed precision(+42% step/s,-40% GPU)、DDIM sampler(50 steps 比 DDPM-500 快 10×)、classifier-free guidance(`--cfg-drop 0.1` 训练,`--guidance G` 采样)。bf16 和 DDIM 是**净收益无退化**,CFG 在 g=1.5 时部分改善 leverage(-0.011 → -0.003)但 g≥3.0 严重破坏其他指标。

M8 专攻 leverage:加 `--sign-cond` → 在 `mkt_proj` 和 `sector_proj` 旁加两条**只看负数部分**的独立 MLP 支路(`mkt_neg_proj(relu(-m_t))`、`sector_neg_proj(relu(-s_t))`),打破加法 conditioning 的线性对称性。Smoke test 验证模型确实变非对称(`|f(-m)-f(+m)|.mean()=0.15`),但训完采样 leverage 仍为 -0.007 ~ -0.010,**没修复**。

**根因发现**:真实 CSI800 leverage = **+0.013(正)**,我们所有模型都是经典**负 leverage**(-0.003 ~ -0.011)。**方向反了**,不是幅度问题。A 股的 +0.013 是相对于 Black 1976 经典结果的 sign-flipped 现象,可能和 ±10% 涨跌停、散户追涨、T+1 结算有关。详细分析见 [[a-share-positive-leverage]]。

M8 的经验:**模型有容量但 MSE-on-noise 不提供方向梯度** → 仅加 asymmetric 支路不够,需要显式 leverage aux loss 或两阶段 GARCH+InterDiff。

完整 M7/M8 写作见 [[m7-m8-modernization-and-leverage]]。

**决定**:接受 leverage 小 gap(|0.013| 本身是弱效应),推进 α-sweep 下游验证。leverage 若在下游真影响性能,再回头用 aux loss 或 GARCH。

### 2026-04-18 — α-Sweep: no collapse

下游用 138k-param transformer(`NextDayPredictor`,reuse InterBlock)做 next-day rank-IC,训练集混合 (1-α)·real + α·synth。全扫 α ∈ {0, 0.1, 0.25, 0.5, 0.75, 0.9} × 10 seeds × 1500 steps,测试 2023。

**结论:**
- **没看到 model collapse** —— α=0.9(90% 合成)反而取到**最佳 IC**
- 所有 α > 0 都 paired-positive vs α=0(+0.004 ~ +0.006),但方差吃掉了统计显著性(t-stats < 2)
- predictor 本身是瓶颈(raw OHLC 特征太弱 + 2023 test 还是有分布漂移),所有 IC 都在 -0.02 ~ -0.03 的噪声带
- 换句话说:**M6 合成数据下游上约等于真实数据**,没有贬损也没有明显增益

意义:对 M6 是"最大 upper bound 信号" —— 合成数据没有破坏性,可作数据增强。代价:真正检测 phase transition α* 需要(a) 更强 baseline predictor,(b) 递归生成的 scenario(本次是 one-step augmentation),或(c) distribution-level MMD(eval_compare stylized-fact 已经覆盖)。

完整 writeup 见 [[alpha-sweep-csi800-m6]]。

### 2026-04-18 (续) — LGBM α-sweep + recursive collapse

**LGBM α-sweep**([[lgbm-alpha-sweep-phase-transition]]):换成 LightGBM + 17 engineered 特征,baseline IC 从 transformer 的 -0.027 升到 +0.003(可用)。5 seeds 扫 α,**α=0.5 给 +0.0125 IC(paired t=1.85,接近显著)**,α=0.75 回落到 -0.001,α=0.9 恢复到 +0.005。**这是 phase-transition 的早期迹象**:α=0.5 增益最大,α=0.75 首次退化。

**Recursive collapse**([[recursive-collapse-csi800]]):Gen0=M6,Gen(g) 只在 Gen(g-1) 的合成数据上训。3 代后验证 verdicts 从 7/7 → 6/7 → 5/7 → 5/7。关键发现:**不是经典"分布缩窄"式 collapse,而是 factor-conditioning 特有的"common-mode amplification"**:
- pair_corr 每代 +10%(0.347 → 0.394 → 0.429 → 0.483),到 gen3 比 real 高 38%
- leverage 2× 加深然后饱和(-0.007 → -0.016 → -0.029 → -0.029)
- marginals(std/kurt/Hill)基本稳住

机制:每代 market/sector factor 是前一代的 echo 放大。factor conditioning 是 cross-section 问题的解决方案,也是 recursive collapse 的直接致因。**1-step augmentation 安全,recursive self-training 到 gen 1 就破 OK verdict**。

下一步可以试: α-mix recursive(每代 0.5·real + 0.5·prev-synth)看是否 plateau pair_corr。

### 2026-04-17 (续3) — M9 候选:WGR 方法学枢轴

**缘起**:M8 的核心教训是"MSE-on-noise 不提供方向梯度"——即便加了 sign-cond
两条负向支路(smoke test 确认了非对称容量已经有了),leverage 仍停在
-0.003 ~ -0.010。这是 loss 本身的结构问题,不是架构容量问题。

**诊断**:对称损失 $\|\epsilon - \epsilon_\theta\|^2$ 的梯度 在噪声符号下对
称,因此不会给"生成非对称尾巴"任何方向性奖励。需要一个**直接看样本**的损失项。

**候选方案:WGR 双目标损失**(详见 [[wasserstein-generative-regression-framework]])

$$\mathcal{L}_{\text{WGR}} = \lambda_w \cdot \mathcal{L}_W(g_\theta, f_\phi) + \lambda_\ell \cdot \mathcal{L}_{\text{LS}}(g_\theta)$$

- $\mathcal{L}_W$:1-Lipschitz critic 的 Wasserstein-1 对偶 → 看到样本级的
  方向性非对称,是 leverage 唯一可观的直接梯度路径
- $\mathcal{L}_{\text{LS}}$:对条件均值的 MSE,保住边际 calibration 不被
  critic 带偏
- 来自 [[wasserstein-generative-regression]](Huang 组 JRSSB 2025 旗舰);
  该论文 CT-Slices 上 **PI coverage 96% vs cWGAN 48%** 的对比是对我们
  最直接相关的实证:cWGAN 路线的 PI 系统性偏窄,WGR 不会。

**具体落地路径**:选 hybrid(选项 II)—— 保留现有 M5 DDPM 训练流程,在
其上叠加 WGR 辅助项,不推翻 M0-M6 的架构投资。

```
L_total = L_DDPM_existing                              # keep M0-M6 signal
        + mu * [ lambda_w * (-L_W(f_phi; y_gen, Y)) 
               + lambda_ell * ||Y - mean_seed(y_gen)||^2 ]
```

其中 `y_gen` 由 DDIM-5(M7 已经有的 fast sampler)从 $x_t$ 快速 partial
denoise 出 $\hat x_0$ 得到。单 batch 新增成本 ≤ 2× baseline step。

**Critic 设计**:和 InterDenoiser 同构的 panel-symmetric InterBlock + 
N-轴全局池化(stock permutation equivariant);Lipschitz 先用 spectral
normalisation,不够紧再切 gradient penalty。

**对比实验设计**(M9 vs M5 / M8):

| 实验 | 配置 | 预期差异 |
|---|---|---|
| **M9a** | M5 + WGR hybrid,$\mu=0.1$, $\lambda_w=\lambda_\ell=1$ | baseline 看 leverage 是否从 -0.003 移向 0 或 + |
| **M9b** | M9a + sign-cond(M8 的 negative-only branches) | 验证 M8 加的 asymmetric 容量 *是否现在有梯度信号了* |
| **M9c** | M9a,但 $\mu=0.3$ 并把 DDPM 权重递减 | 看纯 WGR 主导是否牺牲了 M4/M5 的 cross-section 收益 |

每个 run 20k 步 × CSI300,~20 min wall;M9b 是 critical test(M8 的
结构 + WGR 的梯度是否互补)。

**预期结果**(写下来事后好验证):

1. M9a leverage 从 -0.003 向 0 或 + 移动至少 0.005(把 50% 的 sign gap 拿回)
2. M9b 比 M9a 更进一步(加法:容量 + 梯度信号都就位)
3. M9c 会牺牲 market_factor_var 的 gap(从 -0.7% 退到 -3% ~ -5%),这告诉
   我们 WGR 权重的上限

**失败路径预案**:

- 若 M9a leverage 无改善 → 说明 critic 架构太弱,学不出 sign-sensitive 特征,
  需要换非对称 aggregator(例如 critic 里直接暴露 $|r|$ 和 $\mathrm{sgn}(r)$
  的乘积作为特征)
- 若训练不稳定(critic 碾压 generator 导致 D-G 崩溃)→ 降 $\mu$ 到 0.03,
  或把 WGR 项 delay 到 DDPM 收敛后(fine-tune phase)
- 若 leverage 动了但 cross-section 退化 → 说明 critic 抓住 panel-level 信号
  后反而忽略 common mode,需要 critic 里分离 residual 和 market 两条 head

**不做**:

- 不做选项 III(distillation 到 one-step generator)—— 太重,等 M9 验证之后再议
- 不改 M0-M6 的 DDPM 架构 —— M4/M5 的 cross-section 收益是结构性的,不要
  动;只在 loss 层叠加
- 不碰 CSI800 —— M9 阶段先在 CSI300 快速迭代,拿到方向性结论再放大

**为什么这是 deepen 而不是 replace**:

M0-M6 做的是**架构**(InterBlock / 市场因子 / 行业因子)—— 这些证明是对的。
M7/M8 做的是**工程+架构细节**(bf16 / DDIM / CFG / sign-cond)—— 前三个是净收
益,第四个没生效。M9 做的是**loss**——换一个能把 M8 sign-cond 的容量"点亮"的
训练目标。三层改动互不冲突;M9 成功是 M4/M5/M8 的增量,不是推翻。

见 [[wasserstein-generative-regression-framework]] 的完整 recipe、陷阱和超参
默认。
