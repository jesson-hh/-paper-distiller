# paper-distiller

> Conversational research agent for arXiv papers. Search → deep-distill → cross-reference proofs. Writes Obsidian-compatible markdown vaults.

[![CI](https://github.com/jesson-hh/paper-distiller/actions/workflows/ci.yml/badge.svg)](https://github.com/jesson-hh/paper-distiller/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/paper-distiller.svg)](https://pypi.org/project/paper-distiller/)
[![Python versions](https://img.shields.io/pypi/pyversions/paper-distiller.svg)](https://pypi.org/project/paper-distiller/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-436_passing-brightgreen.svg)](#)

paper-distiller is a **conversational research agent** that talks to you in natural language, decides which of 7 LLM-callable tools to use, and turns arXiv papers into a deeply-distilled, cross-referenced markdown knowledge base.

```
❯ 帮我搜索最近三年关于扩散模型理论的论文，挑 5 篇蒸馏

⏺ search(topic="diffusion model theory", sort="date", source="arxiv")
● search → 30 candidates  [0.01s]                ← local mirror, zero API hit

⏺ distill_by_id(ids=[...], topic="diffusion theory")
  paper-processor[1/5] LLM distill: Latent Diffusion Convergence...
  paper-processor[2/5] PDF fetch: 2510.12345
  ...
● distill_by_id → 5 articles · 23 theorems extracted · ¥0.21  [12m]

● 已蒸馏 5 篇，全部存进 vault。其中 Theorem 4.3 (arxiv:2510.12345) 用了 Bernstein
  concentration + Dudley chaining，跟 Paper B 的 Lemma 5.1 是同一套技术——已在两篇
  的"与已有 wiki 的关联"中互链。

qwen3.5-plus  ·  54,000 ↑  12,500 ↓  ·  ¥0.2147  ·  default
```

Output is plain markdown + YAML frontmatter + `[[wikilinks]]` — opens directly in [Obsidian](https://obsidian.md), works with Dataview, graph view, full-text search.

---

## Features

- **Conversational REPL** — natural language in, LLM decides tool calls, no flag-juggling
- **7 LLM-callable tools** — `search` · `distill_by_id` · `show` · `ask` · `research` · `ask_user` · `find_proof`
- **Local arXiv mirror** (~1.7M papers, ~5 GB) — bootstrap once via OAI-PMH, search forever zero-latency
- **Deep 12-section distillation** — 3-6k Chinese chars per paper, capturing theorems / proofs / experiments / techniques in a researcher-grade lab-notebook format
- **Cross-paper proof retrieval (RAG)** — every paper's proof sidecar (theorems + techniques) goes into a vault-local SQLite + FTS5 store; future distillations retrieve relevant prior theorems and feed them to the LLM as context, so notation + technique naming converges across the vault
- **Three-way candidate gathering** — hardcoded keyword scan + FTS5 abstract match + LLM pre-extract → cap-and-merge retrieval
- **Multi-source fallback** — arxiv (live + local mirror) / Semantic Scholar / OpenAlex with global per-source throttle + 429 cooldown
- **5 permission modes** — `default` / `auto` / `bypass` / `plan` / `safe`, controlling plan-mode preview behavior
- **Persistent input history** — ↑/↓ navigate past prompts across sessions, Ctrl-R reverse search (prompt_toolkit)
- **Streaming output + spinners** — incremental text, per-agent activity reporting, abort with Ctrl-C
- **Cost tracking** — per-turn + session-wide token + ¥ display, configurable budget gates

---

## Install

```bash
pip install paper-distiller
```

Requires Python **3.10+**. From source:

```bash
git clone https://github.com/jesson-hh/paper-distiller
cd paper-distiller
pip install -e ".[dev]"
pytest -v       # 436 tests should pass
```

---

## Configure

paper-distiller needs an OpenAI-compatible LLM endpoint. Cheapest reliable option: Aliyun Bailian's `qwen-plus` (~¥0.04 per paper at v1.7+ depth).

```bash
cp examples/example.env .env
# Edit .env — set PD_API_KEY, PD_BASE_URL, PD_MODEL
```

### Provider quick reference

| Provider | `PD_BASE_URL` | `PD_MODEL` |
|---|---|---|
| **Aliyun Bailian** (default) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| Aliyun coding plan | `https://coding.dashscope.aliyuncs.com/v1` | `qwen3.5-plus` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| OpenRouter | `https://openrouter.ai/api/v1` | `qwen/qwen3.5-plus` |
| Local Ollama | `http://localhost:11434/v1` | `qwen2.5` |

### Configuration env vars

| Variable | Default | Purpose |
|---|---|---|
| `PD_API_KEY` ✱ | — | LLM API key |
| `PD_BASE_URL` ✱ | — | LLM endpoint base |
| `PD_MODEL` ✱ | — | Model identifier |
| `PD_PERMISSION_MODE` | `default` | Startup mode: `default`/`auto`/`bypass`/`plan`/`safe` |
| `PD_PLAN_THRESHOLD_CNY` | `10.0` | Plan-mode kicks in above this cost |
| `PD_LLM_TIMEOUT` | `600` | LLM read timeout (s); deep distillations can take 3-5 min |
| `PD_FANOUT_CONCURRENCY` | `5` | Parallel LLM calls during multi-paper distill |
| `PD_ARXIV_LOCAL_ONLY` | `0` | If `1`, never fall back to live arXiv API |
| `PD_ARXIV_LOCAL_DIR` | `~/.paper-distiller/arxiv` | Local mirror DB location |
| `PD_HISTORY_FILE` | `~/.paper-distiller/history.jsonl` | Input history file |
| `PD_SS_API_KEY` | (none) | Semantic Scholar API key (raises rate limit ~100×) |

✱ required. Full list in [docs/configuration.md](docs/configuration.md).

---

## Quick start

### 1. One-time arXiv mirror bootstrap (optional but recommended)

```bash
paper-distiller-arxiv bootstrap --since 2020-01-01
# ~2 hours, ~3 GB. Pulls ~600k papers via OAI-PMH. Auto-resumes on SSL errors.
```

Without this, `search` falls back to live arXiv API (rate-limited).
After bootstrap, search hits a local SQLite + FTS5 index at <10 ms.

### 2. Launch the conversational REPL

```bash
paper-distiller-chat --vault /path/to/your/vault
```

You'll see a welcome banner with version, vault, model, and current permission mode. Then talk to it:

```
❯ 给我介绍一下 yuling jiao 最近五年的代表论文
❯ /mode plan                                  # require my OK before any tool
❯ 帮我深度研究扩散模型理论 (research)
❯ vault 里哪些定理用了 Bernstein 不等式？      # → calls find_proof
❯ /cost
❯ /exit
```

`↑` / `↓` cycles through past prompts (across sessions).
Ctrl-C cancels a running tool (conversation continues).
Twice within 1.5s exits the REPL.

### 3. Single-shot mode (for scripts / cron)

```bash
# Distill 5 papers
paper-distiller-chat distill --vault X --topic "diffusion theory" --n 5

# Multi-round QA
paper-distiller-chat ask --vault X --question "近期扩散模型的收敛速率怎样？" --max-rounds 5

# Long deep research (5-phase loop)
paper-distiller-chat research --vault X --question "..." --duration 6h --max-papers 40
```

---

## The 7 LLM-callable tools

| Tool | Purpose |
|---|---|
| `search(topic, n, source, sort)` | Find papers — defaults to local arXiv mirror |
| `distill_by_id(ids, topic)` | Download PDFs + 12-section deep distill + sidecar |
| `show(slug, category)` | Read a vault entry back |
| `ask(question, ...)` | Multi-round QA loop: search → distill → reflect |
| `research(question, ...)` | Long-running 5-phase deep research (default 6h, 40 papers) |
| `ask_user(question, options)` | Pause and let the user pick between 2-4 options |
| `find_proof(query_type, query)` | Query the vault's accumulated theorem / technique knowledge base |

System prompt steers the LLM to use these autonomously. See [docs/tools.md](docs/tools.md) for full schemas.

---

## What a distilled article looks like

Every paper produces a **12-section markdown entry** with this structure:

```markdown
# 双向 GAN 的非渐近误差界

> **场合**: NeurIPS 2021
> **主题**: 首次为 BiGAN 提供联合分布匹配下的非渐近误差界理论保证
> **领域**: 理论机器学习 / 统计学习理论

## TL;DR (一句话)
本文首次为双向 GAN (BiGAN) 提供了基于 Dudley 距离的非渐近误差界...

## 1. 问题动因
传统 GAN 理论分析存在三个显著脱离实际的假设：(1) 维度匹配；(2) 紧支撑...

## 2. 设定与记号
- **目标分布** $\mu$：支撑在 $\mathbb{R}^d$ 上的数据分布
- **联合分布**：$\hat{\nu} = \tilde{g}\#\nu$，$\hat{\mu} = \tilde{e}\#\mu$
- **核心假设**: $\mathcal{F}_1$ 一致有界 1-Lipschitz...

## 3. 核心方法
### 3.1 主要思想
### 3.2 算法/构造
### 3.3 理论分析

## 4. 关键定理 / 命题
**Theorem 4.3** (Cross-Dimensional Empirical Pushforward): ...
*Proof sketch*: ...

## 5. 实验设置
- 数据集: CelebA-HQ (256×256), CIFAR-10
- 基线: BiGAN-baseline, ALI, ALAE
- 评估指标: FID, Inception Score
- 资源: 8× V100, 训练 72 小时

## 6. 关键结果
- 在 CelebA-HQ 上 FID 从 18.4 降到 12.7 (-31%)
- 证明了 $O(n^{-1/2})$ 而非 $O(n^{-1/4})$

## 7. 消融与敏感性
## 8. 局限与失败模式
## 9. 与已有 wiki 的关联       ← [[wikilinks]] to other distilled papers
## 10. 复现要点
## 11. 我的 take
## 12. 引用网络 (可选)
```

Plus a **`proof_sidecar` JSON** stored in `.proof_store/proofs.db`:

```json
{
  "theorems": [
    {
      "name": "Theorem 4.3",
      "statement": "...",
      "proof_sketch": "...",
      "techniques_used": ["Bernstein", "Dudley chaining", "ReLU approximation"]
    }
  ],
  "key_techniques": ["Bernstein", "IPM duality", "ReLU approximation", ...]
}
```

When you later distill a related paper, the LLM **automatically receives** prior theorems whose techniques overlap — keeping notation and citation patterns coherent across the vault.

---

## Permission modes

```
❯ /mode
current permission_mode: default
available modes: default, auto, bypass, plan, safe

  default   show plan-mode preview for tools >= ¥10 (auto-proceed after 5s)
  auto      skip plan-mode previews entirely
  bypass    same as auto (reserved for future destructive-op gates)
  plan      ALWAYS show plan preview, wait for explicit Enter / q
  safe      like plan, but at ¥0 threshold (every tool prompts)

❯ /mode plan
permission_mode → plan
```

The status line color-codes the current mode:

- `default` — dim
- `auto` — yellow
- `bypass` — **bold red** (signal: dangerous)
- `plan` — cyan
- `safe` — bold green

---

## Local arXiv mirror

```bash
paper-distiller-arxiv bootstrap [--since 2020-01-01] [--source auto|oai_pmh|internet_archive|kaggle]
paper-distiller-arxiv sync [--since DATE]      # daily increment
paper-distiller-arxiv search "diffusion" --n 10 --sort date --category cs.LG
paper-distiller-arxiv stats                    # papers count, db size, last sync
paper-distiller-arxiv doctor                   # diagnose integrity + connectivity
```

The mirror uses SQLite + FTS5 + BM25 for keyword + ranked retrieval, all local. Built-in author-search fallback when FTS5 misses on title/abstract.

---

## Cost

Each deep distillation uses ~20-30k input tokens (paper full text) and ~10k output tokens. At qwen-plus rates (¥0.8/M in, ¥2.0/M out):

| Operation | Typical cost | Time |
|---|---|---|
| 1 paper distilled (v1.7+ deep) | ~¥0.04 | ~3 min |
| 5-paper survey | ~¥0.21 | ~5-10 min (5-way concurrent) |
| `ask` 5 rounds × 3 papers | ~¥1-3 | ~15-25 min |
| `research` 6h budget, 40 papers | ~¥2-5 | ~1 hour (with local mirror) |

Configurable via `--max-cost-cny` flags + global `PD_PLAN_THRESHOLD_CNY` env. Plan-mode shows a budget preview before any tool over the threshold runs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ AgentLoop (chat/agent_loop.py)                              │
│   prompt_toolkit input → 7 LLM tools → streaming output     │
└──────────────────┬──────────────────────────────────────────┘
                   ↓ tool call
┌─────────────────────────────────────────────────────────────┐
│ Async DAG orchestrator (agents/orchestrator.py)             │
│   topological scheduling + asyncio.Semaphore fanout cap     │
└─────┬────────────────────┬──────────────────┬───────────────┘
      ↓                    ↓                  ↓
┌──────────────┐  ┌──────────────────┐  ┌─────────────────┐
│ search/      │  │ paper-processor  │  │ vault-writer    │
│ arxiv-local  │  │ × N concurrent   │  │ proof-store     │
│ → 7 agents   │  │ → fetch+distill  │  │ → SQLite + md   │
└──────────────┘  └──────────────────┘  └─────────────────┘
```

Full module map and data flow: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Vault layout

```
your-vault/
├── articles/         # one .md + .html per distilled paper
├── surveys/          # multi-paper syntheses, qa-* final answers
├── techniques/       # reserved for hand-curated notes
├── directions/
├── open-problems/
├── authors/
└── .proof_store/
    └── proofs.db     # SQLite + FTS5 of extracted theorems
```

Markdown is Obsidian-compatible. HTML siblings have MathJax for LaTeX.

---

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, test workflow, and conventions.

Issues: [GitHub Issues](https://github.com/jesson-hh/paper-distiller/issues) (templates for bug / feature / question).

Security disclosures: see [SECURITY.md](SECURITY.md).

Code of conduct: [Contributor Covenant 2.1](CODE_OF_CONDUCT.md).

---

## Citation

If you use paper-distiller in academic work, please cite via [CITATION.cff](CITATION.cff).

---

## License

MIT — see [LICENSE](LICENSE).
