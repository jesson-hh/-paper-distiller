# paper-distiller v1.0 — Chat-First Architecture (Design Spec)

**Date**: 2026-05-19
**Author**: jesson-hh + Claude
**Status**: brainstorm-approved; pending writing-plans

---

## 1. Goal

Replace paper-distiller's two-CLI batch architecture (`paper-distiller` + `paper-distiller-qa`) with a single chat-first entry point `paper-distiller-chat`. The new tool exposes a REPL with slash commands + natural-language input, internally orchestrated as an async DAG of sub-agents whose execution status is rendered in a live table.

This is a v1.0 release (major semver bump) — there is no backwards compatibility with v0.5.x script entry points.

## 2. Motivation

Today paper-distiller is a batch tool. The user types a long flag-laden command, waits, sees a final wall of text. Two friction points:

1. **Discoverability** — users with research goals (vs. specific topics) struggle to translate intent into the right CLI + flags. They either ask Claude Code to do the translation for them (which works but assumes Claude Code is available), or they get the flags wrong and waste API budget.
2. **Observability** — during a multi-paper distill or multi-round QA loop, the user has no idea what's currently happening. Mid-run cancellation requires Ctrl+C with no warning of what'll be lost.

Both friction points dissolve in a chat-first interface with live agent-status tables and natural-language routing. The user pattern becomes:

```
> 帮我研究下扩散在金融时序
[intent-router classifies + asks missing params]
> default budget? [Y/n]
> Y
[live status table during execution]
```

instead of:

```bash
paper-distiller-qa --vault ... --question "扩散在金融时序" \
    --max-rounds 3 --per-round 2 --max-cost-cny 5 --verbose
[wall of verbose text]
```

## 3. Non-goals (YAGNI calls)

- **Multi-user chat / collaboration features.** Single-user, single-vault tool.
- **Web UI.** Stays terminal-only.
- **A general-purpose agent framework as a separable library.** The framework is built for paper-distiller's specific DAG patterns; not designed for reuse elsewhere.
- **Tool calls / function-calling integration with the LLM.** Intent routing is one JSON-out LLM call, not a tool-calling loop.
- **Replacing the LLM client abstraction.** `LLMClient` (OpenAI-compatible) survives unchanged.
- **Changing the vault format or schema.** Output is still markdown + YAML frontmatter + `[[wikilink]]` to the same 6 category directories.

## 4. Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│  paper-distiller-chat  (entry point)                            │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  REPL Layer                                              │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │ Input parser                                       │  │   │
│  │  │  - slash command → direct dispatch                 │  │   │
│  │  │  - natural language → intent-router agent          │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                ↓                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Orchestrator (asyncio)                                  │   │
│  │  - registers agents + their dependencies                 │   │
│  │  - schedules in topological order                        │   │
│  │  - runs deps-free siblings concurrently                  │   │
│  │  - emits status events                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                ↓                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Agents (10 total)                                       │   │
│  │  arxiv-searcher  ss-searcher                             │   │
│  │  candidate-merger  candidate-ranker                      │   │
│  │  paper-processor ×N (parallel)                           │   │
│  │  vault-writer  survey-composer                           │   │
│  │  progress-reflector  answer-synthesizer  intent-router   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                ↓                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Console Renderer (rich live table)                      │   │
│  │  emits status events as live-updating table              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 5. Entry point

Single console script: `paper-distiller-chat`.

```bash
paper-distiller-chat                                  # interactive REPL
paper-distiller-chat distill --topic X --n 3          # one-shot single-pass
paper-distiller-chat ask --question Y                 # one-shot QA loop
paper-distiller-chat resume <sid>                     # resume paused QA
```

One-shot subcommands construct a single DAG, run it, render the same status table, then exit. They use the same agent framework as the REPL — no parallel implementations.

Removed in v1.0 (breaking change vs. v0.5.x):
- `paper-distiller` console script
- `paper-distiller-qa` console script

## 6. Framework specification

Implemented in `src/paper_distiller/agents/` (~300 LOC framework, ~700 LOC agents).

### 6.1 Agent protocol

```python
# src/paper_distiller/agents/base.py
from typing import Protocol

class Agent(Protocol):
    name: str
    deps: list[str]                  # other agent names this depends on

    async def run(self, ctx: Context) -> dict: ...
    # Returns a dict merged into ctx.shared for downstream agents.
```

### 6.2 Context

```python
@dataclass
class Context:
    cfg: Config                      # paper-distiller config (env vars + CLI)
    llm: LLMClient                   # shared LLM client (token accounting included)
    vault: VaultStore                # vault read/write
    shared: dict                     # mutable inter-agent state
    on_status: Callable              # callback for status events
```

`shared` accumulates as agents run. Examples:
- after `arxiv-searcher`: `shared["candidates_arxiv"] = [Paper, ...]`
- after `candidate-merger`: `shared["candidates"] = [Paper, ...]`
- after `candidate-ranker`: `shared["ranked"] = [Paper, ...][:N]`
- after `paper-processor`: `shared["articles"] = [ArticleResult, ...]`

### 6.3 DAG

```python
class DAG:
    def __init__(self, agents: list[Agent]):
        self.agents = {a.name: a for a in agents}
        self._validate_topology()

    def _validate_topology(self):
        # raise on cycles, missing deps, name conflicts
        ...

    def topo_levels(self) -> list[list[str]]:
        # returns groups: [[level0], [level1], ...] where each group
        # has no deps on another within the same group → can run concurrently
        ...
```

### 6.4 Orchestrator

```python
class Orchestrator:
    def __init__(self, dag: DAG, ctx: Context):
        ...

    async def run(self) -> dict:
        # for each topo level:
        #   await asyncio.gather(*(self._run_one(name) for name in level))
        # returns ctx.shared
        ...

    async def _run_one(self, name: str):
        agent = self.dag.agents[name]
        self.ctx.on_status(name, "running")
        try:
            result = await agent.run(self.ctx)
            self.ctx.shared.update(result)
            self.ctx.on_status(name, "done")
        except Exception as e:
            self.ctx.on_status(name, "failed", error=e)
            raise
```

`paper-processor` is special — it's instantiated **per paper** at runtime, not pre-registered. The DAG supports "fanout agents" that produce a list of sub-instances at runtime; see `FanoutAgent` below.

### 6.5 Fanout agents

```python
class FanoutAgent(Protocol):
    """An agent that produces N sub-agents at runtime based on ctx.shared."""
    name: str
    deps: list[str]

    def expand(self, ctx: Context) -> list[Agent]: ...
    # Returns list of leaf agents that all run in parallel.
```

The orchestrator handles fanout by, after deps complete, calling `expand()` and running the returned agents as a synthetic topo level.

### 6.6 ConsoleRenderer

Uses `rich.live.Live` + `rich.table.Table`. Receives status events from the orchestrator's `on_status` callback. Renders:

```
DAG · QA: <question> · Round R/maxR
┌──────────────────────┬──────────┬──────────┐
│ Agent                │ Status   │ Elapsed  │
├──────────────────────┼──────────┼──────────┤
│ arxiv-searcher       │ done     │ 1.2s     │
│ ss-searcher          │ done     │ 1.4s     │
│ candidate-merger     │ done     │ 0.0s     │
│ candidate-ranker     │ done     │ 3.1s     │
│ paper-processor[1/2] │ running  │ 14.7s    │
│ paper-processor[2/2] │ running  │ 14.5s    │
│ vault-writer         │ queued   │ —        │
│ survey-composer      │ queued   │ —        │
└──────────────────────┴──────────┴──────────┘
```

Updates live (~10 Hz). For one-shot subcommands the final state stays on screen after exit.

## 7. Agent inventory

Detailed responsibility + I/O for each agent.

### 7.1 `arxiv-searcher`

- **Deps**: none
- **Input**: `ctx.cfg.topic` (single-pass) or `ctx.shared["next_query"]` (QA mode)
- **Output**: `shared["candidates_arxiv"]: list[Paper]`
- **Wraps**: `sources/arxiv.py::search`

### 7.2 `ss-searcher`

- **Deps**: none
- **Input**: same as arxiv-searcher
- **Output**: `shared["candidates_ss"]: list[Paper]`
- **Wraps**: `sources/semantic_scholar.py::search`

### 7.3 `candidate-merger`

- **Deps**: arxiv-searcher, ss-searcher
- **Input**: `shared["candidates_arxiv"]`, `shared["candidates_ss"]`
- **Output**: `shared["candidates"]: list[Paper]` (deduped)
- **Logic**: same as `pipeline.merge_candidates` today

### 7.4 `candidate-ranker`

- **Deps**: candidate-merger
- **Input**: `shared["candidates"]`, `ctx.cfg.qa_per_round` or `ctx.cfg.top_n`
- **Output**: `shared["ranked"]: list[Paper]` (top-N)
- **Wraps**: `distill/filter.py::rank` (one LLM call)

### 7.5 `paper-processor` (fanout)

- **Deps**: candidate-ranker
- **Fanout**: produces one `_PaperProcessor` instance per paper in `shared["ranked"]`
- **Per-instance work**: PDF fetch + PyMuPDF extract + LLM distill
- **Per-instance output**: appends to `shared["articles"]: list[ArticleResult]`
- **Wraps**: `pipeline.fetch_with_fallback` + `extract/pymupdf_extractor.extract` + `distill/article.py::distill`

### 7.6 `vault-writer`

- **Deps**: paper-processor (all fanout instances)
- **Input**: `shared["articles"]`
- **Output**: `shared["saved_slugs"]: list[str]`
- **Wraps**: `vault/store.py::VaultStore.save_entry` per article

### 7.7 `survey-composer`

- **Deps**: vault-writer
- **Input**: `shared["articles"]`, `ctx.cfg.min_papers_for_survey`
- **Output**: `shared["survey_slug"]: str | None`
- **Wraps**: `distill/survey.py::compose_survey` (one LLM call, skipped if < min)

### 7.8 `progress-reflector` (QA only)

- **Deps**: none (runs at the start of each QA round)
- **Input**: `ctx.cfg.qa_question`, `shared["articles"]` so far, `shared["prior_queries"]`
- **Output**: `shared["reflection"]: dict` with `is_done`, `confidence`, `next_query`, etc.
- **Wraps**: `qa/reflection.py::reflect`

### 7.9 `answer-synthesizer` (QA only)

- **Deps**: vault-writer (after final round)
- **Input**: `ctx.cfg.qa_question`, `shared["articles"]` (all rounds)
- **Output**: `shared["answer_survey_slug"]: str`
- **Wraps**: `qa/answer.py::synthesize` + writes to `surveys/qa-...md`

### 7.10 `intent-router` (chat REPL only)

- **Deps**: none
- **Input**: user's natural-language string
- **Output**: `shared["intent"]: dict` with `command` (one of: distill/ask/resume/show), `params` (slot dict), `missing_params` (list[str])
- **LLM call**: one JSON-out call with a routing prompt

## 8. REPL UX

### 8.1 Welcome banner

```
─ Welcome ────────────────────────────────
paper-distiller v1.0.0
Provider: aliyun-bailian / qwen3.5-plus
Vault: G:\Math research Agent\wiki
Agents registered: 10

Slash commands:
  /distill <topic> [N=3]   /ask <question>   /resume <sid>
  /sessions   /vault   /provider   /agents   /show <slug>
  /help   /quit

Natural language: '帮我研究下扩散在金融时序'
──────────────────────────────────────────
>
```

### 8.2 Slash commands

| Command | Args | Action |
|---|---|---|
| `/distill` | `<topic> [N=3]` | Build & run single-pass DAG |
| `/ask` | `<question>` | Build & run QA DAG. Prompts for missing budget params. |
| `/resume` | `<sid>` | Load SessionState, continue QA DAG |
| `/sessions` | — | List `.paper_distiller/qa-sessions/` entries with question + stop_reason + cost |
| `/vault` | — | Show vault stats: per-category file count, last 5 modified |
| `/provider` | — | Show LLM provider config (masked key) |
| `/agents` | — | List registered agents + dependencies as a tree |
| `/show` | `<slug>` | `cat` an article from vault |
| `/help` | — | Print this same list |
| `/quit` | — | Exit |

Slash commands are deterministic — no LLM call.

### 8.3 Natural-language input

Any input not starting with `/` is dispatched to `intent-router` agent. Intent-router output:

```json
{
  "command": "ask",
  "params": {
    "question": "扩散在金融时序的最新进展",
    "max_rounds": null,
    "per_round": null,
    "max_cost_cny": null
  },
  "missing_params": ["max_rounds", "per_round", "max_cost_cny"]
}
```

REPL then asks user:

```
[intent-router] Intent: ask | Question: "扩散在金融时序的最新进展"
                Missing: max_rounds, per_round, max_cost_cny

Default: max_rounds=3, per_round=2, max_cost_cny=5? [Y/n/edit]
>
```

Three responses:
- `Y` / enter → use defaults, dispatch DAG
- `n` → cancel
- `edit` → enter slot-fill prompt: `max_rounds (default 3): ` for each missing param

### 8.4 Live status table

During DAG execution, the table updates ~10Hz. After completion:

```
DAG · QA: 扩散在金融时序的最新进展 · DONE (stop_reason=llm_done, rounds=2)
┌──────────────────────┬──────┬─────────┐
│ Agent                │ Done │ Elapsed │
├──────────────────────┼──────┼─────────┤
│ ... (final state)
└──────────────────────┴──────┴─────────┘

Cost: ¥3.42  |  Articles distilled: 4  |  Survey: qa-extension-finance-20260519.md
>
```

## 9. State persistence

For QA mode, `qa/state.py::SessionState` survives. Saved per round to `<vault>/.paper_distiller/qa-sessions/<sid>/state.json`. The orchestrator hooks into the QA loop's "after each round" point to persist.

`/resume <sid>` loads SessionState and re-enters the QA loop DAG at the next round.

## 10. Migration mapping (v0.5.x → v1.0)

| v0.5.x | v1.0 fate |
|---|---|
| `paper_distiller/__init__.py` | Bump to 1.0.0 |
| `paper_distiller/cli.py` | DELETE |
| `paper_distiller/pipeline.py` (gather_candidates etc.) | Logic absorbed into `agents/` |
| `paper_distiller/distill/{filter,article,survey}.py` | Logic absorbed into `agents/` |
| `paper_distiller/qa/cli.py` | DELETE |
| `paper_distiller/qa/loop.py` | DELETE; logic into `agents/` + orchestrator's QA-loop driver |
| `paper_distiller/qa/{reflection,answer}.py` | Wrapped by `progress-reflector` and `answer-synthesizer` agents |
| `paper_distiller/qa/state.py` | KEEP — used by orchestrator + `/resume` |
| `paper_distiller/llm/openai_compatible.py` | KEEP unchanged |
| `paper_distiller/sources/{arxiv,semantic_scholar}.py` | KEEP unchanged |
| `paper_distiller/extract/pymupdf_extractor.py` | KEEP unchanged |
| `paper_distiller/vault/{schema,store,crosslink}.py` | KEEP unchanged |
| `paper_distiller/prompts/*.md` | KEEP unchanged |
| `paper_distiller/qa/prompts/*.md` | KEEP unchanged |
| `pyproject.toml` `[project.scripts]` | Two entries removed; `paper-distiller-chat` added |
| `tests/test_*.py` (78 tests) | Most rewritten (see §11) |
| `README.md` | Major rewrite |
| `docs/ARCHITECTURE.md` | Major rewrite |
| `CHANGELOG.md` | `[1.0.0]` section noting BREAKING CHANGE |

## 11. Test strategy

New test layout:

```
tests/
├── test_smoke.py                              (kept, version bumped)
├── agents/
│   ├── test_arxiv_searcher.py                 unit per agent
│   ├── test_ss_searcher.py
│   ├── test_candidate_merger.py
│   ├── test_candidate_ranker.py
│   ├── test_paper_processor.py
│   ├── test_vault_writer.py
│   ├── test_survey_composer.py
│   ├── test_progress_reflector.py
│   ├── test_answer_synthesizer.py
│   └── test_intent_router.py
├── framework/
│   ├── test_dag_topology.py                   cycles, missing deps, validation
│   ├── test_orchestrator.py                   topo execution, fanout, error propagation
│   └── test_console_renderer.py               status event → table rendering
├── repl/
│   ├── test_slash_dispatch.py                 each slash command
│   ├── test_intent_routing_flow.py            NL → params → confirm
│   └── test_resume_flow.py                    /resume reads state.json
└── integration/
    ├── test_single_pass_dag.py                end-to-end single-pass with mocked LLM
    ├── test_qa_loop_dag.py                    end-to-end QA with mocked LLM
    └── test_one_shot_subcommand.py            paper-distiller-chat distill / ask one-shot
```

Estimated test count: ~50 (down from 78, because per-agent tests are tighter than today's pipeline tests).

All LLM calls mocked. No real API calls in CI.

## 12. Implementation phases

Suggested plan-level decomposition (full plan to be written by writing-plans skill):

1. **Phase A — framework only.** `Agent`, `Context`, `DAG`, `Orchestrator`, `ConsoleRenderer`. Unit-tested with 2-3 trivial test agents. ~15 tests.
2. **Phase B — port single-pass.** Implement agents 7.1-7.7 as wrappers around existing v0.5 code. Add `distill` one-shot subcommand. ~15 tests.
3. **Phase C — port QA loop.** Implement agents 7.8-7.9 + QA-loop driver in orchestrator. Add `ask` and `resume` one-shot subcommands. ~10 tests.
4. **Phase D — REPL + intent-router.** Implement REPL loop, slash dispatch, intent-router agent (7.10), confirmation flow. ~10 tests.
5. **Phase E — cleanup.** Delete old CLIs, update pyproject scripts, rewrite README + ARCHITECTURE, write CHANGELOG, bump 1.0.0, tag, release.

Each phase merges incrementally to main; v1.0 not tagged until Phase E done.

## 13. Open questions

These are deferred to implementation-plan or implementation time:

1. **Status table rendering during Ctrl+C** — does `rich.live.Live` cleanly exit, or do we get half-rendered table? Implementation will verify.
2. **Intent-router prompt** — exact JSON schema + system prompt. Drafted in plan task.
3. **Cancellation mid-DAG** — does asyncio cancel propagate? What partial state survives? Plan-time concern.
4. **One-shot subcommand exit codes** — distinct codes per stop_reason? Inherit from v0.5 (0 success / 2 config / 3 runtime)?
5. **Concurrent rounds in QA mode** — current QA is strictly sequential rounds. Could we parallelize "search next round's query" with "distill this round's papers"? Plausible future v1.1, not v1.0.

## 14. Acceptance criteria

v1.0 ships when:

- [ ] `paper-distiller-chat` REPL launches, all slash commands work, `/quit` exits cleanly
- [ ] Natural-language input routes to correct command with slot-fill prompt
- [ ] Single-pass DAG runs end-to-end, status table renders live, vault gets articles
- [ ] QA DAG runs end-to-end with the existing 7 stop reasons; `/resume` works
- [ ] One-shot subcommands `distill` / `ask` / `resume` work
- [ ] Old CLIs (`paper-distiller`, `paper-distiller-qa`) removed; no broken entry points
- [ ] CI green on Python 3.10/3.11/3.12 (~50 tests passing)
- [ ] PyPI publishes v1.0.0 via release.yml workflow
- [ ] README + ARCHITECTURE rewritten for v1.0
- [ ] CHANGELOG documents BREAKING CHANGE in `[1.0.0]` section
