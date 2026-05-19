# paper-distiller v1.0 — Plan 1 (Framework + Single-Pass Agents)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the v1.0 framework + 7 single-pass agents on top of existing v0.5 pipeline code. After this plan: `paper-distiller-chat distill --topic X --n 3` runs end-to-end with a live status table; old `paper-distiller` and `paper-distiller-qa` entry points remain functional (unchanged).

**Architecture:** New `src/paper_distiller/agents/` package containing (a) framework: `Agent` protocol, `Context`, `DAG`, `Orchestrator`, `ConsoleRenderer`; (b) seven agents wrapping existing logic: `arxiv-searcher`, `ss-searcher`, `candidate-merger`, `candidate-ranker`, `paper-processor` (fanout), `vault-writer`, `survey-composer`. New entry point `paper-distiller-chat` initially exposes one subcommand: `distill`.

**Tech Stack:** Python 3.10+, asyncio, `rich` (new dep — terminal table rendering). Existing deps unchanged.

**Spec:** [docs/superpowers/specs/2026-05-19-paper-distiller-v1.0-chat-design.md](../specs/2026-05-19-paper-distiller-v1.0-chat-design.md)

**Working directory:** `G:\paper-distiller\`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add `rich>=13` dep; add `paper-distiller-chat = "paper_distiller.chat.cli:main"` script |
| `src/paper_distiller/agents/__init__.py` | Create | Package marker |
| `src/paper_distiller/agents/base.py` | Create | `Agent` Protocol, `Context` dataclass, `Status` enum |
| `src/paper_distiller/agents/dag.py` | Create | `DAG` class — topology validation + `topo_levels()` |
| `src/paper_distiller/agents/orchestrator.py` | Create | `Orchestrator` — async DAG execution + status events |
| `src/paper_distiller/agents/fanout.py` | Create | `FanoutAgent` protocol + orchestrator hook |
| `src/paper_distiller/agents/renderer.py` | Create | `ConsoleRenderer` — rich live table |
| `src/paper_distiller/agents/searchers.py` | Create | `ArxivSearcher`, `SemanticScholarSearcher` |
| `src/paper_distiller/agents/curation.py` | Create | `CandidateMerger`, `CandidateRanker` |
| `src/paper_distiller/agents/processor.py` | Create | `PaperProcessor` fanout agent |
| `src/paper_distiller/agents/writer.py` | Create | `VaultWriter`, `SurveyComposer` |
| `src/paper_distiller/chat/__init__.py` | Create | Package marker |
| `src/paper_distiller/chat/cli.py` | Create | `paper-distiller-chat` entry; argparse for `distill` subcommand |
| `tests/agents/test_base.py` | Create | Context, Status enum tests |
| `tests/agents/test_dag.py` | Create | Topology validation, cycle detection, topo_levels |
| `tests/agents/test_orchestrator.py` | Create | Sequential and parallel execution, error propagation |
| `tests/agents/test_fanout.py` | Create | Fanout expansion + parallel sub-instance execution |
| `tests/agents/test_renderer.py` | Create | Status event → table state |
| `tests/agents/test_searchers.py` | Create | ArxivSearcher + SemanticScholarSearcher |
| `tests/agents/test_curation.py` | Create | Merger + Ranker |
| `tests/agents/test_processor.py` | Create | PaperProcessor fanout |
| `tests/agents/test_writer.py` | Create | VaultWriter + SurveyComposer |
| `tests/chat/test_distill_cli.py` | Create | One-shot `distill` subcommand end-to-end (mocked) |

**Test count after Plan 1:** 78 (existing) + ~30 (new) = **~108 tests**.

---

## Task 1: agents/ package + Agent protocol + Context + Status

**Files:**
- Create: `src/paper_distiller/agents/__init__.py`
- Create: `src/paper_distiller/agents/base.py`
- Create: `tests/agents/__init__.py`
- Create: `tests/agents/test_base.py`

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_base.py`:

```python
"""Tests for paper_distiller.agents.base — Agent protocol, Context, Status."""
from dataclasses import is_dataclass
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context, Status


def test_status_enum_values():
    """Status enum has the five required states."""
    assert Status.QUEUED.value == "queued"
    assert Status.RUNNING.value == "running"
    assert Status.DONE.value == "done"
    assert Status.FAILED.value == "failed"
    assert Status.SKIPPED.value == "skipped"


def test_context_is_dataclass():
    """Context is a dataclass with the required fields."""
    assert is_dataclass(Context)


def test_context_construction(tmp_path):
    """Context can be constructed with the required attributes."""
    cfg = MagicMock()
    llm = MagicMock()
    vault = MagicMock()
    on_status = MagicMock()

    ctx = Context(cfg=cfg, llm=llm, vault=vault, shared={}, on_status=on_status)

    assert ctx.cfg is cfg
    assert ctx.llm is llm
    assert ctx.vault is vault
    assert ctx.shared == {}
    assert ctx.on_status is on_status


def test_context_shared_is_mutable():
    """ctx.shared can be mutated by agents."""
    ctx = Context(
        cfg=MagicMock(), llm=MagicMock(), vault=MagicMock(),
        shared={}, on_status=MagicMock(),
    )
    ctx.shared["foo"] = "bar"
    assert ctx.shared == {"foo": "bar"}
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_base.py -v
```

Expected: `ModuleNotFoundError: No module named 'paper_distiller.agents'`.

- [ ] **Step 3: Create `agents/__init__.py`**

```python
"""v1.0 sub-agent framework — DAG-orchestrated paper-distiller pipeline."""
```

- [ ] **Step 4: Create `agents/base.py`**

```python
"""Agent protocol, shared Context, status enum."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol


class Status(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Context:
    """State passed between agents during a DAG run."""
    cfg: Any                       # paper_distiller.config.Config
    llm: Any                       # paper_distiller.llm.openai_compatible.LLMClient
    vault: Any                     # paper_distiller.vault.store.VaultStore
    shared: dict = field(default_factory=dict)
    on_status: Callable[[str, Status], None] = lambda name, status: None


class Agent(Protocol):
    """An agent is a named async unit with declared dependencies."""
    name: str
    deps: list[str]

    async def run(self, ctx: Context) -> dict:
        """Run the agent's work. Returns a dict merged into ctx.shared."""
        ...
```

- [ ] **Step 5: Create `tests/agents/__init__.py`**

```python
```

(empty file, just a package marker)

- [ ] **Step 6: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_base.py -v
```

Expected: 4 passed.

- [ ] **Step 7: Run full suite to verify no regressions**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: 78 + 4 = **82 passed**.

- [ ] **Step 8: Commit**

```bash
git add src/paper_distiller/agents/__init__.py src/paper_distiller/agents/base.py tests/agents/__init__.py tests/agents/test_base.py
git commit -m "feat(agents): Agent protocol + Context + Status enum (framework foundation)"
```

---

## Task 2: DAG class with topology validation

**Files:**
- Create: `src/paper_distiller/agents/dag.py`
- Create: `tests/agents/test_dag.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_dag.py`:

```python
"""Tests for paper_distiller.agents.dag — topology validation + topo_levels."""
import pytest

from paper_distiller.agents.dag import DAG, DAGError


class _FakeAgent:
    def __init__(self, name: str, deps: list[str] | None = None):
        self.name = name
        self.deps = deps or []

    async def run(self, ctx):
        return {}


def test_dag_constructs_with_valid_agents():
    a = _FakeAgent("a")
    b = _FakeAgent("b", deps=["a"])
    dag = DAG([a, b])
    assert set(dag.agents.keys()) == {"a", "b"}


def test_dag_rejects_duplicate_names():
    a1 = _FakeAgent("a")
    a2 = _FakeAgent("a")
    with pytest.raises(DAGError, match="duplicate"):
        DAG([a1, a2])


def test_dag_rejects_missing_dep():
    a = _FakeAgent("a", deps=["nonexistent"])
    with pytest.raises(DAGError, match="missing dependency"):
        DAG([a])


def test_dag_rejects_cycle():
    a = _FakeAgent("a", deps=["b"])
    b = _FakeAgent("b", deps=["a"])
    with pytest.raises(DAGError, match="cycle"):
        DAG([a, b])


def test_topo_levels_groups_parallel_agents():
    """Agents with no deps OR all deps in earlier levels go in the same level."""
    a = _FakeAgent("a")          # level 0
    b = _FakeAgent("b")          # level 0
    c = _FakeAgent("c", ["a", "b"])  # level 1
    d = _FakeAgent("d", ["c"])   # level 2
    dag = DAG([a, b, c, d])
    levels = dag.topo_levels()
    assert set(levels[0]) == {"a", "b"}
    assert levels[1] == ["c"]
    assert levels[2] == ["d"]


def test_topo_levels_linear_chain():
    a = _FakeAgent("a")
    b = _FakeAgent("b", ["a"])
    c = _FakeAgent("c", ["b"])
    dag = DAG([a, b, c])
    levels = dag.topo_levels()
    assert levels == [["a"], ["b"], ["c"]]
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_dag.py -v
```

Expected: `ModuleNotFoundError: No module named 'paper_distiller.agents.dag'`.

- [ ] **Step 3: Create `agents/dag.py`**

```python
"""Static DAG of agents — topology validation + topological level grouping."""

from __future__ import annotations

from .base import Agent


class DAGError(ValueError):
    pass


class DAG:
    """Topology of agents. Validates at construction time.

    Provides topo_levels() which returns a list of lists — each inner list
    is a set of agent names that can run concurrently (no deps on each other).
    """

    def __init__(self, agents: list[Agent]):
        names = [a.name for a in agents]
        if len(set(names)) != len(names):
            seen = set()
            dupes = [n for n in names if n in seen or seen.add(n)]
            raise DAGError(f"duplicate agent names: {dupes}")
        self.agents: dict[str, Agent] = {a.name: a for a in agents}
        self._validate_deps()

    def _validate_deps(self) -> None:
        for a in self.agents.values():
            for dep in a.deps:
                if dep not in self.agents:
                    raise DAGError(
                        f"agent {a.name!r} has missing dependency {dep!r}"
                    )
        # cycle detection via DFS
        visiting: set[str] = set()
        visited: set[str] = set()

        def dfs(name: str, path: list[str]) -> None:
            if name in visiting:
                cycle = path[path.index(name):] + [name]
                raise DAGError(f"cycle detected: {' -> '.join(cycle)}")
            if name in visited:
                return
            visiting.add(name)
            for dep in self.agents[name].deps:
                dfs(dep, path + [name])
            visiting.discard(name)
            visited.add(name)

        for name in self.agents:
            dfs(name, [])

    def topo_levels(self) -> list[list[str]]:
        """Group agents into parallel-executable levels.

        Level k contains all agents whose deps are all in levels < k.
        Returns a list of level groups, in execution order.
        """
        levels: list[list[str]] = []
        placed: set[str] = set()
        remaining = set(self.agents.keys())

        while remaining:
            this_level = [
                name for name in remaining
                if all(dep in placed for dep in self.agents[name].deps)
            ]
            if not this_level:
                raise DAGError("topo_levels: stuck (should never happen if validation passed)")
            this_level.sort()  # stable ordering
            levels.append(this_level)
            placed.update(this_level)
            remaining -= set(this_level)
        return levels
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_dag.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **88 passed** (82 + 6).

- [ ] **Step 6: Commit**

```bash
git add src/paper_distiller/agents/dag.py tests/agents/test_dag.py
git commit -m "feat(agents): DAG class with topology validation and topo_levels"
```

---

## Task 3: Orchestrator with async DAG execution

**Files:**
- Create: `src/paper_distiller/agents/orchestrator.py`
- Create: `tests/agents/test_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_orchestrator.py`:

```python
"""Tests for paper_distiller.agents.orchestrator — async DAG execution."""
import asyncio
import time
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context, Status
from paper_distiller.agents.dag import DAG
from paper_distiller.agents.orchestrator import Orchestrator, AgentFailed


class _StubAgent:
    def __init__(self, name, deps=None, output=None, sleep=0.0, raises=None):
        self.name = name
        self.deps = deps or []
        self._output = output or {}
        self._sleep = sleep
        self._raises = raises
        self.run_started_at = None
        self.run_finished_at = None

    async def run(self, ctx):
        self.run_started_at = time.monotonic()
        if self._sleep:
            await asyncio.sleep(self._sleep)
        if self._raises:
            raise self._raises
        self.run_finished_at = time.monotonic()
        return self._output


def _ctx(**overrides):
    base = dict(
        cfg=MagicMock(), llm=MagicMock(), vault=MagicMock(),
        shared={}, on_status=MagicMock(),
    )
    base.update(overrides)
    return Context(**base)


@pytest.mark.asyncio
async def test_orchestrator_runs_single_agent():
    a = _StubAgent("a", output={"x": 1})
    ctx = _ctx()
    result = await Orchestrator(DAG([a]), ctx).run()
    assert result["x"] == 1
    assert a.run_finished_at is not None


@pytest.mark.asyncio
async def test_orchestrator_runs_linear_chain():
    a = _StubAgent("a", output={"a_out": 1})
    b = _StubAgent("b", deps=["a"], output={"b_out": 2})
    ctx = _ctx()
    result = await Orchestrator(DAG([a, b]), ctx).run()
    assert result == {"a_out": 1, "b_out": 2}


@pytest.mark.asyncio
async def test_orchestrator_runs_parallel_siblings():
    """Two no-deps agents both sleep 0.2s — total wall time < 0.4s if parallel."""
    a = _StubAgent("a", sleep=0.2)
    b = _StubAgent("b", sleep=0.2)
    ctx = _ctx()
    t0 = time.monotonic()
    await Orchestrator(DAG([a, b]), ctx).run()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.35  # parallel, not 0.4+ sequential


@pytest.mark.asyncio
async def test_orchestrator_emits_status_events():
    a = _StubAgent("a")
    events = []
    ctx = _ctx(on_status=lambda name, status, **kw: events.append((name, status)))
    await Orchestrator(DAG([a]), ctx).run()
    statuses = [s for _, s in events]
    assert Status.RUNNING in statuses
    assert Status.DONE in statuses


@pytest.mark.asyncio
async def test_orchestrator_propagates_agent_error():
    a = _StubAgent("a", raises=RuntimeError("boom"))
    ctx = _ctx()
    with pytest.raises(AgentFailed) as exc_info:
        await Orchestrator(DAG([a]), ctx).run()
    assert exc_info.value.agent_name == "a"
    assert "boom" in str(exc_info.value.__cause__)
```

You'll need `pytest-asyncio` for the `@pytest.mark.asyncio` marker.

- [ ] **Step 2: Add pytest-asyncio to dev deps**

Edit `pyproject.toml`:

Find:
```toml
dev = ["pytest>=8.0", "pytest-mock>=3.12", "ruff>=0.5"]
```

Replace with:
```toml
dev = ["pytest>=8.0", "pytest-mock>=3.12", "pytest-asyncio>=0.23", "ruff>=0.5"]
```

Add to `[tool.pytest.ini_options]`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
asyncio_mode = "auto"
```

(The existing `[tool.pytest.ini_options]` section already has `testpaths` and `pythonpath` — add `asyncio_mode = "auto"` to it.)

- [ ] **Step 3: Install the new dep**

```bash
.venv\Scripts\python.exe -m pip install -e ".[dev]" --quiet
```

Expected: pytest-asyncio installed; no errors.

- [ ] **Step 4: Run tests to confirm they fail**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_orchestrator.py -v
```

Expected: `ModuleNotFoundError: No module named 'paper_distiller.agents.orchestrator'`.

- [ ] **Step 5: Create `agents/orchestrator.py`**

```python
"""Async DAG orchestrator. Schedules agents in topological order,
runs parallel siblings concurrently, propagates errors as AgentFailed."""

from __future__ import annotations

import asyncio

from .base import Context, Status
from .dag import DAG


class AgentFailed(RuntimeError):
    def __init__(self, agent_name: str):
        super().__init__(f"agent {agent_name!r} failed")
        self.agent_name = agent_name


class Orchestrator:
    def __init__(self, dag: DAG, ctx: Context):
        self.dag = dag
        self.ctx = ctx

    async def run(self) -> dict:
        for name in self.dag.agents:
            self.ctx.on_status(name, Status.QUEUED)

        for level in self.dag.topo_levels():
            await asyncio.gather(*(self._run_one(name) for name in level))
        return self.ctx.shared

    async def _run_one(self, name: str) -> None:
        agent = self.dag.agents[name]
        self.ctx.on_status(name, Status.RUNNING)
        try:
            result = await agent.run(self.ctx)
            self.ctx.shared.update(result or {})
            self.ctx.on_status(name, Status.DONE)
        except Exception as e:
            self.ctx.on_status(name, Status.FAILED, error=e)
            raise AgentFailed(name) from e
```

- [ ] **Step 6: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_orchestrator.py -v
```

Expected: 5 passed.

- [ ] **Step 7: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **93 passed** (88 + 5).

- [ ] **Step 8: Commit**

```bash
git add src/paper_distiller/agents/orchestrator.py tests/agents/test_orchestrator.py pyproject.toml
git commit -m "feat(agents): asyncio Orchestrator + pytest-asyncio dev dep"
```

---

## Task 4: FanoutAgent support

**Files:**
- Create: `src/paper_distiller/agents/fanout.py`
- Modify: `src/paper_distiller/agents/orchestrator.py`
- Create: `tests/agents/test_fanout.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_fanout.py`:

```python
"""Tests for FanoutAgent — runtime expansion of one agent into N parallel sub-agents."""
import asyncio
import time
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context, Status
from paper_distiller.agents.dag import DAG
from paper_distiller.agents.orchestrator import Orchestrator
from paper_distiller.agents.fanout import FanoutAgent


class _LeafAgent:
    def __init__(self, name, value):
        self.name = name
        self.deps = []
        self._value = value

    async def run(self, ctx):
        await asyncio.sleep(0.1)
        return {f"leaf_{self._value}": self._value}


class _FanOutOfThree(FanoutAgent):
    name = "fan"
    deps = []

    def expand(self, ctx):
        return [_LeafAgent(f"leaf-{i}", i) for i in range(3)]


def _ctx(**overrides):
    base = dict(
        cfg=MagicMock(), llm=MagicMock(), vault=MagicMock(),
        shared={}, on_status=MagicMock(),
    )
    base.update(overrides)
    return Context(**base)


@pytest.mark.asyncio
async def test_fanout_produces_n_sub_agents_running_in_parallel():
    fan = _FanOutOfThree()
    ctx = _ctx()
    t0 = time.monotonic()
    result = await Orchestrator(DAG([fan]), ctx).run()
    elapsed = time.monotonic() - t0
    assert result == {"leaf_0": 0, "leaf_1": 1, "leaf_2": 2}
    # 3 leaves each sleep 0.1s — parallel total should be < 0.2s
    assert elapsed < 0.25


@pytest.mark.asyncio
async def test_fanout_emits_status_for_each_sub_agent():
    fan = _FanOutOfThree()
    events = []
    ctx = _ctx(on_status=lambda name, status, **kw: events.append((name, status)))
    await Orchestrator(DAG([fan]), ctx).run()
    leaf_done = {n for n, s in events if s == Status.DONE and n.startswith("leaf-")}
    assert leaf_done == {"leaf-0", "leaf-1", "leaf-2"}
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_fanout.py -v
```

Expected: `ModuleNotFoundError: No module named 'paper_distiller.agents.fanout'`.

- [ ] **Step 3: Create `agents/fanout.py`**

```python
"""FanoutAgent — agent that expands into N runtime-determined sub-agents."""

from __future__ import annotations

from typing import Protocol

from .base import Agent, Context


class FanoutAgent(Protocol):
    """An agent whose work is to produce N parallel sub-agents at runtime.

    The orchestrator detects FanoutAgent via the `expand` method (vs. `run`)
    and treats the returned list as a synthetic parallel level.
    """
    name: str
    deps: list[str]

    def expand(self, ctx: Context) -> list[Agent]:
        ...
```

- [ ] **Step 4: Modify orchestrator to handle FanoutAgent**

In `src/paper_distiller/agents/orchestrator.py`, replace `_run_one` with:

```python
    async def _run_one(self, name: str) -> None:
        agent = self.dag.agents[name]
        if hasattr(agent, "expand") and not hasattr(agent, "run"):
            # FanoutAgent: expand and run sub-agents in parallel
            self.ctx.on_status(name, Status.RUNNING)
            try:
                sub_agents = agent.expand(self.ctx)
                for sub in sub_agents:
                    self.ctx.on_status(sub.name, Status.QUEUED)
                await asyncio.gather(*(self._run_sub(sub) for sub in sub_agents))
                self.ctx.on_status(name, Status.DONE)
            except Exception as e:
                self.ctx.on_status(name, Status.FAILED, error=e)
                raise AgentFailed(name) from e
            return

        # Regular Agent
        self.ctx.on_status(name, Status.RUNNING)
        try:
            result = await agent.run(self.ctx)
            self.ctx.shared.update(result or {})
            self.ctx.on_status(name, Status.DONE)
        except Exception as e:
            self.ctx.on_status(name, Status.FAILED, error=e)
            raise AgentFailed(name) from e

    async def _run_sub(self, sub: "Agent") -> None:
        self.ctx.on_status(sub.name, Status.RUNNING)
        try:
            result = await sub.run(self.ctx)
            self.ctx.shared.update(result or {})
            self.ctx.on_status(sub.name, Status.DONE)
        except Exception as e:
            self.ctx.on_status(sub.name, Status.FAILED, error=e)
            raise AgentFailed(sub.name) from e
```

(Add `from .base import Agent` import if not already present.)

- [ ] **Step 5: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_fanout.py tests/agents/test_orchestrator.py -v
```

Expected: 2 + 5 = 7 passed (the new fanout tests + existing orchestrator tests still pass).

- [ ] **Step 6: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **95 passed** (93 + 2).

- [ ] **Step 7: Commit**

```bash
git add src/paper_distiller/agents/fanout.py src/paper_distiller/agents/orchestrator.py tests/agents/test_fanout.py
git commit -m "feat(agents): FanoutAgent support (runtime expansion into parallel sub-agents)"
```

---

## Task 5: ConsoleRenderer (rich live table)

**Files:**
- Create: `src/paper_distiller/agents/renderer.py`
- Create: `tests/agents/test_renderer.py`
- Modify: `pyproject.toml` (already added rich dep in Task 6 below — wait; actually add it here)

- [ ] **Step 1: Add rich dep**

Edit `pyproject.toml` dependencies section:

Find:
```toml
dependencies = [
    "httpx>=0.27",
    "arxiv>=2.1",
    "pymupdf>=1.24",
    "python-dotenv>=1.0",
    "tomli>=2.0;python_version<'3.11'",
]
```

Replace with:
```toml
dependencies = [
    "httpx>=0.27",
    "arxiv>=2.1",
    "pymupdf>=1.24",
    "python-dotenv>=1.0",
    "rich>=13",
    "tomli>=2.0;python_version<'3.11'",
]
```

- [ ] **Step 2: Install**

```bash
.venv\Scripts\python.exe -m pip install -e . --quiet
```

Expected: rich installed; no errors.

- [ ] **Step 3: Write the failing tests**

Create `tests/agents/test_renderer.py`:

```python
"""Tests for ConsoleRenderer — status events accumulate into a table state."""
import time

import pytest

from paper_distiller.agents.base import Status
from paper_distiller.agents.renderer import ConsoleRenderer


def test_renderer_records_queued_state():
    r = ConsoleRenderer(title="Test")
    r.on_status("a", Status.QUEUED)
    snap = r.snapshot()
    assert snap["a"]["status"] == Status.QUEUED
    assert snap["a"]["elapsed"] is None


def test_renderer_running_records_start_time():
    r = ConsoleRenderer(title="Test")
    r.on_status("a", Status.RUNNING)
    snap = r.snapshot()
    assert snap["a"]["status"] == Status.RUNNING
    assert snap["a"]["started_at"] is not None


def test_renderer_done_records_elapsed():
    r = ConsoleRenderer(title="Test")
    r.on_status("a", Status.RUNNING)
    time.sleep(0.05)
    r.on_status("a", Status.DONE)
    snap = r.snapshot()
    assert snap["a"]["status"] == Status.DONE
    assert snap["a"]["elapsed"] >= 0.05


def test_renderer_failed_records_error():
    r = ConsoleRenderer(title="Test")
    r.on_status("a", Status.RUNNING)
    r.on_status("a", Status.FAILED, error=RuntimeError("boom"))
    snap = r.snapshot()
    assert snap["a"]["status"] == Status.FAILED
    assert "boom" in str(snap["a"]["error"])


def test_renderer_build_table_returns_rich_table():
    """build_table() returns something rich can render (no exceptions)."""
    from rich.table import Table
    r = ConsoleRenderer(title="Test")
    r.on_status("a", Status.RUNNING)
    r.on_status("a", Status.DONE)
    table = r.build_table()
    assert isinstance(table, Table)
```

- [ ] **Step 4: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_renderer.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 5: Create `agents/renderer.py`**

```python
"""ConsoleRenderer — receives status events, exposes a rich Table snapshot.

Live rendering (rich.live.Live) is wired up in the CLI layer; this module
only owns state + table construction.
"""

from __future__ import annotations

import time
from typing import Any

from rich.table import Table

from .base import Status


class ConsoleRenderer:
    def __init__(self, title: str = ""):
        self.title = title
        self._rows: dict[str, dict[str, Any]] = {}

    def on_status(self, name: str, status: Status, **kw) -> None:
        """Status event callback. Updates internal row state."""
        row = self._rows.setdefault(name, {
            "status": Status.QUEUED,
            "started_at": None,
            "elapsed": None,
            "error": None,
        })
        row["status"] = status
        if status == Status.RUNNING and row["started_at"] is None:
            row["started_at"] = time.monotonic()
        elif status in (Status.DONE, Status.FAILED, Status.SKIPPED):
            if row["started_at"] is not None:
                row["elapsed"] = time.monotonic() - row["started_at"]
            if "error" in kw:
                row["error"] = kw["error"]

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a copy of the current row state. For tests."""
        return {k: dict(v) for k, v in self._rows.items()}

    def build_table(self) -> Table:
        """Return a rich Table reflecting current state."""
        table = Table(title=self.title or None, show_header=True)
        table.add_column("Agent")
        table.add_column("Status")
        table.add_column("Elapsed")

        for name, row in self._rows.items():
            status_str = row["status"].value
            if row["elapsed"] is not None:
                elapsed_str = f"{row['elapsed']:.1f}s"
            elif row["started_at"] is not None:
                elapsed_str = f"{(time.monotonic() - row['started_at']):.1f}s"
            else:
                elapsed_str = "—"
            table.add_row(name, status_str, elapsed_str)
        return table
```

- [ ] **Step 6: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_renderer.py -v
```

Expected: 5 passed.

- [ ] **Step 7: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **100 passed** (95 + 5).

- [ ] **Step 8: Commit**

```bash
git add src/paper_distiller/agents/renderer.py tests/agents/test_renderer.py pyproject.toml
git commit -m "feat(agents): ConsoleRenderer + rich runtime dep"
```

---

## Task 6: ArxivSearcher + SemanticScholarSearcher agents

**Files:**
- Create: `src/paper_distiller/agents/searchers.py`
- Create: `tests/agents/test_searchers.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_searchers.py`:

```python
"""Tests for ArxivSearcher + SemanticScholarSearcher agents — wrap existing source modules."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.searchers import ArxivSearcher, SemanticScholarSearcher
from paper_distiller.sources.arxiv import Paper


def _paper(arxiv_id):
    return Paper(
        source="arxiv", paper_id=arxiv_id, arxiv_id=arxiv_id,
        title=f"P{arxiv_id}", authors=[], abstract="...",
        pdf_url="...", published="2025-01-01", categories=[],
    )


def _ctx_with_topic(topic="diffusion models"):
    cfg = SimpleNamespace(
        topic=topic, author=None, pool=10, source="both",
        ss_api_key=None,
    )
    return Context(
        cfg=cfg, llm=MagicMock(), vault=MagicMock(),
        shared={}, on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_arxiv_searcher_writes_candidates_arxiv(mocker):
    fake_papers = [_paper("2501.00001"), _paper("2501.00002")]
    mocker.patch(
        "paper_distiller.agents.searchers.arxiv_search",
        return_value=fake_papers,
    )
    ctx = _ctx_with_topic()
    agent = ArxivSearcher()
    out = await agent.run(ctx)
    assert out["candidates_arxiv"] == fake_papers


@pytest.mark.asyncio
async def test_arxiv_searcher_uses_qa_next_query_when_present(mocker):
    """If shared['next_query'] is set (QA mode), it overrides cfg.topic."""
    fake_search = mocker.patch(
        "paper_distiller.agents.searchers.arxiv_search",
        return_value=[],
    )
    ctx = _ctx_with_topic("ignored_topic")
    ctx.shared["next_query"] = "qa-mode-query"
    await ArxivSearcher().run(ctx)
    fake_search.assert_called_once()
    assert fake_search.call_args.kwargs.get("topic") == "qa-mode-query" \
        or "qa-mode-query" in fake_search.call_args.args


@pytest.mark.asyncio
async def test_ss_searcher_writes_candidates_ss(mocker):
    fake_papers = [_paper("ss-1")]
    mocker.patch(
        "paper_distiller.agents.searchers.ss_search",
        return_value=fake_papers,
    )
    ctx = _ctx_with_topic()
    out = await SemanticScholarSearcher().run(ctx)
    assert out["candidates_ss"] == fake_papers


@pytest.mark.asyncio
async def test_searchers_have_no_deps():
    assert ArxivSearcher().deps == []
    assert SemanticScholarSearcher().deps == []


@pytest.mark.asyncio
async def test_searchers_skip_when_source_excludes_them(mocker):
    """If cfg.source == 'arxiv', SS searcher returns empty without calling the API."""
    fake_search = mocker.patch("paper_distiller.agents.searchers.ss_search")
    ctx = _ctx_with_topic()
    ctx.cfg.source = "arxiv"
    out = await SemanticScholarSearcher().run(ctx)
    fake_search.assert_not_called()
    assert out == {"candidates_ss": []}
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_searchers.py -v
```

Expected: ModuleNotFoundError on `paper_distiller.agents.searchers`.

- [ ] **Step 3: Inspect existing arxiv/SS source APIs**

Run:
```bash
.venv\Scripts\python.exe -c "from paper_distiller.sources.arxiv import search; help(search)"
.venv\Scripts\python.exe -c "from paper_distiller.sources.semantic_scholar import search; help(search)"
```

Note their signatures — they will be wrapped sync calls. The agent's `async def run()` will use `asyncio.to_thread()` to call them.

- [ ] **Step 4: Create `agents/searchers.py`**

```python
"""Search-source agents — wrap existing sources/{arxiv,semantic_scholar}.py.

Both run as no-deps agents and can execute in parallel (each level-0 in the DAG).
"""

from __future__ import annotations

import asyncio

from ..sources.arxiv import search as arxiv_search
from ..sources.semantic_scholar import search as ss_search
from .base import Context


class ArxivSearcher:
    name = "arxiv-searcher"
    deps: list[str] = []

    async def run(self, ctx: Context) -> dict:
        if ctx.cfg.source not in ("arxiv", "both"):
            return {"candidates_arxiv": []}
        query = ctx.shared.get("next_query") or ctx.cfg.topic or ctx.cfg.author or ""
        papers = await asyncio.to_thread(
            arxiv_search,
            topic=query,
            pool=ctx.cfg.pool,
        )
        return {"candidates_arxiv": papers}


class SemanticScholarSearcher:
    name = "ss-searcher"
    deps: list[str] = []

    async def run(self, ctx: Context) -> dict:
        if ctx.cfg.source not in ("ss", "both"):
            return {"candidates_ss": []}
        query = ctx.shared.get("next_query") or ctx.cfg.topic or ctx.cfg.author or ""
        papers = await asyncio.to_thread(
            ss_search,
            topic=query,
            pool=ctx.cfg.pool,
            ss_api_key=ctx.cfg.ss_api_key,
        )
        return {"candidates_ss": papers}
```

Note: if the actual function signatures from Step 3 differ (e.g., `arxiv_search(query, ...)` vs. `arxiv_search(topic=...)`), adjust the kwargs accordingly. The test mocks the module-level name so the call shape just has to work.

- [ ] **Step 5: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_searchers.py -v
```

Expected: 5 passed. If any fail due to argument signature mismatch, adjust either the agent code (Step 4) or the test (Step 1) to match the actual existing `arxiv.search` / `semantic_scholar.search` signatures observed in Step 3.

- [ ] **Step 6: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **105 passed** (100 + 5).

- [ ] **Step 7: Commit**

```bash
git add src/paper_distiller/agents/searchers.py tests/agents/test_searchers.py
git commit -m "feat(agents): ArxivSearcher + SemanticScholarSearcher (parallel source agents)"
```

---

## Task 7: CandidateMerger + CandidateRanker agents

**Files:**
- Create: `src/paper_distiller/agents/curation.py`
- Create: `tests/agents/test_curation.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_curation.py`:

```python
"""Tests for CandidateMerger + CandidateRanker agents."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.curation import CandidateMerger, CandidateRanker
from paper_distiller.sources.arxiv import Paper


def _paper(pid, doi=None):
    return Paper(
        source="arxiv", paper_id=pid, arxiv_id=pid, doi=doi,
        title=f"P{pid}", authors=[], abstract="...",
        pdf_url="...", published="2025-01-01", categories=[],
    )


def _ctx(**shared):
    return Context(
        cfg=SimpleNamespace(top_n=2, qa_per_round=2),
        llm=MagicMock(), vault=MagicMock(),
        shared=dict(shared),
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_merger_combines_two_sources_and_dedups():
    a = [_paper("X1"), _paper("X2")]
    b = [_paper("X2"), _paper("X3")]  # X2 duplicates
    ctx = _ctx(candidates_arxiv=a, candidates_ss=b)
    out = await CandidateMerger().run(ctx)
    ids = [p.arxiv_id for p in out["candidates"]]
    # X2 dedup'd; arxiv wins on tie so X2 stays from `a`
    assert ids == ["X1", "X2", "X3"]


@pytest.mark.asyncio
async def test_merger_handles_empty_sources():
    ctx = _ctx(candidates_arxiv=[], candidates_ss=[])
    out = await CandidateMerger().run(ctx)
    assert out["candidates"] == []


@pytest.mark.asyncio
async def test_merger_deps():
    assert set(CandidateMerger().deps) == {"arxiv-searcher", "ss-searcher"}


@pytest.mark.asyncio
async def test_ranker_uses_top_n(mocker):
    candidates = [_paper(f"X{i}") for i in range(5)]
    fake_rank = mocker.patch(
        "paper_distiller.agents.curation.rank",
        side_effect=lambda candidates, topic, top_n, llm: candidates[:top_n],
    )
    ctx = _ctx(candidates=candidates)
    ctx.cfg.top_n = 3
    ctx.cfg.qa_per_round = None  # single-pass uses top_n
    out = await CandidateRanker().run(ctx)
    assert len(out["ranked"]) == 3


@pytest.mark.asyncio
async def test_ranker_uses_qa_per_round_when_set(mocker):
    """In QA mode, qa_per_round overrides top_n."""
    candidates = [_paper(f"X{i}") for i in range(5)]
    mocker.patch(
        "paper_distiller.agents.curation.rank",
        side_effect=lambda candidates, topic, top_n, llm: candidates[:top_n],
    )
    ctx = _ctx(candidates=candidates)
    ctx.cfg.top_n = 999
    ctx.cfg.qa_per_round = 2
    out = await CandidateRanker().run(ctx)
    assert len(out["ranked"]) == 2


@pytest.mark.asyncio
async def test_ranker_deps():
    assert CandidateRanker().deps == ["candidate-merger"]
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_curation.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Inspect existing merge + rank functions**

```bash
.venv\Scripts\python.exe -c "from paper_distiller.pipeline import merge_candidates; help(merge_candidates)"
.venv\Scripts\python.exe -c "from paper_distiller.distill.filter import rank; help(rank)"
```

Note the signatures to call correctly.

- [ ] **Step 4: Create `agents/curation.py`**

```python
"""CandidateMerger + CandidateRanker — combine + LLM-rank candidate Papers."""

from __future__ import annotations

import asyncio

from ..distill.filter import rank
from ..pipeline import merge_candidates
from .base import Context


class CandidateMerger:
    name = "candidate-merger"
    deps = ["arxiv-searcher", "ss-searcher"]

    async def run(self, ctx: Context) -> dict:
        a = ctx.shared.get("candidates_arxiv", [])
        b = ctx.shared.get("candidates_ss", [])
        merged = merge_candidates(a, b)
        return {"candidates": merged}


class CandidateRanker:
    name = "candidate-ranker"
    deps = ["candidate-merger"]

    async def run(self, ctx: Context) -> dict:
        candidates = ctx.shared.get("candidates", [])
        if not candidates:
            return {"ranked": []}
        # Prefer qa_per_round (set in QA mode) over top_n (single-pass)
        top_n = ctx.cfg.qa_per_round if ctx.cfg.qa_per_round else ctx.cfg.top_n
        topic = ctx.shared.get("next_query") or ctx.cfg.topic or ""
        ranked = await asyncio.to_thread(
            rank, candidates, topic, top_n, ctx.llm,
        )
        return {"ranked": ranked}
```

- [ ] **Step 5: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_curation.py -v
```

Expected: 6 passed. If `merge_candidates` has a different signature, adjust the agent code accordingly.

- [ ] **Step 6: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **111 passed** (105 + 6).

- [ ] **Step 7: Commit**

```bash
git add src/paper_distiller/agents/curation.py tests/agents/test_curation.py
git commit -m "feat(agents): CandidateMerger + CandidateRanker (curation phase)"
```

---

## Task 8: PaperProcessor (fanout) agent

**Files:**
- Create: `src/paper_distiller/agents/processor.py`
- Create: `tests/agents/test_processor.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_processor.py`:

```python
"""Tests for PaperProcessor fanout agent — one sub-instance per ranked paper."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.dag import DAG
from paper_distiller.agents.orchestrator import Orchestrator
from paper_distiller.agents.processor import PaperProcessor
from paper_distiller.sources.arxiv import Paper
from paper_distiller.distill.article import ArticleResult


def _paper(pid):
    return Paper(
        source="arxiv", paper_id=pid, arxiv_id=pid,
        title=f"P{pid}", authors=[], abstract="...",
        pdf_url=f"https://x/{pid}.pdf", published="2025-01-01",
        categories=[],
    )


def _ctx_with_ranked(papers, **cfg_overrides):
    cfg = SimpleNamespace(
        pdf_timeout_sec=60, verbose=False, source="both",
        **cfg_overrides,
    )
    return Context(
        cfg=cfg, llm=MagicMock(), vault=MagicMock(),
        shared={"ranked": papers},
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_processor_fans_out_one_subagent_per_paper(mocker):
    papers = [_paper("X1"), _paper("X2"), _paper("X3")]
    mocker.patch("paper_distiller.agents.processor.fetch_with_fallback", return_value="x" * 600)
    mocker.patch(
        "paper_distiller.agents.processor.distill_article",
        side_effect=lambda paper, full_text, wiki_index, llm: ArticleResult(
            slug=f"a-{paper.arxiv_id}", title=f"T-{paper.arxiv_id}",
            body="b", tags=[], refs=[f"arxiv:{paper.arxiv_id}"],
            depth="full-pdf",
        ),
    )
    mocker.patch("paper_distiller.agents.processor.load_index", return_value=MagicMock(slugs=lambda: set()))

    ctx = _ctx_with_ranked(papers)
    orch = Orchestrator(DAG([PaperProcessor()]), ctx)
    await orch.run()
    assert len(ctx.shared["articles"]) == 3
    assert {a.slug for a in ctx.shared["articles"]} == {"a-X1", "a-X2", "a-X3"}


@pytest.mark.asyncio
async def test_processor_handles_distill_failure_gracefully(mocker):
    """Per-paper distill failure does NOT abort the whole fanout — just drops that paper."""
    from paper_distiller.llm.openai_compatible import LLMError
    papers = [_paper("X1"), _paper("X2")]
    mocker.patch("paper_distiller.agents.processor.fetch_with_fallback", return_value="x" * 600)
    mocker.patch(
        "paper_distiller.agents.processor.distill_article",
        side_effect=[
            ArticleResult(slug="a-X1", title="T1", body="b", tags=[], refs=[], depth="full-pdf"),
            LLMError("LLM borked"),
        ],
    )
    mocker.patch("paper_distiller.agents.processor.load_index", return_value=MagicMock(slugs=lambda: set()))

    ctx = _ctx_with_ranked(papers)
    orch = Orchestrator(DAG([PaperProcessor()]), ctx)
    await orch.run()
    assert len(ctx.shared["articles"]) == 1
    assert ctx.shared["articles"][0].slug == "a-X1"


@pytest.mark.asyncio
async def test_processor_no_ranked_papers_is_noop():
    ctx = _ctx_with_ranked([])
    orch = Orchestrator(DAG([PaperProcessor()]), ctx)
    await orch.run()
    assert ctx.shared.get("articles", []) == []


@pytest.mark.asyncio
async def test_processor_deps():
    assert PaperProcessor().deps == ["candidate-ranker"]
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_processor.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Create `agents/processor.py`**

```python
"""PaperProcessor — fanout agent: one sub-agent per paper. Each does
fetch + extract + distill independently, in parallel.

Per-paper LLM failures are logged + dropped — they don't abort the run.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from ..distill.article import distill as distill_article
from ..llm.openai_compatible import LLMError
from ..pipeline import fetch_with_fallback
from ..vault.crosslink import load_index
from .base import Agent, Context


class _DistillOne:
    """Per-paper leaf agent — fetch + extract + distill one Paper."""

    def __init__(self, paper, idx, total, tmpdir, wiki_index):
        self.name = f"paper-processor[{idx + 1}/{total}]"
        self.deps: list[str] = []
        self._paper = paper
        self._tmpdir = tmpdir
        self._wiki_index = wiki_index

    async def run(self, ctx: Context) -> dict:
        try:
            full_text = await asyncio.to_thread(
                fetch_with_fallback, self._paper, ctx.cfg, self._tmpdir,
            )
            article = await asyncio.to_thread(
                distill_article, self._paper, full_text, self._wiki_index, ctx.llm,
            )
        except LLMError:
            if ctx.cfg.verbose:
                print(f"  distill failed for {self._paper.arxiv_id}")
            return {}
        existing = ctx.shared.get("articles", [])
        return {"articles": existing + [article]}


class PaperProcessor:
    """Fanout agent — produces N _DistillOne sub-agents at runtime."""
    name = "paper-processor"
    deps = ["candidate-ranker"]

    def expand(self, ctx: Context) -> list[Agent]:
        papers = ctx.shared.get("ranked", [])
        if not papers:
            return []
        tmpdir = Path(tempfile.mkdtemp(prefix="paper-distiller-"))
        wiki_index = load_index(ctx.vault)
        return [
            _DistillOne(p, i, len(papers), tmpdir, wiki_index)
            for i, p in enumerate(papers)
        ]
```

**Note**: there's a race-condition concern — multiple `_DistillOne` instances mutate `ctx.shared["articles"]` concurrently. The orchestrator's `_run_sub` does `ctx.shared.update(result or {})` which replaces the whole `"articles"` key. To avoid this, we read+append in each sub-agent's result. This is still racy if two sub-agents complete simultaneously. Fix in Step 4.

- [ ] **Step 4: Fix race by adding a lock**

Modify `_DistillOne.run` in `agents/processor.py` to use a lock for the `articles` accumulation. Update `agents/processor.py`:

Replace the file contents with:

```python
"""PaperProcessor — fanout agent: one sub-agent per paper. Each does
fetch + extract + distill independently, in parallel.

Per-paper LLM failures are logged + dropped — they don't abort the run.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from ..distill.article import distill as distill_article
from ..llm.openai_compatible import LLMError
from ..pipeline import fetch_with_fallback
from ..vault.crosslink import load_index
from .base import Agent, Context

# Module-level lock — fine because each run() invocation creates a fresh DAG
_articles_lock = asyncio.Lock()


class _DistillOne:
    def __init__(self, paper, idx, total, tmpdir, wiki_index):
        self.name = f"paper-processor[{idx + 1}/{total}]"
        self.deps: list[str] = []
        self._paper = paper
        self._tmpdir = tmpdir
        self._wiki_index = wiki_index

    async def run(self, ctx: Context) -> dict:
        try:
            full_text = await asyncio.to_thread(
                fetch_with_fallback, self._paper, ctx.cfg, self._tmpdir,
            )
            article = await asyncio.to_thread(
                distill_article, self._paper, full_text, self._wiki_index, ctx.llm,
            )
        except LLMError:
            if ctx.cfg.verbose:
                print(f"  distill failed for {self._paper.arxiv_id}")
            return {}
        async with _articles_lock:
            current = ctx.shared.get("articles", [])
            current.append(article)
            ctx.shared["articles"] = current
        return {}


class PaperProcessor:
    name = "paper-processor"
    deps = ["candidate-ranker"]

    def expand(self, ctx: Context) -> list[Agent]:
        papers = ctx.shared.get("ranked", [])
        if not papers:
            # initialize empty list so downstream agents always see the key
            ctx.shared["articles"] = []
            return []
        ctx.shared.setdefault("articles", [])
        tmpdir = Path(tempfile.mkdtemp(prefix="paper-distiller-"))
        wiki_index = load_index(ctx.vault)
        return [
            _DistillOne(p, i, len(papers), tmpdir, wiki_index)
            for i, p in enumerate(papers)
        ]
```

- [ ] **Step 5: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_processor.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **115 passed** (111 + 4).

- [ ] **Step 7: Commit**

```bash
git add src/paper_distiller/agents/processor.py tests/agents/test_processor.py
git commit -m "feat(agents): PaperProcessor fanout (parallel per-paper distill)"
```

---

## Task 9: VaultWriter + SurveyComposer agents

**Files:**
- Create: `src/paper_distiller/agents/writer.py`
- Create: `tests/agents/test_writer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_writer.py`:

```python
"""Tests for VaultWriter + SurveyComposer agents."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.writer import VaultWriter, SurveyComposer
from paper_distiller.distill.article import ArticleResult


def _article(slug):
    return ArticleResult(
        slug=slug, title=f"T-{slug}", body="b",
        tags=[], refs=[f"arxiv:{slug}"], depth="full-pdf",
    )


def _ctx(articles, **cfg_overrides):
    cfg = SimpleNamespace(
        min_papers_for_survey=2,
        topic="t", verbose=False,
        **cfg_overrides,
    )
    return Context(
        cfg=cfg, llm=MagicMock(), vault=MagicMock(),
        shared={"articles": articles},
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_vault_writer_calls_save_entry_per_article():
    arts = [_article("a"), _article("b")]
    ctx = _ctx(arts)
    ctx.vault.save_entry = MagicMock(side_effect=lambda **kw: {"slug": kw["slug"]})
    out = await VaultWriter().run(ctx)
    assert ctx.vault.save_entry.call_count == 2
    assert set(out["saved_slugs"]) == {"a", "b"}


@pytest.mark.asyncio
async def test_vault_writer_empty_articles_is_noop():
    ctx = _ctx([])
    ctx.vault.save_entry = MagicMock()
    out = await VaultWriter().run(ctx)
    ctx.vault.save_entry.assert_not_called()
    assert out["saved_slugs"] == []


@pytest.mark.asyncio
async def test_vault_writer_deps():
    assert VaultWriter().deps == ["paper-processor"]


@pytest.mark.asyncio
async def test_survey_composer_skipped_when_below_min(mocker):
    """fewer than min_papers_for_survey → skip (return None)."""
    fake_compose = mocker.patch("paper_distiller.agents.writer.compose_survey")
    ctx = _ctx([_article("a")])  # 1 article, min=2
    out = await SurveyComposer().run(ctx)
    fake_compose.assert_not_called()
    assert out["survey_slug"] is None


@pytest.mark.asyncio
async def test_survey_composer_runs_when_above_min(mocker):
    fake_compose = mocker.patch(
        "paper_distiller.agents.writer.compose_survey",
        return_value={"title": "S", "body": "...", "tags": ["t"], "slug": "s-1"},
    )
    arts = [_article("a"), _article("b"), _article("c")]
    ctx = _ctx(arts)
    ctx.vault.save_entry = MagicMock(side_effect=lambda **kw: {"slug": kw["slug"]})
    out = await SurveyComposer().run(ctx)
    fake_compose.assert_called_once()
    assert out["survey_slug"] == "s-1"


@pytest.mark.asyncio
async def test_survey_composer_deps():
    assert SurveyComposer().deps == ["vault-writer"]
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_writer.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Inspect existing save_entry + compose_survey signatures**

```bash
.venv\Scripts\python.exe -c "from paper_distiller.vault.store import VaultStore; help(VaultStore.save_entry)"
.venv\Scripts\python.exe -c "from paper_distiller.distill.survey import compose_survey; help(compose_survey)"
```

- [ ] **Step 4: Create `agents/writer.py`**

```python
"""VaultWriter — saves distilled articles to vault.
SurveyComposer — composes optional cross-article survey."""

from __future__ import annotations

import asyncio

from ..distill.survey import compose_survey
from .base import Context


class VaultWriter:
    name = "vault-writer"
    deps = ["paper-processor"]

    async def run(self, ctx: Context) -> dict:
        articles = ctx.shared.get("articles", [])
        saved = []
        for article in articles:
            await asyncio.to_thread(
                ctx.vault.save_entry,
                category="articles",
                **article.to_save_kwargs(),
            )
            saved.append(article.slug)
        return {"saved_slugs": saved}


class SurveyComposer:
    name = "survey-composer"
    deps = ["vault-writer"]

    async def run(self, ctx: Context) -> dict:
        articles = ctx.shared.get("articles", [])
        if len(articles) < ctx.cfg.min_papers_for_survey:
            return {"survey_slug": None}
        survey = await asyncio.to_thread(
            compose_survey, articles, ctx.cfg.topic or "", ctx.llm,
        )
        saved = await asyncio.to_thread(
            ctx.vault.save_entry,
            category="surveys",
            title=survey["title"],
            body=survey["body"],
            tags=survey.get("tags") or [],
            refs=[f"articles:{a.slug}" for a in articles],
            slug=survey.get("slug"),
        )
        return {"survey_slug": saved["slug"]}
```

- [ ] **Step 5: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_writer.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **121 passed** (115 + 6).

- [ ] **Step 7: Commit**

```bash
git add src/paper_distiller/agents/writer.py tests/agents/test_writer.py
git commit -m "feat(agents): VaultWriter + SurveyComposer (persistence phase)"
```

---

## Task 10: paper-distiller-chat entry + `distill` one-shot subcommand

**Files:**
- Create: `src/paper_distiller/chat/__init__.py`
- Create: `src/paper_distiller/chat/cli.py`
- Modify: `pyproject.toml` (add console script)
- Create: `tests/chat/__init__.py`
- Create: `tests/chat/test_distill_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/chat/__init__.py` (empty).

Create `tests/chat/test_distill_cli.py`:

```python
"""Tests for paper-distiller-chat 'distill' subcommand — end-to-end with mocks."""
from unittest.mock import MagicMock

import pytest


def test_chat_cli_parses_distill_args(monkeypatch):
    """build_parser exposes the distill subcommand with --topic / --n / --vault."""
    from paper_distiller.chat.cli import build_parser
    p = build_parser()
    args = p.parse_args(["distill", "--vault", "/tmp/v", "--topic", "X", "--n", "3"])
    assert args.subcommand == "distill"
    assert args.vault == "/tmp/v"
    assert args.topic == "X"
    assert args.n == 3


def test_chat_cli_dispatches_to_orchestrator(mocker, tmp_path, monkeypatch):
    """`paper-distiller-chat distill ...` builds DAG, runs Orchestrator,
    returns 0."""
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")

    # Patch the orchestrator's run to a no-op
    fake_run = mocker.patch(
        "paper_distiller.chat.cli.Orchestrator.run",
        return_value=MagicMock(),
    )
    # Avoid actually instantiating VaultStore / LLMClient
    mocker.patch("paper_distiller.chat.cli.VaultStore")
    mocker.patch("paper_distiller.chat.cli.LLMClient")

    from paper_distiller.chat.cli import main
    rc = main([
        "distill", "--vault", str(tmp_path), "--topic", "X", "--n", "1",
    ])
    assert rc == 0
    fake_run.assert_called_once()
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/chat/ -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Create `src/paper_distiller/chat/__init__.py`**

```python
"""Chat-first entry point for paper-distiller (v1.0+)."""
```

- [ ] **Step 4: Create `src/paper_distiller/chat/cli.py`**

```python
"""paper-distiller-chat entry point.

In Plan 1, supports only the one-shot `distill` subcommand. Plan 2 adds
`ask` + `resume`; Plan 3 adds the interactive REPL.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from rich.console import Console
from rich.live import Live

from ..agents.base import Context
from ..agents.curation import CandidateMerger, CandidateRanker
from ..agents.dag import DAG
from ..agents.orchestrator import Orchestrator, AgentFailed
from ..agents.processor import PaperProcessor
from ..agents.renderer import ConsoleRenderer
from ..agents.searchers import ArxivSearcher, SemanticScholarSearcher
from ..agents.writer import SurveyComposer, VaultWriter
from ..config import load_config
from ..llm.openai_compatible import LLMClient
from ..vault.store import VaultStore


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paper-distiller-chat",
        description="Chat-first paper distillation. Plan-1 subset: one-shot `distill`.",
    )
    sub = p.add_subparsers(dest="subcommand", required=True)

    distill = sub.add_parser("distill", help="Single-pass: search a topic, distill N papers")
    distill.add_argument("--vault", required=True)
    distill.add_argument("--topic", help="Search topic")
    distill.add_argument("--author", help="Search author (alternative to --topic)")
    distill.add_argument("--n", type=int, default=3, help="Articles to distill (default 3)")
    distill.add_argument("--pool", type=int, default=30, help="Search pool size (default 30)")
    distill.add_argument("--source", choices=["arxiv", "ss", "both"], default="both")
    distill.add_argument("--dry-run", action="store_true")
    distill.add_argument("--verbose", "-v", action="store_true")
    distill.add_argument("--model", help="Override PD_MODEL env var")
    distill.add_argument("--provider", help="Override PD_PROVIDER_NAME label")
    return p


def _build_single_pass_dag() -> DAG:
    return DAG([
        ArxivSearcher(),
        SemanticScholarSearcher(),
        CandidateMerger(),
        CandidateRanker(),
        PaperProcessor(),
        VaultWriter(),
        SurveyComposer(),
    ])


async def _run_distill(args) -> int:
    try:
        cfg = load_config(
            vault_path=args.vault,
            topic=args.topic,
            author=args.author,
            top_n=args.n,
            pool=args.pool,
            source=args.source,
            dry_run=args.dry_run,
            verbose=args.verbose,
            model_override=args.model,
            provider_override=args.provider,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if cfg.dry_run:
        print(f"[DRY-RUN] Would distill {cfg.top_n} papers on {cfg.topic!r}")
        return 0

    vault = VaultStore(cfg.vault_path)
    llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model)
    renderer = ConsoleRenderer(title=f"distill · {cfg.topic or cfg.author}")
    ctx = Context(cfg=cfg, llm=llm, vault=vault, shared={}, on_status=renderer.on_status)

    console = Console()
    dag = _build_single_pass_dag()
    orch = Orchestrator(dag, ctx)

    with Live(renderer.build_table(), refresh_per_second=10, console=console) as live:
        async def _refresher():
            while True:
                live.update(renderer.build_table())
                await asyncio.sleep(0.1)
        refresher_task = asyncio.create_task(_refresher())
        try:
            await orch.run()
        except AgentFailed as e:
            print(f"\nAgent {e.agent_name!r} failed: {e.__cause__}", file=sys.stderr)
            return 3
        finally:
            refresher_task.cancel()
            try:
                await refresher_task
            except asyncio.CancelledError:
                pass
            live.update(renderer.build_table())

    articles = ctx.shared.get("articles", [])
    survey_slug = ctx.shared.get("survey_slug")
    print()
    print(f"  Articles distilled: {len(articles)}")
    print(f"  Survey slug:        {survey_slug or '(none)'}")
    print(f"  Tokens in/out:      {llm.total_tokens_in} / {llm.total_tokens_out}")
    return 0


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.subcommand == "distill":
        return asyncio.run(_run_distill(args))
    return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Add console script entry to pyproject.toml**

Edit the `[project.scripts]` block in `pyproject.toml`:

Find:
```toml
[project.scripts]
paper-distiller = "paper_distiller.cli:main"
paper-distiller-qa = "paper_distiller.qa.cli:main"
```

Replace with:
```toml
[project.scripts]
paper-distiller = "paper_distiller.cli:main"
paper-distiller-qa = "paper_distiller.qa.cli:main"
paper-distiller-chat = "paper_distiller.chat.cli:main"
```

(Plan 3 will delete the first two; Plan 1 keeps both alive.)

- [ ] **Step 6: Reinstall editable to register new script**

```bash
.venv\Scripts\python.exe -m pip install -e . --no-deps --quiet
```

- [ ] **Step 7: Smoke-check help**

```bash
.venv\Scripts\paper-distiller-chat.exe --help
.venv\Scripts\paper-distiller-chat.exe distill --help
```

Expected: argparse help for both top-level and `distill` subcommand.

- [ ] **Step 8: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/chat/ -v
```

Expected: 2 passed.

- [ ] **Step 9: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **123 passed** (121 + 2).

- [ ] **Step 10: Commit**

```bash
git add src/paper_distiller/chat/__init__.py src/paper_distiller/chat/cli.py pyproject.toml tests/chat/__init__.py tests/chat/test_distill_cli.py
git commit -m "feat(chat): paper-distiller-chat entry + distill subcommand (one-shot)

First chat-CLI entry point. Plan-1 subset: only the 'distill' subcommand,
which runs the v1.0 single-pass agent DAG with a rich Live status table.
Existing paper-distiller and paper-distiller-qa script entries unchanged.
"
```

---

## Task 11: End-to-end integration test for `distill` subcommand

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_distill_e2e.py`

- [ ] **Step 1: Write the integration test**

Create `tests/integration/__init__.py` (empty).

Create `tests/integration/test_distill_e2e.py`:

```python
"""End-to-end integration test for paper-distiller-chat distill — all subsystems mocked."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper_distiller.distill.article import ArticleResult
from paper_distiller.sources.arxiv import Paper


def _paper(i):
    return Paper(
        source="arxiv", paper_id=f"2501.0000{i}", arxiv_id=f"2501.0000{i}",
        title=f"P{i}", authors=[], abstract=f"abstract {i}",
        pdf_url=f"https://x/{i}.pdf", published="2025-01-01", categories=[],
    )


def test_distill_e2e_writes_articles_to_vault(mocker, tmp_path, monkeypatch):
    """`paper-distiller-chat distill --vault tmp --topic X --n 2`
    should write 2 articles + 1 survey to the vault."""
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")

    # Mock source searches
    mocker.patch(
        "paper_distiller.agents.searchers.arxiv_search",
        return_value=[_paper(1), _paper(2), _paper(3)],
    )
    mocker.patch(
        "paper_distiller.agents.searchers.ss_search",
        return_value=[],
    )
    # Mock rank
    mocker.patch(
        "paper_distiller.agents.curation.rank",
        side_effect=lambda candidates, topic, top_n, llm: candidates[:top_n],
    )
    # Mock fetch + distill
    mocker.patch(
        "paper_distiller.agents.processor.fetch_with_fallback",
        return_value="x" * 600,
    )
    mocker.patch(
        "paper_distiller.agents.processor.distill_article",
        side_effect=lambda paper, full_text, wiki_index, llm: ArticleResult(
            slug=f"a-{paper.arxiv_id}", title=f"T-{paper.arxiv_id}",
            body=f"body {paper.arxiv_id}", tags=["t"], refs=[f"arxiv:{paper.arxiv_id}"],
            depth="full-pdf",
        ),
    )
    # Mock survey
    mocker.patch(
        "paper_distiller.agents.writer.compose_survey",
        return_value={"title": "S", "body": "...", "tags": ["s"], "slug": "session-survey-1"},
    )
    # Don't actually call LLM in load_index
    mocker.patch(
        "paper_distiller.agents.processor.load_index",
        return_value=MagicMock(slugs=lambda: set()),
    )

    vault = tmp_path / "vault"
    vault.mkdir()

    from paper_distiller.chat.cli import main
    rc = main(["distill", "--vault", str(vault), "--topic", "diffusion", "--n", "2"])
    assert rc == 0

    # Vault should have 2 articles + 1 survey
    articles_dir = vault / "articles"
    surveys_dir = vault / "surveys"
    assert articles_dir.exists()
    assert surveys_dir.exists()
    assert len(list(articles_dir.glob("*.md"))) == 2
    assert len(list(surveys_dir.glob("*.md"))) == 1
```

- [ ] **Step 2: Run, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/integration/test_distill_e2e.py -v
```

Expected: 1 passed. (If failing, the most likely culprit is a mismatch between what `_run_distill` expects and what was mocked — adjust accordingly.)

- [ ] **Step 3: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **124 passed** (123 + 1).

- [ ] **Step 4: Manual smoke (dry-run only — no API spend)**

```powershell
.\.venv\Scripts\paper-distiller-chat.exe distill --vault "G:\Math research Agent\wiki" --topic test --n 1 --dry-run
```

Expected: prints "[DRY-RUN] Would distill 1 papers on 'test'", exits 0.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/__init__.py tests/integration/test_distill_e2e.py
git commit -m "test(chat): end-to-end integration test for distill subcommand"
```

---

## Task 12: Plan-1 wrap-up checkpoint

- [ ] **Step 1: Final full suite + summary**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **124 passed** (78 baseline + 46 new).

- [ ] **Step 2: Verify both old CLIs still work**

```powershell
.\.venv\Scripts\paper-distiller.exe --help
.\.venv\Scripts\paper-distiller-qa.exe --help
.\.venv\Scripts\paper-distiller-chat.exe --help
.\.venv\Scripts\paper-distiller-chat.exe distill --help
```

Expected: all four show help without errors. Plan 1 is purely additive — no old behavior broken.

- [ ] **Step 3: Push to GitHub**

```bash
git push origin main
```

CI should run (workflow `.github/workflows/ci.yml`) and pass.

- [ ] **Step 4: Confirm CI green**

Open https://github.com/jesson-hh/paper-distiller/actions and verify the latest CI run is green (pytest matrix on Python 3.10 / 3.11 / 3.12).

If yellow/red on any matrix cell, fix locally and force-push the fix before moving on to Plan 2.

- [ ] **Step 5: Write a Plan-1 done marker (no commit needed)**

After Plan 1 is fully done, in the conversation say: "Plan 1 done — framework + 7 single-pass agents in place. Ready for Plan 2 (QA loop + REPL)."

The Plan 2 writing happens fresh in a new conversation using the design spec.

---

## Plan-1 success criteria

- [ ] All 12 tasks completed
- [ ] 124 tests passing (78 baseline + 46 new)
- [ ] `paper-distiller-chat distill --vault X --topic Y --n N` runs end-to-end with status table
- [ ] Old `paper-distiller` and `paper-distiller-qa` CLIs unchanged + still working
- [ ] CI green on Python 3.10 / 3.11 / 3.12
- [ ] No new top-level runtime dependencies beyond `rich`

Plan 2 (QA loop port + REPL) will be written separately based on lessons from Plan 1.
