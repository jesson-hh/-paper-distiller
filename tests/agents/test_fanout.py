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
