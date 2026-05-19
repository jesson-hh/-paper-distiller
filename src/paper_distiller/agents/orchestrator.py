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

    async def _run_sub(self, sub) -> None:
        self.ctx.on_status(sub.name, Status.RUNNING)
        try:
            result = await sub.run(self.ctx)
            self.ctx.shared.update(result or {})
            self.ctx.on_status(sub.name, Status.DONE)
        except Exception as e:
            self.ctx.on_status(sub.name, Status.FAILED, error=e)
            raise AgentFailed(sub.name) from e
