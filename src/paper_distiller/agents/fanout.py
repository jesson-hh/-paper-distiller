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
