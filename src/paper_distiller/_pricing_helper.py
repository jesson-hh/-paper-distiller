"""Internal helper: build a cost SSE event from an LLMClient instance."""

from __future__ import annotations

from .llm.openai_compatible import LLMClient


def estimate_cost_event(llm: LLMClient) -> dict:
    """Return a cost SSE event dict from the client's accumulated token counts."""
    return {
        "type": "cost",
        "tokens_in": llm.total_tokens_in,
        "tokens_out": llm.total_tokens_out,
        "cny": llm.estimated_cost_cny,
    }
