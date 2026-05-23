"""Async agent event stream driver.

Drives the LLM agent loop and yields SSE event dicts per the API contract:

    {"type": "text",            "delta": "..."}
    {"type": "tool_call_start", "id": "tc_abc", "name": "search", "args": {...}}
    {"type": "tool_call_done",  "id": "tc_abc", "result": {...}}
    {"type": "cost",            "tokens_in": N, "tokens_out": N, "cny": 0.12}
    {"type": "done",            "history": [...]}
    {"type": "error",           "message": "..."}

The server is stateless: client sends full history with each turn.
Long-running tools (distill, review) block via asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator

from .._pricing_helper import estimate_cost_event
from ..chat.agent_tools import TOOL_SCHEMAS, execute_tool
from ..llm.openai_compatible import LLMClient, StreamChunk

_MAX_TOOL_CALLS = 10  # per-turn safety cap


async def agent_event_stream(
    message: str,
    history: list[dict],
    vault_path: str,
    llm: LLMClient,
) -> AsyncGenerator[dict, None]:
    """Drive the LLM agent loop and yield SSE event dicts.

    Parameters
    ----------
    message:
        The new user message for this turn.
    history:
        Full conversation history *before* this turn (as sent by the client).
    vault_path:
        Path to the vault directory for tool execution.
    llm:
        Configured LLMClient instance (sync, wraps httpx).
    """
    # 1. Build working history by appending the new user message
    working: list[dict] = list(history) + [{"role": "user", "content": message}]

    tool_call_count = 0

    try:
        while True:
            # 2. Stream LLM response in a thread (blocking httpx call)
            text_buf = ""
            tool_calls_acc: dict[int, dict] = {}  # idx -> {id, name, args_str}
            current_idx: int = -1  # index of the tool call currently accumulating

            def _stream_llm():
                return list(llm.complete_with_tools_stream(working, TOOL_SCHEMAS))

            chunks: list[StreamChunk] = await asyncio.to_thread(_stream_llm)

            for chunk in chunks:
                if chunk.text_delta:
                    text_buf += chunk.text_delta
                    yield {"type": "text", "delta": chunk.text_delta}

                if chunk.tool_call_id or chunk.tool_name_delta or chunk.tool_arg_delta:
                    # Accumulate tool call fragments by index.
                    # A chunk carrying tool_call_id starts a new entry; subsequent
                    # name/arg deltas without an id append to current_idx.
                    if chunk.tool_call_id:
                        current_idx = len(tool_calls_acc)
                        tool_calls_acc[current_idx] = {
                            "id": chunk.tool_call_id,
                            "name": chunk.tool_name_delta or "",
                            "args_str": chunk.tool_arg_delta or "",
                        }
                    elif current_idx >= 0:
                        if chunk.tool_name_delta:
                            tool_calls_acc[current_idx]["name"] += chunk.tool_name_delta
                        if chunk.tool_arg_delta:
                            tool_calls_acc[current_idx]["args_str"] += chunk.tool_arg_delta

            # 3. Parse accumulated tool calls
            tool_calls: list[dict] = []
            for _idx, tc in sorted(tool_calls_acc.items()):
                try:
                    args = json.loads(tc["args_str"] or "{}")
                except (json.JSONDecodeError, ValueError):
                    args = {}
                tool_calls.append({"id": tc["id"], "name": tc["name"], "args": args})

            # 4. If no tool calls → assistant replied in plain text; we're done
            if not tool_calls:
                # Append assistant message to history
                working.append({"role": "assistant", "content": text_buf})
                # Emit cost
                try:
                    cost_event = estimate_cost_event(llm)
                except Exception:
                    cost_event = {
                        "type": "cost",
                        "tokens_in": llm.total_tokens_in,
                        "tokens_out": llm.total_tokens_out,
                        "cny": llm.estimated_cost_cny,
                    }
                yield cost_event
                yield {"type": "done", "history": working}
                return

            # 5. Append assistant message with tool_calls into history
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": text_buf or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    for tc in tool_calls
                ],
            }
            working.append(assistant_msg)

            # 6. Execute each tool
            for tc in tool_calls:
                if tool_call_count >= _MAX_TOOL_CALLS:
                    yield {
                        "type": "error",
                        "message": f"per-turn tool call cap ({_MAX_TOOL_CALLS}) reached",
                    }
                    yield {"type": "done", "history": working}
                    return

                tool_call_count += 1
                yield {
                    "type": "tool_call_start",
                    "id": tc["id"],
                    "name": tc["name"],
                    "args": tc["args"],
                }

                # Run synchronous tool in a thread to not block the event loop
                result = await asyncio.to_thread(
                    execute_tool,
                    tc["name"],
                    tc["args"],
                    vault_path=vault_path,
                )

                yield {
                    "type": "tool_call_done",
                    "id": tc["id"],
                    "result": result,
                }

                # Append tool result into history
                working.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result),
                })

            # 7. Loop back to call LLM again with the tool results

    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        yield {"type": "done", "history": working}
