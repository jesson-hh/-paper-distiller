"""T1.3 — Chat stream + agent driver tests.

Uses a stub LLM (no real network) to verify the SSE event sequence:
    text → tool_call_start → tool_call_done → text → done
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from paper_distiller.llm.openai_compatible import StreamChunk
from paper_distiller.web.server import create_app
from paper_distiller.web.agent_stream import agent_event_stream


# ── Stub LLM helpers ─────────────────────────────────────────────────────────


class StubLLMTextOnly:
    """Stub that yields plain text (no tool calls)."""
    total_tokens_in = 10
    total_tokens_out = 20

    @property
    def estimated_cost_cny(self):
        return 0.01

    def complete_with_tools_stream(self, messages, tools, temperature=0.5):
        yield StreamChunk(text_delta="Hello ")
        yield StreamChunk(text_delta="world.")
        yield StreamChunk(finish_reason="stop")


class StubLLMOneToolCall:
    """Stub that yields one tool_call (search) then a plain text reply."""
    total_tokens_in = 50
    total_tokens_out = 80
    _call_count = 0

    @property
    def estimated_cost_cny(self):
        return 0.05

    def complete_with_tools_stream(self, messages, tools, temperature=0.5):
        self._call_count += 1
        if self._call_count == 1:
            # First LLM call: emit a tool_call for "search"
            yield StreamChunk(tool_call_id="tc_001", tool_name_delta="search", tool_arg_delta="")
            yield StreamChunk(tool_arg_delta='{"topic": "transformers"}')
            yield StreamChunk(finish_reason="tool_calls")
        else:
            # Second LLM call (after tool result): emit plain text
            yield StreamChunk(text_delta="Found results.")
            yield StreamChunk(finish_reason="stop")


class StubLLMTwoToolCalls:
    """Stub that yields two sequential tool calls, then a text reply."""
    _call_count = 0
    total_tokens_in = 80
    total_tokens_out = 120

    @property
    def estimated_cost_cny(self):
        return 0.08

    def complete_with_tools_stream(self, messages, tools, temperature=0.5):
        self._call_count += 1
        if self._call_count == 1:
            yield StreamChunk(tool_call_id="tc_A", tool_name_delta="search", tool_arg_delta='{"topic":"x"}')
            yield StreamChunk(finish_reason="tool_calls")
        elif self._call_count == 2:
            yield StreamChunk(tool_call_id="tc_B", tool_name_delta="search", tool_arg_delta='{"topic":"y"}')
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(text_delta="Done.")
            yield StreamChunk(finish_reason="stop")


class StubLLMParallelTwoToolCalls:
    """Stub that emits two parallel tool calls interleaved in ONE turn, then text.

    Simulates the OpenAI streaming format for parallel tool calls where chunks
    for both tool calls arrive interleaved by index (tc_P at index 0, tc_Q at
    index 1) with no id on subsequent arg-delta chunks.
    """
    _call_count = 0
    total_tokens_in = 100
    total_tokens_out = 150

    @property
    def estimated_cost_cny(self):
        return 0.10

    def complete_with_tools_stream(self, messages, tools, temperature=0.5):
        self._call_count += 1
        if self._call_count == 1:
            # First chunk for tool call 0 (carries id + partial name + partial args)
            yield StreamChunk(tool_call_id="tc_P", tool_name_delta="search", tool_arg_delta="")
            # Arg delta for tool call 0 (no id → appends to current_idx = 0)
            yield StreamChunk(tool_arg_delta='{"topic":"parallel_x"}')
            # First chunk for tool call 1 (new id → current_idx becomes 1)
            yield StreamChunk(tool_call_id="tc_Q", tool_name_delta="search", tool_arg_delta="")
            # Arg delta for tool call 1 (no id → appends to current_idx = 1)
            yield StreamChunk(tool_arg_delta='{"topic":"parallel_y"}')
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(text_delta="Both done.")
            yield StreamChunk(finish_reason="stop")


# ── Canned execute_tool result ────────────────────────────────────────────────

_CANNED_SEARCH_RESULT = {"candidates": [{"id": "1234.5678", "title": "Test Paper", "authors": [], "year": "2023", "abstract": "...", "pdf_url": ""}]}


def _mock_execute_tool(name, args, *, vault_path):
    return _CANNED_SEARCH_RESULT


# ── Async helpers ─────────────────────────────────────────────────────────────

async def _collect_events(coro_or_gen) -> list[dict]:
    events = []
    async for ev in coro_or_gen:
        events.append(ev)
    return events


# ── Tests: agent_event_stream directly ───────────────────────────────────────

class TestAgentEventStreamTextOnly:
    @pytest.mark.asyncio
    async def test_emits_text_delta(self):
        llm = StubLLMTextOnly()
        with patch("paper_distiller.web.agent_stream.execute_tool", _mock_execute_tool):
            events = await _collect_events(agent_event_stream("hello", [], "/tmp/vault", llm))
        text_events = [e for e in events if e["type"] == "text"]
        assert len(text_events) == 2
        assert text_events[0]["delta"] == "Hello "
        assert text_events[1]["delta"] == "world."

    @pytest.mark.asyncio
    async def test_emits_cost_and_done(self):
        llm = StubLLMTextOnly()
        with patch("paper_distiller.web.agent_stream.execute_tool", _mock_execute_tool):
            events = await _collect_events(agent_event_stream("hello", [], "/tmp/vault", llm))
        types = [e["type"] for e in events]
        assert "cost" in types
        assert "done" in types
        # done must be last
        assert types[-1] == "done"

    @pytest.mark.asyncio
    async def test_done_event_has_history(self):
        llm = StubLLMTextOnly()
        with patch("paper_distiller.web.agent_stream.execute_tool", _mock_execute_tool):
            events = await _collect_events(agent_event_stream("ping", [], "/tmp/vault", llm))
        done = next(e for e in events if e["type"] == "done")
        assert "history" in done
        hist = done["history"]
        assert hist[0]["role"] == "user"
        assert hist[0]["content"] == "ping"
        assert hist[-1]["role"] == "assistant"


class TestAgentEventStreamWithToolCall:
    @pytest.mark.asyncio
    async def test_emits_tool_call_start_and_done(self):
        llm = StubLLMOneToolCall()
        with patch("paper_distiller.web.agent_stream.execute_tool", _mock_execute_tool):
            events = await _collect_events(agent_event_stream("search for x", [], "/tmp/vault", llm))
        types = [e["type"] for e in events]
        assert "tool_call_start" in types
        assert "tool_call_done" in types

    @pytest.mark.asyncio
    async def test_tool_call_start_has_name_and_args(self):
        llm = StubLLMOneToolCall()
        with patch("paper_distiller.web.agent_stream.execute_tool", _mock_execute_tool):
            events = await _collect_events(agent_event_stream("search", [], "/tmp/vault", llm))
        starts = [e for e in events if e["type"] == "tool_call_start"]
        assert len(starts) == 1
        assert starts[0]["name"] == "search"
        assert starts[0]["id"] == "tc_001"
        assert "args" in starts[0]

    @pytest.mark.asyncio
    async def test_tool_call_done_has_result(self):
        llm = StubLLMOneToolCall()
        with patch("paper_distiller.web.agent_stream.execute_tool", _mock_execute_tool):
            events = await _collect_events(agent_event_stream("search", [], "/tmp/vault", llm))
        dones = [e for e in events if e["type"] == "tool_call_done"]
        assert len(dones) == 1
        assert dones[0]["id"] == "tc_001"
        assert "result" in dones[0]
        assert dones[0]["result"] == _CANNED_SEARCH_RESULT

    @pytest.mark.asyncio
    async def test_event_sequence_order(self):
        """Sequence must be: (text? →) tool_call_start → tool_call_done → text → cost → done."""
        llm = StubLLMOneToolCall()
        with patch("paper_distiller.web.agent_stream.execute_tool", _mock_execute_tool):
            events = await _collect_events(agent_event_stream("search", [], "/tmp/vault", llm))
        types = [e["type"] for e in events]
        start_idx = types.index("tool_call_start")
        done_idx = types.index("tool_call_done")
        text_after = next((i for i, t in enumerate(types) if t == "text" and i > done_idx), None)
        final_done = types[-1]
        assert start_idx < done_idx
        assert text_after is not None
        assert final_done == "done"

    @pytest.mark.asyncio
    async def test_history_contains_tool_messages(self):
        llm = StubLLMOneToolCall()
        with patch("paper_distiller.web.agent_stream.execute_tool", _mock_execute_tool):
            events = await _collect_events(agent_event_stream("search", [], "/tmp/vault", llm))
        done = next(e for e in events if e["type"] == "done")
        hist = done["history"]
        roles = [m["role"] for m in hist]
        assert "tool" in roles

    @pytest.mark.asyncio
    async def test_prepends_history(self):
        """Prior history should be prepended, new user msg appended."""
        llm = StubLLMTextOnly()
        prior = [{"role": "user", "content": "prior"}, {"role": "assistant", "content": "ok"}]
        with patch("paper_distiller.web.agent_stream.execute_tool", _mock_execute_tool):
            events = await _collect_events(agent_event_stream("new", prior, "/tmp/vault", llm))
        done = next(e for e in events if e["type"] == "done")
        hist = done["history"]
        assert hist[0]["content"] == "prior"
        assert hist[2]["content"] == "new"


class TestAgentEventStreamSafetyCap:
    @pytest.mark.asyncio
    async def test_safety_cap_produces_error_after_10(self):
        """A stub that always returns tool_calls should hit the 10-call cap."""

        class InfiniteToolCallStub:
            total_tokens_in = 5
            total_tokens_out = 5
            estimated_cost_cny = 0.0

            def complete_with_tools_stream(self, messages, tools, temperature=0.5):
                yield StreamChunk(tool_call_id="tc_x", tool_name_delta="search", tool_arg_delta='{"topic":"x"}')
                yield StreamChunk(finish_reason="tool_calls")

        llm = InfiniteToolCallStub()
        with patch("paper_distiller.web.agent_stream.execute_tool", _mock_execute_tool):
            events = await _collect_events(agent_event_stream("go", [], "/tmp/vault", llm))
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) >= 1
        assert "cap" in error_events[0]["message"].lower() or "cap" in error_events[0]["message"]


class TestAgentEventStreamError:
    @pytest.mark.asyncio
    async def test_exception_in_llm_produces_error_event(self):
        """If LLM raises, the stream should emit an error event."""

        class BoomLLM:
            total_tokens_in = 0
            total_tokens_out = 0
            estimated_cost_cny = 0.0

            def complete_with_tools_stream(self, messages, tools, temperature=0.5):
                raise RuntimeError("simulated network error")
                yield  # make it a generator

        llm = BoomLLM()
        with patch("paper_distiller.web.agent_stream.execute_tool", _mock_execute_tool):
            events = await _collect_events(agent_event_stream("hello", [], "/tmp/vault", llm))
        types = [e["type"] for e in events]
        assert "error" in types
        assert "done" in types


# ── I4: parallel tool-call accumulation ──────────────────────────────────────


class TestParallelToolCallAccumulation:
    """I4 — two parallel tool calls interleaved in one turn must each get their
    own args intact (not corrupted by max() index logic)."""

    @pytest.mark.asyncio
    async def test_two_parallel_tool_calls_args_intact(self):
        llm = StubLLMParallelTwoToolCalls()
        with patch("paper_distiller.web.agent_stream.execute_tool", _mock_execute_tool):
            events = await _collect_events(
                agent_event_stream("run both", [], "/tmp/vault", llm)
            )

        starts = [e for e in events if e["type"] == "tool_call_start"]
        assert len(starts) == 2, f"expected 2 tool_call_start events, got {starts}"

        # Each tool call must have its own distinct args
        args_by_id = {e["id"]: e["args"] for e in starts}
        assert args_by_id["tc_P"] == {"topic": "parallel_x"}, f"tc_P args wrong: {args_by_id['tc_P']}"
        assert args_by_id["tc_Q"] == {"topic": "parallel_y"}, f"tc_Q args wrong: {args_by_id['tc_Q']}"

    @pytest.mark.asyncio
    async def test_two_parallel_tool_calls_done_events(self):
        llm = StubLLMParallelTwoToolCalls()
        with patch("paper_distiller.web.agent_stream.execute_tool", _mock_execute_tool):
            events = await _collect_events(
                agent_event_stream("run both", [], "/tmp/vault", llm)
            )

        dones = [e for e in events if e["type"] == "tool_call_done"]
        assert len(dones) == 2
        done_ids = {e["id"] for e in dones}
        assert done_ids == {"tc_P", "tc_Q"}


# ── Tests: POST /chat/stream endpoint ────────────────────────────────────────

class TestChatStreamEndpoint:
    @pytest.fixture
    def client(self, tmp_path):
        app = create_app(str(tmp_path))
        return TestClient(app, raise_server_exceptions=False)

    def test_no_env_returns_error_stream(self, client, monkeypatch):
        """Without PD_API_KEY set, endpoint should return an SSE error."""
        monkeypatch.delenv("PD_API_KEY", raising=False)
        monkeypatch.delenv("PD_BASE_URL", raising=False)
        monkeypatch.delenv("PD_MODEL", raising=False)

        r = client.post(
            "/chat/stream",
            json={"message": "hello", "history": [], "vault_path": "/tmp/vault"},
        )
        assert r.status_code == 200
        body = r.text
        assert "error" in body

    def test_returns_text_event_stream_content_type(self, client, monkeypatch, tmp_path):
        """The response Content-Type must be text/event-stream."""
        monkeypatch.setenv("PD_API_KEY", "fake")
        monkeypatch.setenv("PD_BASE_URL", "http://fake.local")
        monkeypatch.setenv("PD_MODEL", "fake-model")

        stub = StubLLMTextOnly()

        async def _fake_stream(message, history, vault_path, llm):
            yield {"type": "text", "delta": "hi"}
            yield {"type": "cost", "tokens_in": 1, "tokens_out": 1, "cny": 0.0}
            yield {"type": "done", "history": []}

        # Patch agent_event_stream at the module level where it's imported
        with patch("paper_distiller.web.routes.chat.agent_event_stream", _fake_stream):
            with patch("paper_distiller.web.routes.chat.LLMClient", lambda *a, **kw: stub):
                r = client.post(
                    "/chat/stream",
                    json={"message": "hi", "history": [], "vault_path": str(tmp_path)},
                )
        assert "text/event-stream" in r.headers.get("content-type", "")
