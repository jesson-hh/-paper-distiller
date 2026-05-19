"""Tests for chat.agent_loop — conversation loop with function-calling."""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubLLM:
    """Stateful LLM stub: emit a scripted sequence of ToolCallResponse objects
    one per call to complete_with_tools."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.last_messages = None
        self.last_tools = None

    def complete_with_tools(self, messages, tools, temperature=0.5):
        self.calls += 1
        self.last_messages = list(messages)
        self.last_tools = list(tools)
        if not self._responses:
            raise AssertionError("StubLLM ran out of scripted responses")
        return self._responses.pop(0)


def _make_response(text="", tool_calls=None, finish_reason=""):
    from paper_distiller.llm.openai_compatible import ToolCallResponse
    return ToolCallResponse(
        text=text,
        tool_calls=list(tool_calls or []),
        finish_reason=finish_reason,
    )


def _make_tool_call(name, arguments, call_id="call-1"):
    from paper_distiller.llm.openai_compatible import ToolCall
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))


# ---------------------------------------------------------------------------
# Plain-text response (no tool calls)
# ---------------------------------------------------------------------------

def test_send_returns_text_when_no_tool_calls(tmp_path):
    from paper_distiller.chat.agent_loop import AgentLoop

    llm = _StubLLM([_make_response(text="你好！我是 paper-distiller。")])
    loop = AgentLoop(llm=llm, vault_path=str(tmp_path))

    reply = loop.send("hi")

    assert reply == "你好！我是 paper-distiller。"
    assert llm.calls == 1
    # Messages: system + user + assistant
    assert len(loop.messages) == 3
    assert loop.messages[0]["role"] == "system"
    assert loop.messages[1] == {"role": "user", "content": "hi"}
    assert loop.messages[2]["role"] == "assistant"
    assert loop.messages[2]["content"] == "你好！我是 paper-distiller。"
    assert "tool_calls" not in loop.messages[2]


# ---------------------------------------------------------------------------
# Tool call → result → text response
# ---------------------------------------------------------------------------

def test_send_dispatches_tool_call_and_returns_final_text(mocker, tmp_path):
    from paper_distiller.chat.agent_loop import AgentLoop

    # Round 1: LLM emits a search tool call.
    # Round 2: LLM responds with text after seeing tool result.
    llm = _StubLLM([
        _make_response(
            text="",
            tool_calls=[_make_tool_call("search", {"topic": "diffusion", "n": 3})],
        ),
        _make_response(text="找到 1 篇相关论文：扩散模型 A。"),
    ])

    fake_result = {"candidates": [{"id": "1.0", "title": "Diffusion A"}]}
    mocker.patch(
        "paper_distiller.chat.agent_loop.execute_tool",
        return_value=fake_result,
    )

    loop = AgentLoop(llm=llm, vault_path=str(tmp_path))
    reply = loop.send("帮我找几篇 diffusion 的论文")

    assert reply == "找到 1 篇相关论文：扩散模型 A。"
    assert llm.calls == 2

    # Messages should be: system, user, assistant(tool_calls), tool, assistant(text)
    roles = [m["role"] for m in loop.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]

    # Tool message must carry the call_id and serialized result.
    tool_msg = loop.messages[3]
    assert tool_msg["tool_call_id"] == "call-1"
    assert json.loads(tool_msg["content"]) == fake_result

    # Assistant tool-call message must be in OpenAI format.
    asst_msg = loop.messages[2]
    assert asst_msg["tool_calls"][0]["function"]["name"] == "search"
    args = json.loads(asst_msg["tool_calls"][0]["function"]["arguments"])
    assert args == {"topic": "diffusion", "n": 3}


def test_send_passes_vault_path_to_execute_tool(mocker, tmp_path):
    from paper_distiller.chat.agent_loop import AgentLoop

    llm = _StubLLM([
        _make_response(
            tool_calls=[_make_tool_call("show", {"slug": "x"})],
        ),
        _make_response(text="done"),
    ])

    fake_exec = mocker.patch(
        "paper_distiller.chat.agent_loop.execute_tool",
        return_value={"slug": "x"},
    )

    loop = AgentLoop(llm=llm, vault_path=str(tmp_path))
    loop.send("show x")

    fake_exec.assert_called_once_with("show", {"slug": "x"}, vault_path=str(tmp_path))


# ---------------------------------------------------------------------------
# Multiple sequential tool calls
# ---------------------------------------------------------------------------

def test_send_handles_multiple_tool_call_rounds(mocker, tmp_path):
    """Round 1: search. Round 2: distill_by_id. Round 3: text reply."""
    from paper_distiller.chat.agent_loop import AgentLoop

    llm = _StubLLM([
        _make_response(
            tool_calls=[_make_tool_call("search", {"topic": "x"}, call_id="c1")],
        ),
        _make_response(
            tool_calls=[
                _make_tool_call("distill_by_id", {"ids": ["1"], "topic": "x"}, call_id="c2")
            ],
        ),
        _make_response(text="蒸馏完成"),
    ])

    results = [
        {"candidates": [{"id": "1", "title": "T"}]},
        {"distilled": [{"slug": "t", "title": "T", "category": "articles"}],
         "matched_count": 1, "requested_count": 1, "survey_slug": None},
    ]
    exec_mock = mocker.patch(
        "paper_distiller.chat.agent_loop.execute_tool",
        side_effect=results,
    )

    loop = AgentLoop(llm=llm, vault_path=str(tmp_path))
    reply = loop.send("帮我搜索并蒸馏一篇关于 x 的论文")

    assert reply == "蒸馏完成"
    assert llm.calls == 3
    assert exec_mock.call_count == 2
    assert exec_mock.call_args_list[0].args == ("search", {"topic": "x"})
    assert exec_mock.call_args_list[1].args == ("distill_by_id", {"ids": ["1"], "topic": "x"})


def test_send_handles_parallel_tool_calls_in_one_response(mocker, tmp_path):
    """Single LLM response with TWO tool_calls — both must be executed and
    both tool results appended before the next LLM call."""
    from paper_distiller.chat.agent_loop import AgentLoop

    llm = _StubLLM([
        _make_response(
            tool_calls=[
                _make_tool_call("show", {"slug": "a"}, call_id="c1"),
                _make_tool_call("show", {"slug": "b"}, call_id="c2"),
            ],
        ),
        _make_response(text="两个都读完了"),
    ])

    mocker.patch(
        "paper_distiller.chat.agent_loop.execute_tool",
        side_effect=[{"slug": "a", "body": "..."}, {"slug": "b", "body": "..."}],
    )

    loop = AgentLoop(llm=llm, vault_path=str(tmp_path))
    reply = loop.send("show a 和 b")
    assert reply == "两个都读完了"

    # System + user + assistant(2 tool_calls) + tool(c1) + tool(c2) + assistant(text)
    assert len(loop.messages) == 6
    assert loop.messages[3]["tool_call_id"] == "c1"
    assert loop.messages[4]["tool_call_id"] == "c2"


# ---------------------------------------------------------------------------
# Safety: max_tool_calls_per_turn
# ---------------------------------------------------------------------------

def test_send_enforces_max_tool_calls_per_turn(mocker, tmp_path):
    """If LLM keeps emitting tool_calls forever, loop must bail with a
    placeholder reply rather than spin infinitely."""
    from paper_distiller.chat.agent_loop import AgentLoop

    # Always emit a tool_call — never a final text.
    def _always_tool_call():
        while True:
            yield _make_response(
                tool_calls=[_make_tool_call("show", {"slug": "x"})],
            )

    gen = _always_tool_call()

    class _InfiniteLLM(_StubLLM):
        def __init__(self):
            super().__init__([])

        def complete_with_tools(self, messages, tools, temperature=0.5):
            self.calls += 1
            self.last_messages = list(messages)
            return next(gen)

    mocker.patch(
        "paper_distiller.chat.agent_loop.execute_tool",
        return_value={"slug": "x", "body": "..."},
    )

    llm = _InfiniteLLM()
    loop = AgentLoop(llm=llm, vault_path=str(tmp_path), max_tool_calls_per_turn=3)
    reply = loop.send("loop forever")

    # Should NOT have looped beyond max+1 LLM calls.
    assert llm.calls == 4  # 3 tool-call rounds + 1 final attempt
    assert "上限" in reply  # "达到单轮工具调用上限" message


# ---------------------------------------------------------------------------
# Conversation persistence across turns
# ---------------------------------------------------------------------------

def test_messages_persist_across_send_calls(tmp_path):
    from paper_distiller.chat.agent_loop import AgentLoop

    llm = _StubLLM([
        _make_response(text="第一轮回答"),
        _make_response(text="第二轮回答"),
    ])
    loop = AgentLoop(llm=llm, vault_path=str(tmp_path))

    loop.send("question 1")
    assert len(loop.messages) == 3  # system + user + assistant

    loop.send("question 2")
    assert len(loop.messages) == 5  # +user +assistant

    # Both user turns are in history.
    user_msgs = [m for m in loop.messages if m["role"] == "user"]
    assert [m["content"] for m in user_msgs] == ["question 1", "question 2"]


# ---------------------------------------------------------------------------
# on_tool_call callback
# ---------------------------------------------------------------------------

def test_on_tool_call_callback_fires(mocker, tmp_path):
    from paper_distiller.chat.agent_loop import AgentLoop

    llm = _StubLLM([
        _make_response(
            tool_calls=[_make_tool_call("show", {"slug": "x"})],
        ),
        _make_response(text="done"),
    ])
    mocker.patch(
        "paper_distiller.chat.agent_loop.execute_tool",
        return_value={"slug": "x"},
    )

    seen = []
    loop = AgentLoop(
        llm=llm,
        vault_path=str(tmp_path),
        on_tool_call=lambda name, args: seen.append((name, args)),
    )
    loop.send("show x")

    assert seen == [("show", {"slug": "x"})]


def test_on_tool_call_callback_exception_does_not_break_loop(mocker, tmp_path):
    """A failing on_tool_call must not abort the conversation."""
    from paper_distiller.chat.agent_loop import AgentLoop

    llm = _StubLLM([
        _make_response(
            tool_calls=[_make_tool_call("show", {"slug": "x"})],
        ),
        _make_response(text="done"),
    ])
    mocker.patch(
        "paper_distiller.chat.agent_loop.execute_tool",
        return_value={"slug": "x"},
    )

    def _bad_callback(name, args):
        raise RuntimeError("callback boom")

    loop = AgentLoop(llm=llm, vault_path=str(tmp_path), on_tool_call=_bad_callback)
    reply = loop.send("show x")  # must NOT raise
    assert reply == "done"


# ---------------------------------------------------------------------------
# Result stringification
# ---------------------------------------------------------------------------

def test_stringify_tool_result_round_trips_small_results():
    from paper_distiller.chat.agent_loop import _stringify_tool_result

    result = {"candidates": [{"id": "1", "title": "T"}], "n": 1}
    s = _stringify_tool_result(result)
    assert json.loads(s) == result


def test_stringify_tool_result_truncates_oversized_body():
    from paper_distiller.chat.agent_loop import _stringify_tool_result

    big_body = "a" * 50000
    result = {"slug": "big", "body": big_body, "title": "T"}
    s = _stringify_tool_result(result, max_chars=8000)
    assert len(s) <= 8000
    # Truncation marker must be present.
    assert "truncated" in s


def test_stringify_tool_result_chinese_not_escaped():
    """ensure_ascii=False so Chinese tool results stay readable in the LLM context."""
    from paper_distiller.chat.agent_loop import _stringify_tool_result

    result = {"text": "扩散模型"}
    s = _stringify_tool_result(result)
    assert "扩散模型" in s
    assert "\\u" not in s


# ---------------------------------------------------------------------------
# System prompt customization
# ---------------------------------------------------------------------------

def test_custom_system_prompt_is_used(tmp_path):
    from paper_distiller.chat.agent_loop import AgentLoop

    llm = _StubLLM([_make_response(text="ok")])
    loop = AgentLoop(
        llm=llm,
        vault_path=str(tmp_path),
        system_prompt="CUSTOM SYSTEM",
    )
    loop.send("hi")

    assert loop.messages[0] == {"role": "system", "content": "CUSTOM SYSTEM"}


def test_default_system_prompt_mentions_five_tools(tmp_path):
    """Default prompt should reference each of the 5 tool names so the LLM
    knows what's available."""
    from paper_distiller.chat.agent_loop import DEFAULT_SYSTEM_PROMPT

    for name in ("search", "distill_by_id", "show", "ask", "research"):
        assert name in DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Tools passed to LLM
# ---------------------------------------------------------------------------

def test_send_passes_tool_schemas_to_llm(tmp_path):
    from paper_distiller.chat.agent_loop import AgentLoop
    from paper_distiller.chat.agent_tools import TOOL_SCHEMAS

    llm = _StubLLM([_make_response(text="ok")])
    loop = AgentLoop(llm=llm, vault_path=str(tmp_path))
    loop.send("anything")

    # The stub captured the tools= argument.
    assert llm.last_tools is not None
    assert len(llm.last_tools) == 5
    names = [t["function"]["name"] for t in llm.last_tools]
    assert set(names) == {s["function"]["name"] for s in TOOL_SCHEMAS}
