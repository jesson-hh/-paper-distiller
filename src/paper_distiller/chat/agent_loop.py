"""Conversational agent loop — paper-distiller's chat brain.

The user types natural language; the loop invokes the LLM with function-calling
enabled (TOOL_SCHEMAS from agent_tools), executes any requested tool calls,
feeds results back, and continues until the LLM emits a plain-text reply for
the user. The loop is the single source of conversational state — it owns the
message history and the current vault_path.

v1.4: 5 tools (search/distill_by_id/show/ask/research) + plain function-calling
v1.5: + ask_user (6th tool) + streaming SSE + slash commands + plan mode +
      Ctrl-C abort + session-wide cost display.

Tool execution is synchronous because each wrapper internally calls
asyncio.run() — see agent_tools.py.
"""

from __future__ import annotations

import json
import sys
from typing import Callable

from rich.console import Console
from rich.rule import Rule

from ..llm.openai_compatible import LLMClient
from .agent_tools import TOOL_SCHEMAS, execute_tool
from .cost_estimator import estimate_tool_cost_cny
from .plan_mode import should_show_plan, confirm_plan


__all__ = ["AgentLoop", "DEFAULT_SYSTEM_PROMPT"]


DEFAULT_SYSTEM_PROMPT = """\
你是 paper-distiller —— 一个研究论文的对话式智能体。用户通过自然语言跟你交流，\
你负责理解意图、调用工具完成任务，并用简洁的中文回复结果。

你拥有 6 个工具：

1. **search(topic, n=10, source="all")** — 在 arxiv + Semantic Scholar + OpenAlex \
并行搜索，返回排序后的候选论文（含 id/title/authors/year/abstract/pdf_url）。\
不下载、不蒸馏，只用来给用户预览。

2. **distill_by_id(ids, topic=...)** — 根据 ID 列表下载并蒸馏论文，存入 vault。\
**强烈建议**同时传 `topic`（用上一次 search 用过的 query），否则匹配率会下降。\
返回 distilled 列表 + survey_slug + matched_count / unmatched。

3. **show(slug, category="articles")** — 读取 vault 中已保存的条目，返回 markdown body。

4. **ask(question, max_rounds=3, per_round=2, max_cost_cny=5.0, max_articles=10)** \
— 多轮 QA 循环：搜索 → 蒸馏 → 反思，直到回答了问题或耗尽预算。返回会话摘要。

5. **research(question, duration="2h", max_papers=20, max_cost_cny=15.0)** — \
长时自主深度研究模式（5 阶段循环），产出 ~20 篇蒸馏文章 + 主题综述 + 最终报告。\
默认 2 小时；用户明确说"深度研究"或"长时跑"时再用，普通问题用 ask。

6. **ask_user(question, options=[{label, description}, ...], multi_select=False)** — \
**关键判断模糊时**调用：让用户从 2-4 个选项中挑。比如 search 返回 10 篇候选你不知该蒸馏哪几篇、\
或者预算紧时让用户确认走 ask 还是 research。**不要**用于琐碎确认，agent 自己能决策的就别问。

## 工作原则

- **遇真模糊调用 ask_user**：决定明显该由用户做（选哪些论文、是否提高预算、方向有歧义）就暂停问。\
不要用于"我应该继续吗"这种你能自决的问题。
- **自己判断预算**：用户已授权你自主决定 max_cost_cny / max_rounds 等参数。\
默认值通常够用；只在用户明确要求"省点"或"放开"时才调整。
- **优先 search → distill_by_id 的两步法**：对于"帮我找几篇关于 X 的论文"这种请求，\
先 search 给用户看摘要，再让用户挑（或者你自己挑 top-N）调 distill_by_id。\
直接对一个具体问题用 ask 也是合理的。
- **distill_by_id 必带 topic**：用上一轮 search 的 topic，否则容易 matched_count=0。
- **简洁回复**：工具返回结果后，用一两段话总结要点；不要把整个 JSON 复述给用户。\
list 类返回值（candidates、distilled）适合用 markdown 编号列表展示。
- **失败不慌**：工具返回 `{"error": "..."}` 时，先解释原因，再问用户怎么办或自己重试。
- **中文为主**：除论文标题、术语、ID 外，全部用中文。

vault 路径已由系统配置好，工具会自动使用，你不用关心它。
"""


def _stringify_tool_result(result: dict, max_chars: int = 8000) -> str:
    """Compact JSON encoding for tool results. Truncates body fields if huge."""
    s = json.dumps(result, ensure_ascii=False, default=str)
    if len(s) <= max_chars:
        return s
    if isinstance(result, dict) and "body" in result and isinstance(result["body"], str):
        truncated = dict(result)
        keep = max_chars - 500
        truncated["body"] = result["body"][:keep] + "\n\n[…body truncated…]"
        s = json.dumps(truncated, ensure_ascii=False, default=str)
        if len(s) <= max_chars:
            return s
    return s[: max_chars - 30] + '..."[truncated]"}'


class AgentLoop:
    """Stateful conversation loop with function-calling enabled.

    Hold one of these per chat session. `send(user_text)` processes one user
    turn (which may involve multiple tool calls) and returns the final
    assistant text. `run()` is the blocking interactive REPL.
    """

    def __init__(
        self,
        llm: LLMClient,
        vault_path: str,
        system_prompt: str | None = None,
        max_tool_calls_per_turn: int = 10,
        console: Console | None = None,
        on_tool_call: Callable[[str, dict], None] | None = None,
    ):
        self.llm = llm
        self.vault_path = vault_path
        self.max_tool_calls = max_tool_calls_per_turn
        self.console = console or Console()
        self.on_tool_call = on_tool_call
        self.messages: list = [
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT}
        ]
        # v1.5
        self.auto_mode = False
        self._abort = None  # set by run() when interactive

    def send(self, user_text: str) -> str:
        """Process one user turn. Returns the final assistant text reply.

        Loops over tool_calls / tool_results until the LLM emits a plain
        text response or the per-turn tool-call budget is exhausted.
        """
        self.messages.append({"role": "user", "content": user_text})

        for _ in range(self.max_tool_calls + 1):
            text_buf, tool_calls = self._stream_one_response()

            assistant_msg: dict = {"role": "assistant", "content": text_buf}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in tool_calls
                ]
            self.messages.append(assistant_msg)

            if not tool_calls:
                return text_buf

            for tc in tool_calls:
                self._execute_one_tool_call(tc)

        return "(达到单轮工具调用上限。如有需要请重新提问，或拆分成更小的步骤。)"

    def _stream_one_response(self):
        """Drive one LLM call. Returns (final_text, list_of_ToolCall)."""
        from ..llm.openai_compatible import ToolCall

        # Fallback path for LLMs that only implement complete_with_tools.
        if not hasattr(self.llm, "complete_with_tools_stream"):
            resp = self.llm.complete_with_tools(self.messages, TOOL_SCHEMAS)
            if resp.text:
                self._render_text_delta(resp.text)
                self._end_text_render()
            return resp.text or "", list(resp.tool_calls)

        text_pieces: list[str] = []
        partial: dict[str, dict] = {}
        order: list[str] = []
        current_call_id: str | None = None

        for chunk in self.llm.complete_with_tools_stream(self.messages, TOOL_SCHEMAS):
            if chunk.text_delta:
                text_pieces.append(chunk.text_delta)
                self._render_text_delta(chunk.text_delta)
            if chunk.tool_call_id:
                current_call_id = chunk.tool_call_id
                if current_call_id not in partial:
                    partial[current_call_id] = {"name": "", "arguments": ""}
                    order.append(current_call_id)
            if chunk.tool_name_delta:
                if current_call_id is None:
                    if not order:
                        current_call_id = "__implicit__"
                        order.append(current_call_id)
                        partial[current_call_id] = {"name": "", "arguments": ""}
                    else:
                        current_call_id = order[-1]
                partial[current_call_id]["name"] += chunk.tool_name_delta
            if chunk.tool_arg_delta:
                cid = current_call_id or (order[-1] if order else None)
                if cid is None:
                    continue
                partial[cid]["arguments"] += chunk.tool_arg_delta
            if chunk.finish_reason:
                break

        if text_pieces:
            self._end_text_render()

        tool_calls: list = []
        for cid in order:
            p = partial[cid]
            try:
                args = json.loads(p["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                id=cid if cid != "__implicit__" else "",
                name=p["name"],
                arguments=args,
            ))

        return "".join(text_pieces), tool_calls

    def _render_text_delta(self, delta: str) -> None:
        """Print one text chunk during streaming (no newline)."""
        try:
            self.console.print(delta, end="", soft_wrap=True, highlight=False)
        except Exception:
            sys.stdout.write(delta)
            sys.stdout.flush()

    def _end_text_render(self) -> None:
        """Finish a streamed text block — print trailing newline."""
        self.console.print()

    def _execute_one_tool_call(self, tc) -> None:
        """Run plan-mode check, execute tool, append result to history."""
        if (
            not getattr(self, "auto_mode", False)
            and should_show_plan(tc.name, tc.arguments)
        ):
            est = estimate_tool_cost_cny(tc.name, tc.arguments)
            if not confirm_plan(tc.name, tc.arguments, estimated_cost_cny=est):
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(
                        {"cancelled": True, "by": "user", "phase": "plan"},
                        ensure_ascii=False,
                    ),
                })
                return

        if self.on_tool_call is not None:
            try:
                self.on_tool_call(tc.name, tc.arguments)
            except Exception:
                pass
        result = execute_tool(tc.name, tc.arguments, vault_path=self.vault_path)
        self.messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": _stringify_tool_result(result),
        })

    def run(self) -> int:
        """Blocking interactive loop. Streaming + slash + plan-mode aware."""
        from .abort import AbortController, install_handler
        from .slash_commands import parse_slash, dispatch_slash, EXIT_SIGNAL

        self.console.print(
            Rule(
                "[bold]paper-distiller[/bold] · 对话式研究助手  ·  "
                "/help for commands  ·  /exit to quit"
            )
        )
        self.console.print(f"[dim]vault: {self.vault_path}[/dim]\n")

        self._abort = AbortController()
        uninstall = install_handler(self._abort)

        try:
            while True:
                if self._abort.exit_requested():
                    self.console.print("\n[dim]再见。[/dim]")
                    return 0
                self._abort.reset()

                try:
                    line = input("you> ").strip()
                except EOFError:
                    self.console.print("\n[dim]再见。[/dim]")
                    return 0
                except KeyboardInterrupt:
                    self.console.print()
                    continue

                if not line:
                    continue

                parsed = parse_slash(line)
                if parsed is not None:
                    name, args = parsed
                    out = dispatch_slash(name, args, self)
                    if out == EXIT_SIGNAL:
                        self.console.print("[dim]再见。[/dim]")
                        return 0
                    self.console.print(out)
                    continue

                try:
                    self.console.print()
                    self.console.print(Rule("[cyan]paper-distiller[/cyan]"))
                    reply = self.send(line)
                    if not reply.strip():
                        self.console.print("[dim](no reply)[/dim]")
                except KeyboardInterrupt:
                    self.console.print("\n[yellow](当前工具已中止)[/yellow]")
                    self.messages.append({
                        "role": "user",
                        "content": "[system: 用户用 Ctrl-C 中止了上一步操作。请继续对话或建议下一步。]",
                    })
                except Exception as e:
                    self.console.print(
                        f"[red]agent error:[/red] {type(e).__name__}: {e}"
                    )
                    continue

                self.console.print(
                    f"\n[dim]tokens in/out: {self.llm.total_tokens_in:,} / "
                    f"{self.llm.total_tokens_out:,}  ·  "
                    f"¥{self.llm.estimated_cost_cny:.4f}[/dim]\n"
                )
        finally:
            uninstall()
