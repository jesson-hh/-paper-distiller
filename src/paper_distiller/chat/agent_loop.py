"""Conversational agent loop — paper-distiller's chat brain.

The user types natural language; the loop invokes the LLM with function-calling
enabled (TOOL_SCHEMAS from agent_tools), executes any requested tool calls,
feeds results back, and continues until the LLM emits a plain-text reply for
the user. The loop is the single source of conversational state — it owns the
message history and the current vault_path.

Tools (search, distill_by_id, show, ask, research) are documented in the
system prompt and dispatched via execute_tool. Tool execution is synchronous
because each wrapper internally calls asyncio.run() — see agent_tools.py.

This module is the user-facing surface in v1.4. The pre-v1.4 slash-command
REPL is retained as the "legacy-repl" subcommand for users who prefer
explicit control.
"""

from __future__ import annotations

import json
import sys
from typing import Callable

from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

from ..llm.openai_compatible import LLMClient
from .agent_tools import TOOL_SCHEMAS, execute_tool


__all__ = ["AgentLoop", "DEFAULT_SYSTEM_PROMPT"]


DEFAULT_SYSTEM_PROMPT = """\
你是 paper-distiller —— 一个研究论文的对话式智能体。用户通过自然语言跟你交流，\
你负责理解意图、调用工具完成任务，并用简洁的中文回复结果。

你拥有 5 个工具：

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

## 工作原则

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
    # For oversize results (typically tool_show returning a long body),
    # truncate the body field and re-encode.
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

    def send(self, user_text: str) -> str:
        """Process one user turn. Returns the final assistant text reply.

        Loops over tool_calls / tool_results until the LLM emits a plain
        text response or the per-turn tool-call budget is exhausted.
        """
        self.messages.append({"role": "user", "content": user_text})

        for _ in range(self.max_tool_calls + 1):
            resp = self.llm.complete_with_tools(self.messages, TOOL_SCHEMAS)

            assistant_msg: dict = {"role": "assistant", "content": resp.text or ""}
            if resp.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in resp.tool_calls
                ]
            self.messages.append(assistant_msg)

            if not resp.tool_calls:
                return resp.text or ""

            for tc in resp.tool_calls:
                if self.on_tool_call is not None:
                    try:
                        self.on_tool_call(tc.name, tc.arguments)
                    except Exception:
                        pass
                result = execute_tool(
                    tc.name, tc.arguments, vault_path=self.vault_path
                )
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": _stringify_tool_result(result),
                    }
                )

        return "(达到单轮工具调用上限。如有需要请重新提问，或拆分成更小的步骤。)"

    def run(self) -> int:
        """Blocking interactive loop. Reads input(), prints response, EOF exits."""
        self.console.print(
            Rule("[bold]paper-distiller[/bold] · 对话式研究助手 (Ctrl-D 退出)")
        )
        self.console.print(f"[dim]vault: {self.vault_path}[/dim]\n")

        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]再见。[/dim]")
                return 0
            if not line:
                continue
            if line.lower() in (":q", ":quit", ":exit", "/exit", "/quit"):
                self.console.print("[dim]再见。[/dim]")
                return 0

            try:
                reply = self.send(line)
            except Exception as e:
                self.console.print(
                    f"[red]agent error:[/red] {type(e).__name__}: {e}",
                    style="red",
                )
                continue

            self.console.print()
            self.console.print(Rule("[cyan]paper-distiller[/cyan]"))
            if reply:
                self.console.print(Markdown(reply))
            else:
                self.console.print("[dim](no reply)[/dim]")
            self.console.print(
                f"[dim]tokens in/out: {self.llm.total_tokens_in} / "
                f"{self.llm.total_tokens_out}[/dim]\n"
            )
