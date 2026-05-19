"""UI primitives for the conversational agent — Claude Code-inspired styling.

Centralizes visual elements so AgentLoop stays focused on agent logic:
- Welcome banner
- Status line (model · tokens · cost · auto-mode)
- Tool-call cards (running / done states)
- Assistant message indent / bullet
- Slash-command output framing

Color palette:
- cyan: paper-distiller branding, tool calls
- yellow: warnings, plan mode
- green: completed / success
- dim: secondary info (cost, tokens, hints)
- red: errors
"""

from __future__ import annotations

import json

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


__all__ = [
    "BULLET",
    "PROMPT",
    "print_welcome_banner",
    "print_status_line",
    "print_tool_call_running",
    "print_tool_call_done",
    "print_assistant_bullet",
    "print_slash_output",
    "format_args_inline",
    "summarize_tool_result",
]


# Visual constants
BULLET = "●"
PROMPT = "❯"


def print_welcome_banner(
    console: Console,
    version: str,
    vault_path: str,
    model: str,
    auto_mode: bool = False,
) -> None:
    """Print the startup banner. Modeled on Claude Code's welcome panel."""
    title = Text()
    title.append("✻ ", style="bold cyan")
    title.append("paper-distiller ", style="bold")
    title.append(f"v{version}", style="dim")

    lines = [
        title,
        Text(),
        Text.from_markup("[dim]研究论文对话式智能体[/dim]"),
        Text(),
        Text.from_markup(f"[dim]vault   ·[/dim] {vault_path}"),
        Text.from_markup(f"[dim]model   ·[/dim] {model}"),
        Text.from_markup(
            f"[dim]mode    ·[/dim] {'auto (no plan previews)' if auto_mode else 'interactive'}"
        ),
        Text(),
        Text.from_markup(
            "[dim]提示：自然语言提问，"
            "[cyan]/help[/cyan] 查命令，"
            "[cyan]/exit[/cyan] 退出[/dim]"
        ),
    ]
    body = Text("\n").join(lines)
    console.print()
    console.print(Panel(
        body,
        border_style="cyan",
        padding=(1, 2),
        expand=False,
    ))
    console.print()


def print_status_line(
    console: Console,
    *,
    model: str,
    tokens_in: int,
    tokens_out: int,
    cost_cny: float,
    auto_mode: bool,
) -> None:
    """One-line dim status footer after each turn."""
    auto_chip = "[yellow]auto[/yellow]" if auto_mode else "[dim]auto:off[/dim]"
    console.print(
        f"[dim]{model}[/dim]  [dim]·[/dim]  "
        f"[dim]{tokens_in:,} ↑  {tokens_out:,} ↓[/dim]  [dim]·[/dim]  "
        f"[dim]¥{cost_cny:.4f}[/dim]  [dim]·[/dim]  {auto_chip}"
    )


def format_args_inline(arguments: dict, max_chars: int = 60) -> str:
    """Inline-format a tool-call's arguments for display.

    Like: `topic="diffusion", n=10, source="all"` — truncated if too long.
    """
    parts = []
    for k, v in (arguments or {}).items():
        if isinstance(v, str):
            s = f'{k}="{v}"' if len(v) <= 30 else f'{k}="{v[:27]}..."'
        elif isinstance(v, (list, tuple)):
            if len(v) <= 3:
                s = f"{k}={list(v)!r}"
            else:
                s = f"{k}=[{len(v)} items]"
        elif isinstance(v, dict):
            s = f"{k}={{{len(v)} fields}}"
        else:
            s = f"{k}={v}"
        parts.append(s)
    out = ", ".join(parts)
    if len(out) > max_chars:
        out = out[: max_chars - 3] + "..."
    return out


def print_tool_call_running(
    console: Console, name: str, arguments: dict
) -> None:
    """Print a tool-call card in the 'running' state.

    Like Claude Code's `⏺ Bash(...)` indicator that shows BEFORE completion.
    """
    args_str = format_args_inline(arguments)
    text = Text()
    text.append("⏺ ", style="bold cyan")
    text.append(name, style="bold cyan")
    text.append("(", style="dim")
    text.append(args_str, style="dim")
    text.append(")", style="dim")
    console.print(text)


def summarize_tool_result(name: str, result: dict) -> str:
    """One-line preview of a tool result. Shown after the tool finishes.

    Heuristic per tool — no `result` shape contract, so be defensive.
    """
    if not isinstance(result, dict):
        return ""
    if "error" in result:
        return f"[red]error[/red]: {result['error']}"
    if "cancelled" in result and result["cancelled"]:
        return "[yellow]cancelled by user[/yellow]"

    if name == "search":
        c = result.get("candidates") or []
        return f"{len(c)} candidates"
    if name == "distill_by_id":
        d = result.get("distilled") or []
        matched = result.get("matched_count", 0)
        requested = result.get("requested_count", 0)
        survey = result.get("survey_slug")
        suffix = f" · survey: [cyan]{survey}[/cyan]" if survey else ""
        return f"{len(d)} distilled ({matched}/{requested}){suffix}"
    if name == "show":
        title = result.get("title", "?")
        body_len = len(result.get("body", ""))
        return f'"{title}"  ·  {body_len} chars'
    if name == "ask":
        rounds = result.get("rounds_completed", "?")
        articles = result.get("articles_distilled_count", "?")
        cost = result.get("cost_cny", 0.0)
        return f"{rounds} rounds · {articles} articles · ¥{cost:.2f}"
    if name == "research":
        papers = result.get("papers_distilled_count", "?")
        themes = result.get("themes_count", "?")
        cost = result.get("total_cost_cny", 0.0)
        report = result.get("final_report_slug")
        suffix = f" · report: [cyan]{report}[/cyan]" if report else ""
        return f"{papers} papers · {themes} themes · ¥{cost:.2f}{suffix}"
    if name == "ask_user":
        sel = result.get("selected") or []
        if sel:
            return f"user picked: {', '.join(repr(s) for s in sel)}"
        return "user cancelled"
    return ""


def print_tool_call_done(
    console: Console,
    name: str,
    arguments: dict,
    result: dict,
    duration_sec: float | None = None,
) -> None:
    """Replace/follow the running card with a done indicator + summary."""
    args_str = format_args_inline(arguments, max_chars=40)
    summary = summarize_tool_result(name, result)
    is_error = isinstance(result, dict) and "error" in result
    icon_style = "bold red" if is_error else "bold green"

    text = Text()
    text.append(f"{BULLET} ", style=icon_style)
    text.append(name, style="bold cyan")
    text.append("(", style="dim")
    text.append(args_str, style="dim")
    text.append(")  ", style="dim")
    if summary:
        text.append("→ ", style="dim")
        text.append_text(Text.from_markup(summary))
    if duration_sec is not None and duration_sec >= 0.5:
        text.append(f"  [{duration_sec:.1f}s]", style="dim")
    console.print(text)


def print_assistant_bullet(console: Console) -> None:
    """Print the leading bullet that introduces an assistant text reply."""
    text = Text()
    text.append(f"{BULLET} ", style="bold cyan")
    console.print(text, end="")


def print_slash_output(console: Console, output: str) -> None:
    """Render a slash-command's text reply in a subtle indented block."""
    if not output:
        return
    # Use a left-bar style so it's visually distinct from agent output.
    console.print(Panel(
        output,
        border_style="dim",
        padding=(0, 1),
        expand=False,
    ))


def print_plan_intercept_banner(console: Console) -> None:
    """Short divider shown right before a plan-mode card."""
    console.print()
