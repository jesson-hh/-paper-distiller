"""REPL main class — input loop + dispatch.

dispatch_one(line) is testable without stdin; the run() method wires prompt_toolkit
+ stdin reading + dispatch in a loop.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from rich.console import Console

from ...agents.router import IntentRouter, RoutingError
from ...llm.openai_compatible import LLMClient
from .commands import KNOWN_COMMANDS, parse_slash, SlashError
from .helpers import (
    handle_agents, handle_help, handle_provider, handle_sessions,
    handle_show, handle_vault,
)


# Confirmation prompt is split out so tests can mock it.
def _confirm(prompt: str) -> bool:
    """Print prompt + read a Y/n response. Returns True if confirmed."""
    try:
        answer = input(prompt).strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False
    return answer in ("", "y", "yes")


_AGENT_DEFAULTS = {
    "ask": {"max_rounds": 3, "per_round": 2, "max_cost_cny": 5.0, "source": "both"},
    "distill": {"n": 3, "source": "both"},
    "research": {"max_papers": 30, "max_cost_cny": 30.0, "duration": "4h", "source": "both"},
}


def _prompt_slot(name: str, default):
    """Ask the user for one slot value with type-coerced default fallback.

    Returns the filled value, or raises ValueError on type mismatch,
    or raises KeyboardInterrupt if the user cancels.
    """
    default_str = f" (default {default})" if default is not None else ""
    user_input = input(f"  {name}{default_str}: ").strip()
    if not user_input:
        return default
    if isinstance(default, bool):
        return user_input.lower() in ("y", "yes", "true", "1")
    if isinstance(default, int) and not isinstance(default, bool):
        return int(user_input)
    if isinstance(default, float):
        return float(user_input)
    return user_input


def _format_proposal(parsed: dict) -> str:
    cmd = parsed["command"]
    params = parsed["params"]
    missing = parsed["missing_params"]
    lines = [f"[intent-router] Intent: {cmd}  | confidence {parsed.get('confidence', '?')}"]
    for k, v in params.items():
        lines.append(f"  {k}: {v}")
    if missing:
        defaults = _AGENT_DEFAULTS.get(cmd, {})
        applied = ", ".join(f"{k}={defaults.get(k, '?')}" for k in missing)
        lines.append(f"Missing: {missing}")
        lines.append(f"Apply defaults ({applied}) and run? [Y/n]")
    else:
        lines.append("Run? [Y/n]")
    return "\n".join(lines)


class REPL:
    def __init__(self, vault_path):
        self.vault_path = Path(vault_path)
        self.console = Console()
        self._llm = None  # lazy

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LLMClient(
                os.getenv("PD_API_KEY"),
                os.getenv("PD_BASE_URL"),
                os.getenv("PD_MODEL"),
            )
        return self._llm

    def dispatch_one(self, line):
        """Dispatch a single input line. Returns 'QUIT' to signal exit, else None."""
        line = (line or "").strip()
        if not line:
            return None
        if line.startswith("/"):
            return self._dispatch_slash(line)
        return self._dispatch_natural_language(line)

    def _dispatch_slash(self, line):
        try:
            parsed = parse_slash(line)
        except SlashError as e:
            print(f"Error: {e}")
            return None
        if parsed.name == "quit":
            return "QUIT"
        if parsed.name == "help":
            print(handle_help())
            return None
        if parsed.name == "vault":
            print(handle_vault(self.vault_path))
            return None
        if parsed.name == "sessions":
            print(handle_sessions(self.vault_path))
            return None
        if parsed.name == "provider":
            print(handle_provider())
            return None
        if parsed.name == "agents":
            print(handle_agents())
            return None
        if parsed.name == "show":
            if not parsed.args:
                print("Usage: /show <slug>")
                return None
            print(handle_show(self.vault_path, parsed.args[0]))
            return None
        # Action commands — defer to dispatch_action
        return self._dispatch_action(parsed.name, parsed.args)

    def _dispatch_action(self, name, args):
        """Run a slash action command by invoking cli.main with synthetic argv."""
        from ..cli import main as cli_main
        cli_argv = [name, "--vault", str(self.vault_path), *args]
        try:
            rc = cli_main(cli_argv)
        except SystemExit as e:
            print(f"  (cli exited with code {e.code})")
            return None
        if rc != 0:
            print(f"  (cli returned exit code {rc})")
        return None

    def _dispatch_natural_language(self, line):
        try:
            router = IntentRouter(llm=self.llm)
            parsed = router.classify(line)
        except RoutingError as e:
            print(f"Intent routing failed: {e}")
            return None

        cmd = parsed["command"]
        params = dict(parsed["params"])

        # Show what router extracted
        print(f"[intent-router] Intent: {cmd}  | confidence {parsed.get('confidence', '?')}")
        for k, v in params.items():
            print(f"  {k}: {v}")

        # Interactive slot-filling: ask the user for each missing param,
        # showing the default. Empty input → use default. Type-coerce.
        defaults = _AGENT_DEFAULTS.get(cmd, {})
        if parsed["missing_params"]:
            print()
            for param_name in parsed["missing_params"]:
                default = defaults.get(param_name)
                try:
                    value = _prompt_slot(param_name, default)
                except (KeyboardInterrupt, EOFError):
                    print("\n  (cancelled)")
                    return None
                except ValueError as e:
                    print(f"  Error: {e}. Cancelling.")
                    return None
                if value is not None:
                    params[param_name] = value

        # `show` is read-only — dispatch directly, NOT via cli.main
        if cmd == "show":
            slug = params.get("slug")
            if not slug:
                print("  (no slug provided)")
                return None
            print(handle_show(self.vault_path, slug))
            return None

        # Build argv, show preview, ask final confirm, dispatch
        argv = self._params_to_argv(cmd, params)
        preview = f"{cmd} --vault \"{self.vault_path}\" " + " ".join(argv[1:])
        print()
        print(f"→ {preview}")
        if not _confirm("Run? [Y/n]: "):
            print("  (cancelled)")
            return None
        return self._dispatch_action(cmd, argv[1:])

    def _params_to_argv(self, cmd, params):
        """Translate a {name: value} param dict into CLI args (after the subcommand name)."""
        argv = [cmd]
        if cmd == "distill":
            if "topic" in params:
                argv += ["--topic", str(params["topic"])]
            if "n" in params:
                argv += ["--n", str(params["n"])]
        elif cmd == "ask":
            if "question" in params:
                argv += ["--question", str(params["question"])]
            if "max_rounds" in params:
                argv += ["--max-rounds", str(params["max_rounds"])]
            if "per_round" in params:
                argv += ["--per-round", str(params["per_round"])]
            if "max_cost_cny" in params:
                argv += ["--max-cost-cny", str(params["max_cost_cny"])]
        elif cmd == "resume":
            if "session_id" in params:
                argv += ["--session-id", str(params["session_id"])]
        elif cmd == "research":
            if "question" in params:
                argv += ["--question", str(params["question"])]
            if "max_papers" in params:
                argv += ["--max-papers", str(params["max_papers"])]
            if "max_cost_cny" in params:
                argv += ["--max-cost-cny", str(params["max_cost_cny"])]
            if "duration" in params:
                argv += ["--duration", str(params["duration"])]
            if "source" in params:
                argv += ["--source", str(params["source"])]
        return argv

    def run(self):
        """Launch the interactive REPL. Returns 0 on clean exit."""
        self._print_banner()
        session = PromptSession(
            completer=WordCompleter(
                ["/" + c for c in KNOWN_COMMANDS],
                ignore_case=True,
            ),
        )
        while True:
            try:
                line = session.prompt("> ")
            except (EOFError, KeyboardInterrupt):
                print("  (bye)")
                return 0
            result = self.dispatch_one(line)
            if result == "QUIT":
                print("  (bye)")
                return 0

    def _print_banner(self):
        from ... import __version__
        from .helpers import handle_provider
        self.console.print("─" * 60)
        self.console.print(f"[bold]paper-distiller v{__version__}[/bold]")
        provider_lines = handle_provider().splitlines()
        # Print the Model: line (index 2)
        if len(provider_lines) > 2:
            self.console.print(provider_lines[2])
        self.console.print(f"Vault: {self.vault_path}")
        self.console.print("")
        self.console.print("Slash commands: /distill /ask /resume /sessions /vault /provider /agents /show /help /quit")
        self.console.print("Natural language: '帮我研究下扩散'")
        self.console.print("─" * 60)
