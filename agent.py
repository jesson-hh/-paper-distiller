import os
import json
from dotenv import load_dotenv

from tools import TOOL_DEFINITIONS, dispatch_tool
from context.conversation import ConversationManager
from context.compressor import maybe_compress
from tracking import get_experiment_log
from llm import get_client

load_dotenv()

MAX_TOKENS = 8192

SYSTEM_PROMPT = """You are a Mathematical Research Agent with deep expertise across pure and applied mathematics.

You have six tools at your disposal:

1. **arxiv_search** - Search arxiv.org for current research papers in any math domain.
   Use this FIRST when exploring a new domain or research question.

2. **symbolic_compute** - Exact symbolic mathematics via SymPy.
   Use for derivatives, integrals, equation solving, factoring, limits, series expansions.
   Always prefer exact symbolic results over numerical approximations when possible.

3. **proof_assist** - Structured proof construction and analysis.
   Use to decompose theorems, suggest proof strategies, identify needed lemmas.

4. **run_code** - Execute Python for numerical experiments and visualization.
   Available: numpy, scipy, matplotlib, sympy, mpmath, networkx, pandas.
   For plots, print images as 'IMG:<base64>' to display inline.

5. **log_experiment** - Record significant research findings for tracking.
   Call this after each meaningful step: a literature search, proof attempt, computation, or experiment.
   This builds a research log for later report generation.

6. **generate_report** - Generate a structured research report from the session's findings.
   Call this at the end of a research investigation to produce a markdown/LaTeX report.

## Your Research Style

- When given a domain to explore: use arxiv_search to survey the landscape (2-3 searches), then synthesize
- **Always call log_experiment** after completing a significant research step
- Present findings in a structured way: active research threads, key open problems, notable recent papers
- For mathematical computations: show your work, present LaTeX results clearly
- For proofs: explain the key ideas before diving into formal steps
- For experiments: explain what you're testing and interpret results
- Always invite follow-up: "Would you like me to deep-dive into X?" or "I can compute Y if you're interested"

## Output Format

- Use markdown formatting with LaTeX for math (e.g., $\\int_0^\\infty e^{-x} dx = 1$)
- Use headers to organize long responses
- Cite paper titles and arxiv IDs when referencing literature
- Be precise but accessible: explain notation when introducing it

You are a research collaborator, not just a calculator. Connect ideas across papers, suggest
interesting directions, and flag when something is an open problem vs. a solved result."""


class MathResearchAgent:
    def __init__(self):
        self.client = get_client()
        self.experiment_log = get_experiment_log()
        self._active_loop = None
        self._current_domain = ""

    def stream_response(self, user_message: str, history: list, domain: str, max_papers: int):
        """
        Generator that yields (updated_history, scratchpad_text, images) tuples.
        Uses streaming API for real-time text display.
        """
        self._current_domain = domain
        conv = ConversationManager(history)

        system = SYSTEM_PROMPT
        if domain and domain not in ("", "custom (specify in chat)"):
            system += f"\n\n**Current research domain**: {domain}. Use max_results={max_papers} for arxiv searches."

        conv.add_user(user_message)
        scratchpad_lines = []
        all_images = []

        while True:
            messages = maybe_compress(conv.messages, threshold=80_000)

            # Stream the LLM response
            result_holder = {"blocks": [], "stop_reason": "end_turn"}
            base_history = conv.to_gradio_history()

            for partial_text in self.client.stream_chat(
                system=system,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                max_tokens=MAX_TOKENS,
                result_holder=result_holder,
            ):
                if partial_text:
                    streaming_history = base_history + [{"role": "assistant", "content": partial_text}]
                    yield streaming_history, "\n".join(scratchpad_lines), list(all_images)

            content_blocks = result_holder["blocks"]
            stop_reason = result_holder["stop_reason"]

            conv.add_assistant(content_blocks)
            yield conv.to_gradio_history(), "\n".join(scratchpad_lines), list(all_images)

            if stop_reason != "tool_use":
                break

            # Dispatch tool calls
            tool_results = []
            for b in content_blocks:
                if b.get("type") != "tool_use":
                    continue

                input_preview = json.dumps(b["input"], ensure_ascii=False)
                if len(input_preview) > 200:
                    input_preview = input_preview[:200] + "..."
                scratchpad_lines.append(f"\n[Tool] {b['name']}\n{input_preview}")
                yield conv.to_gradio_history(), "\n".join(scratchpad_lines), list(all_images)

                result = dispatch_tool(b["name"], b["input"])

                if b["name"] not in ("log_experiment", "generate_report"):
                    self.experiment_log.log_tool_call(
                        tool_name=b["name"],
                        tool_input=b["input"],
                        tool_result=result,
                        domain=self._current_domain,
                    )

                if b["name"] == "run_code" and isinstance(result, dict):
                    imgs = result.get("images", [])
                    all_images.extend(imgs)

                result_str = json.dumps(result, ensure_ascii=False, default=str)
                if len(result_str) > 300:
                    result_str = result_str[:300] + "..."
                scratchpad_lines.append(f"[Result] {result_str}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": b["id"],
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

                yield conv.to_gradio_history(), "\n".join(scratchpad_lines), list(all_images)

            conv.add_tool_results(tool_results)

        yield conv.to_gradio_history(), "\n".join(scratchpad_lines), list(all_images)

    def autonomous_research(self, goal: str, domain: str, max_iterations: int = 20, max_time: int = 600):
        """
        Generator for autonomous research mode.
        Yields (phase, status_msg, plan_text, history, scratchpad, images) tuples.
        """
        from autonomous.research_loop import AutonomousResearchLoop, ResearchBudget

        budget = ResearchBudget(
            max_iterations=max_iterations,
            max_time_seconds=max_time,
        )
        self._active_loop = AutonomousResearchLoop(self, budget)
        yield from self._active_loop.run(goal, domain)
        self._active_loop = None

    def stop_autonomous(self):
        if self._active_loop:
            self._active_loop.request_stop()
