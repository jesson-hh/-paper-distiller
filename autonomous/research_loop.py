import time
import json
import re
from dataclasses import dataclass, field
from typing import Generator

from planner.todo import TodoManager
from tracking import get_experiment_log


@dataclass
class ResearchBudget:
    max_iterations: int = 20
    max_time_seconds: int = 600
    max_api_calls: int = 50


@dataclass
class LoopState:
    goal: str = ""
    iteration: int = 0
    api_calls: int = 0
    start_time: float = 0.0
    status: str = "idle"  # idle, planning, executing, evaluating, complete, stopped
    stop_requested: bool = False


class AutonomousResearchLoop:
    """
    Autonomous research loop inspired by Karpathy's autoresearch.
    Phases: PLANNING → EXECUTING (loop) → REPORTING
    """

    def __init__(self, agent, budget: ResearchBudget):
        self.agent = agent
        self.budget = budget
        self.state = LoopState()
        self.todo = TodoManager()
        self.experiment_log = get_experiment_log()

    def request_stop(self):
        self.state.stop_requested = True

    def run(self, goal: str, domain: str) -> Generator:
        """
        Main autonomous loop generator.
        Yields (phase, status_msg, plan_text, history, scratchpad, images) tuples.
        """
        self.state = LoopState(goal=goal, start_time=time.time())
        history = []
        scratchpad_lines = []
        all_images = []

        # ── Phase 1: PLANNING ──
        self.state.status = "planning"
        yield (
            "PLANNING",
            f"Creating research plan for: {goal}",
            self.todo.render(),
            history, "\n".join(scratchpad_lines), list(all_images),
        )

        planning_prompt = (
            f"I need you to create a research plan for this mathematical investigation:\n\n"
            f"**Goal**: {goal}\n\n"
            f"Break this into 4-8 concrete, sequential research tasks. For each task:\n"
            f"- State what to investigate\n"
            f"- Specify which tools to use (arxiv_search, symbolic_compute, proof_assist, run_code)\n"
            f"- State the expected outcome\n\n"
            f"Format each task as: `TASK N: [description]`\n"
            f"After listing tasks, begin executing TASK 1 immediately."
        )

        last_history = history
        last_scratchpad = ""
        last_images = list(all_images)

        for h, s, imgs in self.agent.stream_response(planning_prompt, history, domain, 8):
            last_history = h
            last_scratchpad = s
            last_images = imgs
            yield (
                "PLANNING", "Agent creating research plan...",
                self.todo.render(), h, s, imgs,
            )

        # Parse tasks from the agent's response
        self._parse_tasks_from_history(last_history)
        history = last_history
        scratchpad_lines = [last_scratchpad] if last_scratchpad else []
        all_images = last_images

        self.state.api_calls += 1

        yield (
            "PLANNING", f"Plan created: {len(self.todo.tasks)} tasks",
            self.todo.render(), history, "\n".join(scratchpad_lines), list(all_images),
        )

        # ── Phase 2: EXECUTING (loop) ──
        self.state.status = "executing"

        while self._within_budget():
            task = self._next_task()
            if not task:
                break

            self.state.iteration += 1
            self.todo.update(task.id, "in_progress")

            yield (
                "EXECUTING",
                f"Task #{task.id}: {task.description[:60]}",
                self.todo.render(),
                history, "\n".join(scratchpad_lines), list(all_images),
            )

            # Build context from previous results
            prev_results = self._previous_results_summary()
            task_prompt = (
                f"Continue the research investigation.\n\n"
                f"**Overall goal**: {goal}\n"
                f"**Current task** (#{task.id}): {task.description}\n\n"
                f"**Previous results**:\n{prev_results}\n\n"
                f"Execute this task thoroughly. Use the appropriate tools. "
                f"After completing, call `log_experiment` to record what you found. "
                f"If this approach doesn't work, explain why and suggest an alternative."
            )

            for h, s, imgs in self.agent.stream_response(task_prompt, history, domain, 8):
                history = h
                all_images = imgs
                yield (
                    "EXECUTING",
                    f"Working on task #{task.id}...",
                    self.todo.render(),
                    h, s, imgs,
                )

            self.state.api_calls += 1

            # ── Evaluate: did it work? ──
            self.state.status = "evaluating"

            eval_prompt = (
                f"Briefly evaluate the result of task #{task.id}: {task.description}\n"
                f"1. Was the task successful? (yes/no)\n"
                f"2. Key finding in one sentence\n"
                f"3. Should we adjust the remaining plan? If so, suggest a new task.\n"
                f"Format: SUCCESS: yes/no | FINDING: ... | ADJUST: yes/no | NEW_TASK: ..."
            )

            eval_history = history
            for h, s, imgs in self.agent.stream_response(eval_prompt, history, domain, 8):
                eval_history = h
                yield (
                    "EVALUATING",
                    f"Evaluating task #{task.id}...",
                    self.todo.render(),
                    h, s, imgs,
                )

            self.state.api_calls += 1
            history = eval_history

            # Parse evaluation from last assistant message
            success, finding, new_task = self._parse_evaluation(history)
            self.todo.update(task.id, "done", finding)

            if new_task:
                self.todo.add(new_task)

            self.state.status = "executing"

            yield (
                "EXECUTING",
                f"Task #{task.id} {'passed' if success else 'failed'}: {finding[:50]}",
                self.todo.render(),
                history, "\n".join(scratchpad_lines), list(all_images),
            )

        # ── Phase 3: REPORTING ──
        self.state.status = "complete"
        stop_reason = "budget exceeded" if not self._within_budget() else "all tasks complete"
        if self.state.stop_requested:
            stop_reason = "stopped by user"

        report_prompt = (
            f"The autonomous research session is now complete ({stop_reason}).\n\n"
            f"**Goal**: {goal}\n"
            f"**Tasks completed**: {self.todo.render()}\n\n"
            f"Please:\n"
            f"1. Provide a concise summary of all findings\n"
            f"2. List key results and open questions\n"
            f"3. Call `generate_report` to produce a full research report"
        )

        for h, s, imgs in self.agent.stream_response(report_prompt, history, domain, 8):
            yield ("COMPLETE", stop_reason, self.todo.render(), h, s, imgs)

        yield (
            "COMPLETE",
            f"Research complete: {stop_reason}",
            self.todo.render(),
            history, "\n".join(scratchpad_lines), list(all_images),
        )

    def _within_budget(self) -> bool:
        if self.state.stop_requested:
            return False
        elapsed = time.time() - self.state.start_time
        return (
            self.state.iteration < self.budget.max_iterations
            and elapsed < self.budget.max_time_seconds
            and self.state.api_calls < self.budget.max_api_calls
        )

    def _next_task(self):
        pending = self.todo.pending()
        return pending[0] if pending else None

    def _previous_results_summary(self) -> str:
        done = [t for t in self.todo.tasks if t.status == "done"]
        if not done:
            return "(no previous results)"
        return "\n".join(f"- Task #{t.id}: {t.description}\n  Result: {t.result_summary}" for t in done)

    def _parse_tasks_from_history(self, history: list):
        """Extract TASK N: descriptions from the last assistant message."""
        if not history:
            return
        last_msg = ""
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                last_msg = msg.get("content", "")
                break

        # Match "TASK N:" or numbered list items
        patterns = [
            r"TASK\s*\d+\s*:\s*(.+)",
            r"\d+\.\s+\*\*(.+?)\*\*",
            r"\d+\.\s+(.+?)(?:\n|$)",
        ]
        tasks_found = []
        for pattern in patterns:
            matches = re.findall(pattern, last_msg)
            if matches:
                tasks_found = matches
                break

        for desc in tasks_found[:8]:
            desc = desc.strip().rstrip(".")
            if len(desc) > 10:
                self.todo.add(desc)

        # Fallback: if no tasks parsed, create a generic one
        if not self.todo.tasks:
            self.todo.add(f"Investigate: {self.state.goal}")

    def _parse_evaluation(self, history: list) -> tuple:
        """Parse SUCCESS/FINDING/NEW_TASK from the last assistant message."""
        last_msg = ""
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                last_msg = msg.get("content", "")
                break

        success = "yes" in last_msg.lower().split("success")[-1][:20] if "success" in last_msg.lower() else True

        finding_match = re.search(r"FINDING:\s*(.+?)(?:\||$|\n)", last_msg)
        finding = finding_match.group(1).strip() if finding_match else last_msg[:100]

        new_task = None
        new_task_match = re.search(r"NEW_TASK:\s*(.+?)(?:\||$|\n)", last_msg)
        if new_task_match:
            task_text = new_task_match.group(1).strip()
            if task_text.lower() not in ("none", "n/a", "no", ""):
                new_task = task_text

        return success, finding, new_task
