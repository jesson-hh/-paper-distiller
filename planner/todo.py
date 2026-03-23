from dataclasses import dataclass, field
from typing import List, Literal

Status = Literal["pending", "in_progress", "done", "blocked"]

ICONS = {
    "pending": "[ ]",
    "in_progress": "[>]",
    "done": "[x]",
    "blocked": "[!]",
}


@dataclass
class Task:
    id: int
    description: str
    status: Status = "pending"
    result_summary: str = ""


class TodoManager:
    def __init__(self):
        self.tasks: List[Task] = []
        self._next_id = 1

    def add(self, description: str) -> Task:
        task = Task(id=self._next_id, description=description)
        self._next_id += 1
        self.tasks.append(task)
        return task

    def update(self, task_id: int, status: Status, result: str = "") -> Task | None:
        for t in self.tasks:
            if t.id == task_id:
                t.status = status
                t.result_summary = result
                return t
        return None

    def pending(self) -> List[Task]:
        return [t for t in self.tasks if t.status == "pending"]

    def reset(self):
        self.tasks = []
        self._next_id = 1

    def render(self) -> str:
        if not self.tasks:
            return "(no tasks)"
        lines = ["=== Research Plan ==="]
        for t in self.tasks:
            icon = ICONS.get(t.status, "[ ]")
            lines.append(f"{icon} #{t.id}: {t.description}")
            if t.result_summary:
                lines.append(f"    -> {t.result_summary[:80]}")
        return "\n".join(lines)

    def to_list(self) -> list:
        """Return tasks as list of dicts for UI display."""
        return [
            {
                "id": t.id,
                "description": t.description,
                "status": t.status,
                "result": t.result_summary,
            }
            for t in self.tasks
        ]
