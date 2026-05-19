"""ConsoleRenderer — receives status events, exposes a rich Table snapshot.

Live rendering (rich.live.Live) is wired up in the CLI layer; this module
only owns state + table construction.
"""

from __future__ import annotations

import time
from typing import Any

from rich.table import Table

from .base import Status


class ConsoleRenderer:
    def __init__(self, title: str = ""):
        self.title = title
        self._rows: dict[str, dict[str, Any]] = {}

    def on_status(self, name: str, status: Status, **kw) -> None:
        """Status event callback. Updates internal row state."""
        row = self._rows.setdefault(name, {
            "status": Status.QUEUED,
            "started_at": None,
            "elapsed": None,
            "error": None,
        })
        row["status"] = status
        if status == Status.RUNNING and row["started_at"] is None:
            row["started_at"] = time.monotonic()
        elif status in (Status.DONE, Status.FAILED, Status.SKIPPED):
            if row["started_at"] is not None:
                row["elapsed"] = time.monotonic() - row["started_at"]
            if "error" in kw:
                row["error"] = kw["error"]

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a copy of the current row state. For tests."""
        return {k: dict(v) for k, v in self._rows.items()}

    def build_table(self) -> Table:
        """Return a rich Table reflecting current state."""
        table = Table(title=self.title or None, show_header=True)
        table.add_column("Agent")
        table.add_column("Status")
        table.add_column("Elapsed")

        for name, row in self._rows.items():
            status_str = row["status"].value
            if row["elapsed"] is not None:
                elapsed_str = f"{row['elapsed']:.1f}s"
            elif row["started_at"] is not None:
                elapsed_str = f"{(time.monotonic() - row['started_at']):.1f}s"
            else:
                elapsed_str = "—"
            table.add_row(name, status_str, elapsed_str)
        return table
