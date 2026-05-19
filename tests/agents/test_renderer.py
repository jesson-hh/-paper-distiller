"""Tests for ConsoleRenderer — status events accumulate into a table state."""
import time

import pytest

from paper_distiller.agents.base import Status
from paper_distiller.agents.renderer import ConsoleRenderer


def test_renderer_records_queued_state():
    r = ConsoleRenderer(title="Test")
    r.on_status("a", Status.QUEUED)
    snap = r.snapshot()
    assert snap["a"]["status"] == Status.QUEUED
    assert snap["a"]["elapsed"] is None


def test_renderer_running_records_start_time():
    r = ConsoleRenderer(title="Test")
    r.on_status("a", Status.RUNNING)
    snap = r.snapshot()
    assert snap["a"]["status"] == Status.RUNNING
    assert snap["a"]["started_at"] is not None


def test_renderer_done_records_elapsed():
    r = ConsoleRenderer(title="Test")
    r.on_status("a", Status.RUNNING)
    time.sleep(0.05)
    r.on_status("a", Status.DONE)
    snap = r.snapshot()
    assert snap["a"]["status"] == Status.DONE
    assert snap["a"]["elapsed"] >= 0.05


def test_renderer_failed_records_error():
    r = ConsoleRenderer(title="Test")
    r.on_status("a", Status.RUNNING)
    r.on_status("a", Status.FAILED, error=RuntimeError("boom"))
    snap = r.snapshot()
    assert snap["a"]["status"] == Status.FAILED
    assert "boom" in str(snap["a"]["error"])


def test_renderer_build_table_returns_rich_table():
    """build_table() returns something rich can render (no exceptions)."""
    from rich.table import Table
    r = ConsoleRenderer(title="Test")
    r.on_status("a", Status.RUNNING)
    r.on_status("a", Status.DONE)
    table = r.build_table()
    assert isinstance(table, Table)
