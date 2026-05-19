"""Shared duration parser used by cli.py and agent_tools.py."""

from __future__ import annotations

import re


def parse_duration(s: str) -> int:
    """Parse '4h' / '30m' / '1h30m' / '3600s' → seconds."""
    m = re.match(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$", s.strip())
    if not m or not any(m.groups()):
        raise ValueError(f"invalid duration: {s!r}")
    h, mn, sc = (int(g or 0) for g in m.groups())
    total = h * 3600 + mn * 60 + sc
    if total < 60:
        raise ValueError(f"duration too short: {total}s (min 60s)")
    return total
