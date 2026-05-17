"""Vault index loader — provides the LLM with a list of existing entries
so it can weave [[slug]] crosslinks into newly written articles.
"""

from dataclasses import dataclass

from .store import VaultStore


@dataclass
class WikiIndex:
    entries: list  # [{category, slug, title, tags}, ...]

    def slugs(self) -> set:
        return {e["slug"] for e in self.entries}

    def to_prompt_lines(self) -> list:
        """Compact lines for stuffing into an LLM prompt."""
        return [
            f"- [[{e['slug']}]] ({e['category']}): {e['title']}" +
            (f" — tags: {', '.join(e['tags'])}" if e["tags"] else "")
            for e in self.entries
        ]


def load_index(store: VaultStore) -> WikiIndex:
    """Pre-load all vault entry metadata for crosslink suggestion."""
    return WikiIndex(entries=store.list_entries())
