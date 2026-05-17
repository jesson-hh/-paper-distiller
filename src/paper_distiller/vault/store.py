"""Obsidian markdown vault CRUD.

Adapted from Math Research Agent's wiki_store.py with two structural changes:
  1. VaultStore takes a vault_path argument (no module-level WIKI_ROOT)
  2. No git auto-commit (the caller decides whether/when to commit)

Frontmatter serialization, CJK-aware slug, and [[wikilink]] extraction are
preserved verbatim. The TF-IDF search helpers from the original module are
intentionally not ported here.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

CATEGORIES = ("articles", "techniques", "directions", "open-problems", "authors", "surveys")

_SLUG_CLEAN = re.compile(r"[^a-z0-9\-]+")
_SLUG_DASH = re.compile(r"-{2,}")
_WIKI_LINK_RE = re.compile(r"\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]")


def slugify(title: str) -> str:
    """Produce a filesystem-safe slug. Falls back to hash for CJK titles."""
    norm = unicodedata.normalize("NFKD", title)
    ascii_part = norm.encode("ascii", "ignore").decode("ascii").lower().strip()
    ascii_part = ascii_part.replace(" ", "-")
    ascii_part = _SLUG_CLEAN.sub("", ascii_part)
    ascii_part = _SLUG_DASH.sub("-", ascii_part).strip("-")
    if len(ascii_part) >= 3:
        return ascii_part[:80]
    h = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
    return f"entry-{h}"


def _fm_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _fm_unescape(s: str) -> str:
    return s.replace('\\"', '"').replace("\\\\", "\\")


def _format_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_format_scalar(x) for x in v) + "]"
    return '"' + _fm_escape(str(v)) + '"'


def _parse_scalar(raw: str):
    raw = raw.strip()
    if not raw:
        return ""
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        items, buf, in_quote, esc = [], "", False, False
        for ch in inner:
            if esc:
                buf += ch; esc = False; continue
            if ch == "\\":
                esc = True; continue
            if ch == '"':
                in_quote = not in_quote; continue
            if ch == "," and not in_quote:
                items.append(_parse_scalar(buf)); buf = ""; continue
            buf += ch
        if buf.strip():
            items.append(_parse_scalar(buf))
        return items
    if raw.startswith('"') and raw.endswith('"'):
        return _fm_unescape(raw[1:-1])
    if raw in ("true", "false"):
        return raw == "true"
    try:
        return float(raw) if "." in raw else int(raw)
    except ValueError:
        return raw


def dump_frontmatter(meta: dict, body: str) -> str:
    lines = ["---"]
    for k, v in meta.items():
        lines.append(f"{k}: {_format_scalar(v)}")
    lines.append("---")
    lines.append("")
    lines.append(body.rstrip() + "\n")
    return "\n".join(lines)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.split("\n")
    if lines[0].strip() != "---":
        return {}, text
    meta, i = {}, 1
    while i < len(lines) and lines[i].strip() != "---":
        if ":" in lines[i]:
            k, _, v = lines[i].partition(":")
            meta[k.strip()] = _parse_scalar(v)
        i += 1
    if i >= len(lines):
        return {}, text
    body = "\n".join(lines[i + 1:])
    if body.startswith("\n"):
        body = body[1:]
    return meta, body


@dataclass
class Entry:
    slug: str
    category: str
    title: str
    tags: list
    refs: list
    links: list
    created: str
    updated: str
    body: str

    def to_dict(self) -> dict:
        return {
            "slug": self.slug, "category": self.category, "title": self.title,
            "tags": self.tags, "refs": self.refs, "links": self.links,
            "created": self.created, "updated": self.updated, "body": self.body,
        }


def _extract_links(body: str) -> list:
    return sorted(set(m.group(1).strip() for m in _WIKI_LINK_RE.finditer(body)))


class VaultStore:
    """Markdown CRUD against an Obsidian-style vault directory.

    The vault root contains one subdirectory per category. VaultStore creates
    them on construction if missing.
    """

    def __init__(self, vault_path: Path | str):
        self.root = Path(vault_path)
        self.root.mkdir(parents=True, exist_ok=True)
        for c in CATEGORIES:
            (self.root / c).mkdir(exist_ok=True)

    def _validate_category(self, category: str) -> None:
        if category not in CATEGORIES:
            raise ValueError(f"Invalid category: {category!r}. Must be one of {CATEGORIES}.")

    def _entry_path(self, category: str, slug: str) -> Path:
        self._validate_category(category)
        return self.root / category / f"{slug}.md"

    def save_entry(
        self,
        title: str,
        category: str,
        body: str,
        tags: list | None = None,
        refs: list | None = None,
        slug: str | None = None,
    ) -> dict:
        self._validate_category(category)
        if not title.strip():
            raise ValueError("title is required")
        if not body.strip():
            raise ValueError("body is required")
        slug = slug or slugify(title)
        path = self._entry_path(category, slug)
        now = datetime.now().isoformat(timespec="seconds")
        created = now
        if path.exists():
            existing_meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
            created = existing_meta.get("created", now)
        links = _extract_links(body)
        meta = {
            "title": title, "category": category, "slug": slug,
            "tags": tags or [], "refs": refs or [], "links": links,
            "created": created, "updated": now,
        }
        path.write_text(dump_frontmatter(meta, body), encoding="utf-8")
        return {**meta, "path": str(path).replace("\\", "/")}

    def read_entry(self, category: str, slug: str) -> Entry | None:
        path = self._entry_path(category, slug)
        if not path.exists():
            return None
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        return Entry(
            slug=meta.get("slug", slug), category=meta.get("category", category),
            title=meta.get("title", slug),
            tags=meta.get("tags") or [], refs=meta.get("refs") or [],
            links=meta.get("links") or [],
            created=meta.get("created", ""), updated=meta.get("updated", ""),
            body=body,
        )

    def slug_exists(self, category: str, slug: str) -> bool:
        return self._entry_path(category, slug).exists()

    def list_entries(self, category: str | None = None) -> list:
        cats = [category] if category else list(CATEGORIES)
        for c in cats:
            self._validate_category(c)
        items = []
        for c in cats:
            folder = self.root / c
            if not folder.exists():
                continue
            for f in folder.glob("*.md"):
                try:
                    meta, _ = parse_frontmatter(f.read_text(encoding="utf-8"))
                    items.append({
                        "slug": meta.get("slug", f.stem),
                        "category": meta.get("category", c),
                        "title": meta.get("title", f.stem),
                        "tags": meta.get("tags") or [],
                        "refs": meta.get("refs") or [],
                        "created": meta.get("created", ""),
                        "updated": meta.get("updated", ""),
                    })
                except Exception:
                    continue
        items.sort(key=lambda x: x.get("updated", ""), reverse=True)
        return items
