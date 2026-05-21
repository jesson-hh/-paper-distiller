"""SQLite + FTS5 store for theorems / techniques extracted during distillation."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = 1


# Used by retrieve_by_text_match to filter noise tokens before FTS5 OR query
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at",
    "for", "with", "from", "by", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "should", "could", "may", "might", "must", "shall", "can",
    "this", "that", "these", "those", "we", "our", "their", "they", "it",
    "its", "such", "more", "most", "less", "very", "much", "many", "any",
    "all", "some", "no", "not", "only", "than", "then", "so", "also",
    "we", "paper", "abstract", "introduction", "conclusion",
    "method", "methods", "results", "discussion", "section",
    "show", "shows", "shown", "showing", "obtain", "obtained", "provide",
    "novel", "new", "based", "using", "use", "used", "approach",
    "model", "models", "data", "result", "study", "studies",
}


@dataclass
class Theorem:
    """One theorem / proposition extracted from a paper."""
    paper_arxiv_id: str
    paper_slug: str | None
    name: str
    statement: str
    proof_sketch: str
    techniques_used: list  # canonical technique names

    # Filled in by the store on insert
    id: int | None = None
    created_at: str | None = None


@dataclass
class Technique:
    """Canonical name for a math technique / inequality / framework."""
    name: str  # canonical short form, e.g. "Hölder"
    description: str = ""
    first_seen_arxiv_id: str | None = None


@dataclass
class ProofSidecar:
    """Sidecar JSON shape produced by the article distiller."""
    theorems: list = field(default_factory=list)         # list[dict]
    key_definitions: list = field(default_factory=list)  # list[dict]
    key_techniques: list = field(default_factory=list)   # list[str]

    @classmethod
    def from_json(cls, raw: dict) -> "ProofSidecar":
        if not isinstance(raw, dict):
            return cls()
        return cls(
            theorems=list(raw.get("theorems") or []),
            key_definitions=list(raw.get("key_definitions") or []),
            key_techniques=list(raw.get("key_techniques") or []),
        )

    def to_json(self) -> dict:
        return {
            "theorems": self.theorems,
            "key_definitions": self.key_definitions,
            "key_techniques": self.key_techniques,
        }


_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS theorems (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_arxiv_id  TEXT NOT NULL,
  paper_slug      TEXT,
  name            TEXT NOT NULL,
  statement       TEXT NOT NULL,
  proof_sketch    TEXT,
  techniques_used TEXT NOT NULL,
  created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_theorems_paper ON theorems(paper_arxiv_id);

CREATE VIRTUAL TABLE IF NOT EXISTS theorems_fts USING fts5(
  name, statement, proof_sketch,
  content='theorems',
  content_rowid='id',
  tokenize='porter unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS theorems_ai AFTER INSERT ON theorems BEGIN
  INSERT INTO theorems_fts(rowid, name, statement, proof_sketch)
  VALUES (new.id, new.name, new.statement, new.proof_sketch);
END;

CREATE TRIGGER IF NOT EXISTS theorems_ad AFTER DELETE ON theorems BEGIN
  INSERT INTO theorems_fts(theorems_fts, rowid, name, statement, proof_sketch)
  VALUES('delete', old.id, old.name, old.statement, old.proof_sketch);
END;

CREATE TABLE IF NOT EXISTS techniques (
  name                  TEXT PRIMARY KEY,
  description           TEXT NOT NULL DEFAULT '',
  first_seen_arxiv_id   TEXT
);

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


class ProofStore:
    """Per-vault SQLite store of extracted theorems + techniques.

    Concurrent reads + single writer are safe under WAL.
    `check_same_thread=False` because the distillation pipeline uses
    asyncio.to_thread which moves work to a worker thread.
    """

    def __init__(self, db_path: Path | str):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_sidecar(
        self,
        sidecar: ProofSidecar,
        paper_arxiv_id: str,
        paper_slug: str | None = None,
    ) -> dict:
        """Insert all theorems + register all techniques from one paper.

        Idempotent at paper-grain: re-ingesting the same paper deletes its
        prior theorems and re-inserts (so re-distilling a paper updates
        cleanly). Techniques are upsert (first_seen_arxiv_id sticks).
        """
        # Wipe prior rows for this paper
        self._conn.execute(
            "DELETE FROM theorems WHERE paper_arxiv_id = ?",
            (paper_arxiv_id,),
        )

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        n_theorems = 0
        for t in sidecar.theorems:
            if not isinstance(t, dict):
                continue
            name = (t.get("name") or "").strip()
            statement = (t.get("statement") or "").strip()
            if not name or not statement:
                continue
            techniques_used = t.get("techniques_used") or []
            if not isinstance(techniques_used, list):
                techniques_used = []
            self._conn.execute(
                """INSERT INTO theorems
                   (paper_arxiv_id, paper_slug, name, statement,
                    proof_sketch, techniques_used, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    paper_arxiv_id, paper_slug, name, statement,
                    (t.get("proof_sketch") or "").strip(),
                    json.dumps(techniques_used, ensure_ascii=False),
                    now,
                ),
            )
            n_theorems += 1

        # Upsert techniques — both from `key_techniques` and per-theorem
        all_techniques: set[str] = set()
        for name in sidecar.key_techniques:
            if isinstance(name, str) and name.strip():
                all_techniques.add(name.strip())
        for t in sidecar.theorems:
            if isinstance(t, dict):
                for name in (t.get("techniques_used") or []):
                    if isinstance(name, str) and name.strip():
                        all_techniques.add(name.strip())

        n_new_techniques = 0
        for name in all_techniques:
            cur = self._conn.execute(
                "SELECT 1 FROM techniques WHERE name = ?", (name,),
            )
            if cur.fetchone() is None:
                self._conn.execute(
                    "INSERT INTO techniques(name, first_seen_arxiv_id) VALUES (?, ?)",
                    (name, paper_arxiv_id),
                )
                n_new_techniques += 1

        self._conn.commit()
        return {
            "theorems_inserted": n_theorems,
            "techniques_new": n_new_techniques,
            "techniques_total_referenced": len(all_techniques),
        }

    # ------------------------------------------------------------------
    # Stats / inspection
    # ------------------------------------------------------------------

    def theorem_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM theorems").fetchone()[0]

    def technique_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM techniques").fetchone()[0]

    def paper_count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(DISTINCT paper_arxiv_id) FROM theorems"
        ).fetchone()[0]

    # ------------------------------------------------------------------
    # Retrieval — used by the distiller and by the agent's tools
    # ------------------------------------------------------------------

    def theorems_using_technique(
        self, technique_name: str, limit: int = 10,
    ) -> list[Theorem]:
        """All theorems whose techniques_used JSON contains `technique_name`."""
        if not technique_name.strip():
            return []
        needle = f'%"{technique_name.strip()}"%'
        rows = self._conn.execute(
            """SELECT * FROM theorems
               WHERE techniques_used LIKE ? COLLATE NOCASE
               ORDER BY id DESC
               LIMIT ?""",
            (needle, limit),
        ).fetchall()
        return [self._row_to_theorem(r) for r in rows]

    def search_theorems(self, query: str, limit: int = 10) -> list[Theorem]:
        """FTS5 search over theorem statement + proof_sketch + name."""
        if not query.strip():
            return []
        # Quote each token for safety
        tokens = ['"' + tok.replace('"', '') + '"' for tok in query.split() if tok]
        fts_query = " ".join(tokens)
        rows = self._conn.execute(
            """SELECT t.*, bm25(theorems_fts) AS score
               FROM theorems t
               JOIN theorems_fts ON theorems_fts.rowid = t.id
               WHERE theorems_fts MATCH ?
               ORDER BY score
               LIMIT ?""",
            (fts_query, limit),
        ).fetchall()
        return [self._row_to_theorem(r) for r in rows]

    def theorems_by_paper(self, paper_arxiv_id: str) -> list[Theorem]:
        rows = self._conn.execute(
            "SELECT * FROM theorems WHERE paper_arxiv_id = ? ORDER BY id",
            (paper_arxiv_id,),
        ).fetchall()
        return [self._row_to_theorem(r) for r in rows]

    def list_techniques(self, limit: int = 100) -> list[Technique]:
        rows = self._conn.execute(
            """SELECT name, description, first_seen_arxiv_id
               FROM techniques
               ORDER BY name
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            Technique(
                name=r["name"],
                description=r["description"] or "",
                first_seen_arxiv_id=r["first_seen_arxiv_id"],
            )
            for r in rows
        ]

    def retrieve_relevant(
        self,
        candidate_techniques: Iterable[str],
        limit_per_technique: int = 3,
        max_total: int = 12,
    ) -> list[Theorem]:
        """For a new paper that *might* use these techniques, return prior
        theorems indexed by those techniques. Dedup across techniques.
        Used by the distiller to inject context before LLM call.
        """
        seen_ids: set[int] = set()
        out: list[Theorem] = []
        for tech in candidate_techniques:
            if len(out) >= max_total:
                break
            for thm in self.theorems_using_technique(tech, limit_per_technique):
                if thm.id is None or thm.id in seen_ids:
                    continue
                seen_ids.add(thm.id)
                out.append(thm)
                if len(out) >= max_total:
                    break
        return out

    def retrieve_by_text_match(
        self, text: str, limit: int = 6,
    ) -> list[Theorem]:
        """Strategy B: FTS5 BM25 match between arbitrary text (e.g. a new
        paper's title + abstract) and stored theorem statements + proof
        sketches. Catches relevant prior work when the keyword scan misses
        because the new paper's abstract uses different vocabulary.

        Tokenizes the input, keeps alphabetic tokens 3-50 chars, drops common
        stopwords, runs FTS5 OR query over the top 30 tokens.
        """
        if not text.strip():
            return []
        tokens: list[str] = []
        seen_words: set[str] = set()
        for w in text.split()[:300]:
            clean = w.strip('".,!?()[]{}<>:;')
            clean = clean.replace('"', "")
            if not (3 <= len(clean) <= 50):
                continue
            if not clean.isalpha():
                continue
            lower = clean.lower()
            if lower in _STOPWORDS or lower in seen_words:
                continue
            seen_words.add(lower)
            tokens.append(f'"{clean}"')
        if not tokens:
            return []
        # FTS5 OR query over first 30 distinct tokens
        fts_query = " OR ".join(tokens[:30])
        try:
            rows = self._conn.execute(
                """SELECT t.*, bm25(theorems_fts) AS score
                   FROM theorems t
                   JOIN theorems_fts ON theorems_fts.rowid = t.id
                   WHERE theorems_fts MATCH ?
                   ORDER BY score
                   LIMIT ?""",
                (fts_query, limit),
            ).fetchall()
        except Exception:
            # FTS5 parse error — return empty rather than crash distill
            return []
        return [self._row_to_theorem(r) for r in rows]

    def list_canonical_technique_names(self, limit: int = 500) -> list[str]:
        """Strategy A: the list of all technique names we've ever seen,
        used to augment hardcoded keyword scan with vault-learned names."""
        rows = self._conn.execute(
            "SELECT name FROM techniques ORDER BY name LIMIT ?",
            (limit,),
        ).fetchall()
        return [r["name"] for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row_to_theorem(self, row) -> Theorem:
        try:
            tech = json.loads(row["techniques_used"] or "[]")
        except json.JSONDecodeError:
            tech = []
        return Theorem(
            id=row["id"],
            paper_arxiv_id=row["paper_arxiv_id"],
            paper_slug=row["paper_slug"],
            name=row["name"],
            statement=row["statement"],
            proof_sketch=row["proof_sketch"] or "",
            techniques_used=tech,
            created_at=row["created_at"],
        )


def open_for_vault(vault_path: Path | str) -> ProofStore:
    """Per-vault ProofStore at <vault>/.proof_store/proofs.db."""
    base = Path(vault_path) / ".proof_store"
    return ProofStore(base / "proofs.db")
