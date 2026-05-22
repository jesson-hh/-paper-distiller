# Proof-Graph Store + Segmentation/Grounding-Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic, LLM-free foundation of the proof graph — extend `proofs/store.py` into a node/edge graph store (with a safe 1→2 migration), and add a `proofgraph` package with paper segmentation + the verbatim-quote "grounding gate".

**Architecture:** Extend the existing per-vault SQLite+FTS5 `ProofStore` (keep `theorems` untouched for `find_proof` back-compat; add `nodes`/`edges`/`node_techniques`/`nodes_fts`). Add a new `proofgraph/` package whose first two pieces are pure functions: deterministic `segment(text)` and `verify_quote(quote, segment_text)`. Everything here is unit-testable without any LLM or network.

**Tech Stack:** Python 3.10+, stdlib `sqlite3` (FTS5), `dataclasses`, `difflib` (fuzzy match), `pytest` (LLM/network mocked elsewhere; nothing mocked here — these are pure/deterministic).

This plan implements **phases 1–2** of `docs/superpowers/specs/2026-05-21-deep-distill-proof-graph-design.md`. Phases 3–7 (extraction pipeline, distill wiring, linker, review agent) are separate plans.

---

## File Structure

- **Modify** `src/paper_distiller/proofs/store.py` — add `Node`/`Edge` dataclasses, graph tables in `_SCHEMA`, `SCHEMA_VERSION=2`, `_migrate()` + `_backfill_theorems_to_nodes()`, and graph CRUD/query methods. `theorems`/`techniques`/existing methods stay byte-for-byte unchanged.
- **Create** `src/paper_distiller/proofgraph/__init__.py` — package marker + docstring.
- **Create** `src/paper_distiller/proofgraph/reader.py` — `Segment` dataclass + `segment(text)`; `GateResult` dataclass + `verify_quote(quote, segment_text)`.
- **Modify** `tests/proofs/test_store.py` — append graph-store tests (mirror existing `tmp_path` style).
- **Create** `tests/proofgraph/__init__.py` — empty test package marker.
- **Create** `tests/proofgraph/test_segment.py` — segmentation tests.
- **Create** `tests/proofgraph/test_grounding_gate.py` — grounding-gate tests.

Run the whole suite with `python -m pytest -q` from the repo root. Default interpreter on this machine is miniconda base (3.13) which has the package editable-installed.

---

## Task 1: Graph tables + SCHEMA_VERSION=2 + migration

**Files:**
- Modify: `src/paper_distiller/proofs/store.py` (the `_SCHEMA` string near line 81, `SCHEMA_VERSION` at line 13, `ProofStore.__init__` near line 136)
- Test: `tests/proofs/test_store.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/proofs/test_store.py`:

```python
def test_graph_tables_exist_on_new_db(tmp_path):
    from paper_distiller.proofs.store import ProofStore
    store = ProofStore(tmp_path / "proofs.db")
    tables = {row[0] for row in store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert {"nodes", "edges", "node_techniques"} <= tables
    fts = {row[0] for row in store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='nodes_fts'"
    )}
    assert "nodes_fts" in fts
    assert store._conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()[0] == "2"
    store.close()


def test_migration_backfills_theorems_into_nodes(tmp_path):
    """A v1-shaped DB (theorems but no theorem-nodes) gets theorem nodes on open."""
    import sqlite3
    db = tmp_path / "proofs.db"
    # Build a minimal v1 DB by hand: just the theorems table + a row + version 1.
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "CREATE TABLE theorems (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "paper_arxiv_id TEXT NOT NULL, paper_slug TEXT, name TEXT NOT NULL, "
        "statement TEXT NOT NULL, proof_sketch TEXT, techniques_used TEXT NOT NULL, "
        "created_at TEXT NOT NULL);"
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
    )
    conn.execute(
        "INSERT INTO theorems(paper_arxiv_id,paper_slug,name,statement,proof_sketch,"
        "techniques_used,created_at) VALUES(?,?,?,?,?,?,?)",
        ("2110.1", "slug-a", "Theorem 1", "X holds.", "sketch",
         '["Bernstein"]', "2026-01-01T00:00:00"),
    )
    conn.execute("INSERT INTO meta(key,value) VALUES('schema_version','1')")
    conn.commit()
    conn.close()

    from paper_distiller.proofs.store import ProofStore
    store = ProofStore(db)  # opening runs the migration
    rows = store._conn.execute(
        "SELECT paper_arxiv_id, kind, label, text FROM nodes WHERE kind='theorem'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["paper_arxiv_id"] == "2110.1"
    assert rows[0]["label"] == "Theorem 1"
    techs = [r["technique"] for r in store._conn.execute(
        "SELECT technique FROM node_techniques")]
    assert techs == ["Bernstein"]
    assert store._conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "2"
    store.close()


def test_migration_is_idempotent(tmp_path):
    """Re-opening a migrated DB must not double-create theorem nodes."""
    import sqlite3
    db = tmp_path / "proofs.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "CREATE TABLE theorems (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "paper_arxiv_id TEXT NOT NULL, paper_slug TEXT, name TEXT NOT NULL, "
        "statement TEXT NOT NULL, proof_sketch TEXT, techniques_used TEXT NOT NULL, "
        "created_at TEXT NOT NULL);"
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
    )
    for i in (1, 2):
        conn.execute(
            "INSERT INTO theorems(paper_arxiv_id,paper_slug,name,statement,"
            "proof_sketch,techniques_used,created_at) VALUES(?,?,?,?,?,?,?)",
            ("2110.1", "slug", f"Theorem {i}", "X.", "s", "[]",
             "2026-01-01T00:00:00"),
        )
    conn.execute("INSERT INTO meta(key,value) VALUES('schema_version','1')")
    conn.commit(); conn.close()

    from paper_distiller.proofs.store import ProofStore
    ProofStore(db).close()   # first open: migrates + backfills 2 theorem nodes
    ProofStore(db).close()   # second open: version already 2 → backfill skipped
    s = ProofStore(db)       # third open
    theorem_nodes = s._conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE kind='theorem'").fetchone()[0]
    assert theorem_nodes == 2  # not 4, not 6
    s.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/proofs/test_store.py -k "graph_tables or migration" -v`
Expected: FAIL (no `nodes` table; `schema_version` is `"1"`).

- [ ] **Step 3: Bump version + add graph DDL.** In `src/paper_distiller/proofs/store.py` change `SCHEMA_VERSION = 1` to `SCHEMA_VERSION = 2`. Then append the following block to the `_SCHEMA` string, immediately before its closing `"""` (after the `meta` table definition):

```sql

CREATE TABLE IF NOT EXISTS nodes (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_arxiv_id  TEXT NOT NULL,
  paper_slug      TEXT,
  kind            TEXT NOT NULL,
  label           TEXT,
  text            TEXT NOT NULL,
  source_quote    TEXT,
  loc             TEXT,
  status          TEXT NOT NULL DEFAULT 'extracted',
  confidence      REAL,
  parent_id       INTEGER,
  ord             INTEGER,
  created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nodes_paper  ON nodes(paper_arxiv_id);
CREATE INDEX IF NOT EXISTS idx_nodes_kind   ON nodes(kind);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);

CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
  label, text, source_quote,
  content='nodes', content_rowid='id',
  tokenize='porter unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
  INSERT INTO nodes_fts(rowid, label, text, source_quote)
  VALUES (new.id, new.label, new.text, new.source_quote);
END;
CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
  INSERT INTO nodes_fts(nodes_fts, rowid, label, text, source_quote)
  VALUES('delete', old.id, old.label, old.text, old.source_quote);
END;

CREATE TABLE IF NOT EXISTS edges (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  src_id        INTEGER NOT NULL,
  dst_id        INTEGER NOT NULL,
  rel           TEXT NOT NULL,
  justification TEXT,
  cross_paper   INTEGER NOT NULL DEFAULT 0,
  created_at    TEXT NOT NULL,
  UNIQUE(src_id, dst_id, rel)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(rel);

CREATE TABLE IF NOT EXISTS node_techniques (
  node_id   INTEGER NOT NULL,
  technique TEXT NOT NULL,
  PRIMARY KEY (node_id, technique)
);
```

- [ ] **Step 4: Replace the version write with a migration.** In `ProofStore.__init__`, replace these lines:

```python
        self._conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()
```

with:

```python
        self._migrate()
        self._conn.commit()
```

Then add these two methods to the class (anywhere after `__init__`, before `close`):

```python
    def _migrate(self) -> None:
        """Idempotent forward migration. v1 = theorems-only; v2 adds the graph
        tables (created by _SCHEMA) and backfills theorem nodes."""
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        current = int(row[0]) if row else 0
        if current < 2:
            self._backfill_theorems_to_nodes()
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

    def _backfill_theorems_to_nodes(self) -> None:
        """Copy existing `theorems` rows into `nodes` as kind='theorem'.
        Guarded by (paper, label) so re-running never double-inserts."""
        rows = self._conn.execute(
            "SELECT paper_arxiv_id, paper_slug, name, statement, "
            "techniques_used, created_at FROM theorems"
        ).fetchall()
        for r in rows:
            exists = self._conn.execute(
                "SELECT 1 FROM nodes WHERE paper_arxiv_id=? AND kind='theorem' "
                "AND label IS ?",
                (r["paper_arxiv_id"], r["name"]),
            ).fetchone()
            if exists:
                continue
            cur = self._conn.execute(
                "INSERT INTO nodes(paper_arxiv_id, paper_slug, kind, label, text, "
                "status, created_at) VALUES (?, ?, 'theorem', ?, ?, 'extracted', ?)",
                (r["paper_arxiv_id"], r["paper_slug"], r["name"],
                 r["statement"], r["created_at"]),
            )
            node_id = cur.lastrowid
            try:
                techs = json.loads(r["techniques_used"] or "[]")
            except json.JSONDecodeError:
                techs = []
            for t in techs:
                if isinstance(t, str) and t.strip():
                    self._conn.execute(
                        "INSERT OR IGNORE INTO node_techniques(node_id, technique) "
                        "VALUES (?, ?)",
                        (node_id, t.strip()),
                    )
```

Note: in this plan `ingest_sidecar` still writes only the `theorems` table — wiring graph-node writes into ingestion is **phase 4** (a later plan). `_backfill_theorems_to_nodes` exists to upgrade vaults that already hold `theorems` rows from prior use; it runs once (guarded by `schema_version`), so reopening a migrated DB never double-creates nodes.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/proofs/test_store.py -k "graph_tables or migration" -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/paper_distiller/proofs/store.py tests/proofs/test_store.py
git commit -m "feat(proofs): add graph tables (nodes/edges) + safe 1->2 migration"
```

---

## Task 2: `Node` dataclass + `add_node` / `get_node` / `nodes_by_paper`

**Files:**
- Modify: `src/paper_distiller/proofs/store.py`
- Test: `tests/proofs/test_store.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_add_and_get_node(tmp_path):
    from paper_distiller.proofs.store import ProofStore, Node
    store = ProofStore(tmp_path / "proofs.db")
    nid = store.add_node(Node(
        paper_arxiv_id="2110.1", kind="proof_step", text="By Hölder, A<=B.",
        label="Step (a)", source_quote="By Hölder, A<=B.", loc='{"sec":"3.2"}',
        techniques=["Hölder"], ord=1,
    ))
    assert isinstance(nid, int)
    got = store.get_node(nid)
    assert got.id == nid
    assert got.kind == "proof_step"
    assert got.techniques == ["Hölder"]
    assert got.status == "extracted"
    by_paper = store.nodes_by_paper("2110.1")
    assert [n.id for n in by_paper] == [nid]
    store.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/proofs/test_store.py::test_add_and_get_node -v`
Expected: FAIL (`cannot import name 'Node'`).

- [ ] **Step 3: Add the `Node` dataclass** (next to the existing `Theorem` dataclass, after line ~46):

```python
@dataclass
class Node:
    """One node in the proof graph (theorem/lemma/def/assumption/step/claim)."""
    paper_arxiv_id: str
    kind: str
    text: str
    paper_slug: str | None = None
    label: str | None = None
    source_quote: str | None = None
    loc: str | None = None            # JSON string, e.g. '{"sec":"3.2","char":4120}'
    status: str = "extracted"
    confidence: float | None = None
    parent_id: int | None = None
    ord: int | None = None
    techniques: list = field(default_factory=list)
    id: int | None = None
    created_at: str | None = None
```

- [ ] **Step 4: Add the methods** to `ProofStore` (after the ingestion section):

```python
    def add_node(self, node: Node) -> int:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cur = self._conn.execute(
            "INSERT INTO nodes(paper_arxiv_id, paper_slug, kind, label, text, "
            "source_quote, loc, status, confidence, parent_id, ord, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (node.paper_arxiv_id, node.paper_slug, node.kind, node.label, node.text,
             node.source_quote, node.loc, node.status, node.confidence,
             node.parent_id, node.ord, now),
        )
        node_id = cur.lastrowid
        for t in node.techniques or []:
            if isinstance(t, str) and t.strip():
                self._conn.execute(
                    "INSERT OR IGNORE INTO node_techniques(node_id, technique) "
                    "VALUES (?, ?)",
                    (node_id, t.strip()),
                )
        self._conn.commit()
        return node_id

    def _row_to_node(self, row) -> Node:
        techs = [r["technique"] for r in self._conn.execute(
            "SELECT technique FROM node_techniques WHERE node_id=? ORDER BY technique",
            (row["id"],),
        )]
        return Node(
            id=row["id"], paper_arxiv_id=row["paper_arxiv_id"],
            paper_slug=row["paper_slug"], kind=row["kind"], label=row["label"],
            text=row["text"], source_quote=row["source_quote"], loc=row["loc"],
            status=row["status"], confidence=row["confidence"],
            parent_id=row["parent_id"], ord=row["ord"],
            techniques=techs, created_at=row["created_at"],
        )

    def get_node(self, node_id: int) -> Node | None:
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        return self._row_to_node(row) if row else None

    def nodes_by_paper(self, paper_arxiv_id: str) -> list[Node]:
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE paper_arxiv_id=? ORDER BY id",
            (paper_arxiv_id,)).fetchall()
        return [self._row_to_node(r) for r in rows]
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/proofs/test_store.py::test_add_and_get_node -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/paper_distiller/proofs/store.py tests/proofs/test_store.py
git commit -m "feat(proofs): Node dataclass + add_node/get_node/nodes_by_paper"
```

---

## Task 3: `Edge` dataclass + `add_edge` (idempotent) + `out_edges`/`in_edges`

**Files:**
- Modify: `src/paper_distiller/proofs/store.py`
- Test: `tests/proofs/test_store.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_add_edge_idempotent_and_query(tmp_path):
    from paper_distiller.proofs.store import ProofStore, Node, Edge
    store = ProofStore(tmp_path / "proofs.db")
    a = store.add_node(Node(paper_arxiv_id="p", kind="proof_step", text="step a"))
    b = store.add_node(Node(paper_arxiv_id="p", kind="assumption", text="A2"))
    store.add_edge(Edge(src_id=a, dst_id=b, rel="uses_assumption"))
    store.add_edge(Edge(src_id=a, dst_id=b, rel="uses_assumption"))  # dup
    out = store.out_edges(a)
    assert len(out) == 1  # UNIQUE(src,dst,rel) collapses the dup
    assert out[0].dst_id == b and out[0].rel == "uses_assumption"
    assert [e.src_id for e in store.in_edges(b)] == [a]
    store.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/proofs/test_store.py::test_add_edge_idempotent_and_query -v`
Expected: FAIL (`cannot import name 'Edge'`).

- [ ] **Step 3: Add `Edge` dataclass** (after `Node`):

```python
@dataclass
class Edge:
    """A typed dependency edge: src --rel--> dst means 'src depends on / uses dst'."""
    src_id: int
    dst_id: int
    rel: str
    justification: str | None = None
    cross_paper: int = 0
    id: int | None = None
    created_at: str | None = None
```

- [ ] **Step 4: Add methods** to `ProofStore`:

```python
    def add_edge(self, edge: Edge) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._conn.execute(
            "INSERT OR IGNORE INTO edges(src_id, dst_id, rel, justification, "
            "cross_paper, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (edge.src_id, edge.dst_id, edge.rel, edge.justification,
             int(edge.cross_paper), now),
        )
        self._conn.commit()

    def _row_to_edge(self, row) -> Edge:
        return Edge(
            id=row["id"], src_id=row["src_id"], dst_id=row["dst_id"], rel=row["rel"],
            justification=row["justification"], cross_paper=row["cross_paper"],
            created_at=row["created_at"],
        )

    def out_edges(self, node_id: int, rel: str | None = None) -> list[Edge]:
        if rel is None:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE src_id=? ORDER BY id", (node_id,)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE src_id=? AND rel=? ORDER BY id",
                (node_id, rel)).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def in_edges(self, node_id: int, rel: str | None = None) -> list[Edge]:
        if rel is None:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE dst_id=? ORDER BY id", (node_id,)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE dst_id=? AND rel=? ORDER BY id",
                (node_id, rel)).fetchall()
        return [self._row_to_edge(r) for r in rows]
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/proofs/test_store.py::test_add_edge_idempotent_and_query -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/paper_distiller/proofs/store.py tests/proofs/test_store.py
git commit -m "feat(proofs): Edge dataclass + idempotent add_edge + out/in_edges"
```

---

## Task 4: `search_nodes` (FTS5) + `nodes_using_technique`

**Files:**
- Modify: `src/paper_distiller/proofs/store.py`
- Test: `tests/proofs/test_store.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_search_nodes_and_by_technique(tmp_path):
    from paper_distiller.proofs.store import ProofStore, Node
    store = ProofStore(tmp_path / "proofs.db")
    store.add_node(Node(paper_arxiv_id="p", kind="proof_step",
                        text="Bound the empirical process via Dudley chaining.",
                        techniques=["Dudley chaining"]))
    store.add_node(Node(paper_arxiv_id="p", kind="proof_step",
                        text="Apply Hölder inequality to split the product.",
                        techniques=["Hölder"]))
    hits = store.search_nodes("chaining")
    assert len(hits) == 1 and "Dudley" in hits[0].text
    by_tech = store.nodes_using_technique("Hölder")
    assert len(by_tech) == 1 and "Hölder" in by_tech[0].text
    store.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/proofs/test_store.py::test_search_nodes_and_by_technique -v`
Expected: FAIL (`AttributeError: 'ProofStore' object has no attribute 'search_nodes'`).

- [ ] **Step 3: Add methods** to `ProofStore`:

```python
    def search_nodes(self, query: str, limit: int = 10) -> list[Node]:
        """FTS5 search over node label + text + source_quote."""
        if not query.strip():
            return []
        tokens = ['"' + tok.replace('"', '') + '"' for tok in query.split() if tok]
        if not tokens:
            return []
        rows = self._conn.execute(
            "SELECT n.* FROM nodes n JOIN nodes_fts ON nodes_fts.rowid = n.id "
            "WHERE nodes_fts MATCH ? ORDER BY bm25(nodes_fts) LIMIT ?",
            (" ".join(tokens), limit),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def nodes_using_technique(self, technique: str, limit: int = 10) -> list[Node]:
        if not technique.strip():
            return []
        rows = self._conn.execute(
            "SELECT n.* FROM nodes n JOIN node_techniques nt ON nt.node_id = n.id "
            "WHERE nt.technique = ? COLLATE NOCASE ORDER BY n.id DESC LIMIT ?",
            (technique.strip(), limit),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/proofs/test_store.py::test_search_nodes_and_by_technique -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/paper_distiller/proofs/store.py tests/proofs/test_store.py
git commit -m "feat(proofs): FTS5 search_nodes + nodes_using_technique"
```

---

## Task 5: `dependency_walk` (DAG traversal over `depends_on`)

**Files:**
- Modify: `src/paper_distiller/proofs/store.py`
- Test: `tests/proofs/test_store.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_dependency_walk(tmp_path):
    from paper_distiller.proofs.store import ProofStore, Node, Edge
    store = ProofStore(tmp_path / "proofs.db")
    thm = store.add_node(Node(paper_arxiv_id="p", kind="theorem", text="T"))
    s2 = store.add_node(Node(paper_arxiv_id="p", kind="proof_step", text="step2"))
    s1 = store.add_node(Node(paper_arxiv_id="p", kind="proof_step", text="step1"))
    store.add_edge(Edge(src_id=thm, dst_id=s2, rel="depends_on"))
    store.add_edge(Edge(src_id=s2,  dst_id=s1, rel="depends_on"))
    walked = store.dependency_walk(thm)
    walked_ids = [n.id for n in walked]
    assert walked_ids == [s2, s1]  # transitive deps, BFS order, excludes the root
    # Cycle safety: add a back-edge and ensure it still terminates.
    store.add_edge(Edge(src_id=s1, dst_id=thm, rel="depends_on"))
    assert len(store.dependency_walk(thm)) <= 3
    store.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/proofs/test_store.py::test_dependency_walk -v`
Expected: FAIL (no `dependency_walk`).

- [ ] **Step 3: Add the method** to `ProofStore`:

```python
    def dependency_walk(
        self, node_id: int, rel: str = "depends_on", max_nodes: int = 500,
    ) -> list[Node]:
        """Breadth-first transitive closure following `rel` edges out of node_id.
        Excludes the root. Cycle-safe (visited set), capped at max_nodes."""
        from collections import deque
        seen: set[int] = {node_id}
        order: list[int] = []
        queue: deque[int] = deque([node_id])
        while queue and len(order) < max_nodes:
            cur = queue.popleft()
            for e in self.out_edges(cur, rel):
                if e.dst_id not in seen:
                    seen.add(e.dst_id)
                    order.append(e.dst_id)
                    queue.append(e.dst_id)
        return [n for n in (self.get_node(i) for i in order) if n is not None]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/proofs/test_store.py::test_dependency_walk -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/paper_distiller/proofs/store.py tests/proofs/test_store.py
git commit -m "feat(proofs): cycle-safe dependency_walk over depends_on edges"
```

---

## Task 6: `delete_paper_graph` (idempotent per-paper rewrite support)

**Files:**
- Modify: `src/paper_distiller/proofs/store.py`
- Test: `tests/proofs/test_store.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_delete_paper_graph(tmp_path):
    from paper_distiller.proofs.store import ProofStore, Node, Edge
    store = ProofStore(tmp_path / "proofs.db")
    a = store.add_node(Node(paper_arxiv_id="keep", kind="theorem", text="K"))
    b = store.add_node(Node(paper_arxiv_id="drop", kind="theorem", text="D",
                            techniques=["Hölder"]))
    c = store.add_node(Node(paper_arxiv_id="drop", kind="proof_step", text="d-step"))
    store.add_edge(Edge(src_id=b, dst_id=c, rel="depends_on"))
    store.delete_paper_graph("drop")
    assert [n.id for n in store.nodes_by_paper("drop")] == []
    assert [n.id for n in store.nodes_by_paper("keep")] == [a]
    # edges + node_techniques for the dropped paper are gone too
    assert store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0
    assert store._conn.execute(
        "SELECT COUNT(*) FROM node_techniques").fetchone()[0] == 0
    store.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/proofs/test_store.py::test_delete_paper_graph -v`
Expected: FAIL (no `delete_paper_graph`).

- [ ] **Step 3: Add the method** to `ProofStore`:

```python
    def delete_paper_graph(self, paper_arxiv_id: str) -> None:
        """Remove all graph nodes/edges/technique-links for one paper, so a
        re-distill can cleanly rewrite them (paper-grained idempotency)."""
        ids = [r["id"] for r in self._conn.execute(
            "SELECT id FROM nodes WHERE paper_arxiv_id=?", (paper_arxiv_id,))]
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        self._conn.execute(
            f"DELETE FROM edges WHERE src_id IN ({placeholders}) "
            f"OR dst_id IN ({placeholders})", (*ids, *ids))
        self._conn.execute(
            f"DELETE FROM node_techniques WHERE node_id IN ({placeholders})", ids)
        self._conn.execute(
            "DELETE FROM nodes WHERE paper_arxiv_id=?", (paper_arxiv_id,))
        self._conn.commit()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/proofs/test_store.py::test_delete_paper_graph -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/paper_distiller/proofs/store.py tests/proofs/test_store.py
git commit -m "feat(proofs): delete_paper_graph for idempotent per-paper rewrites"
```

---

## Task 7: `find_proof` regression guard (theorems untouched)

**Files:**
- Test only: `tests/proofs/test_store.py`

This task adds NO production code — it locks in that the schema/migration changes did not break the existing theorem layer that `find_proof` depends on.

- [ ] **Step 1: Write the test** — append:

```python
def test_theorem_layer_unchanged_after_graph_migration(tmp_path):
    """Existing theorem ingestion + queries still work identically with v2 schema."""
    from paper_distiller.proofs.store import ProofStore
    store = ProofStore(tmp_path / "proofs.db")
    store.ingest_sidecar(_sample_sidecar(), "2110.12319", paper_slug="bigan-bounds")
    assert store.theorem_count() == 2
    assert store.technique_count() == 5
    assert store.paper_count() == 1
    assert len(store.theorems_using_technique("Hölder")) == 1
    assert len(store.search_theorems("Bernstein")) >= 1
    assert len(store.theorems_by_paper("2110.12319")) == 2
    assert "Hölder" in store.list_canonical_technique_names()
    store.close()
```

- [ ] **Step 2: Run to verify it passes immediately** (regression guard — should already be green)

Run: `python -m pytest tests/proofs/test_store.py::test_theorem_layer_unchanged_after_graph_migration -v`
Expected: PASS. If it FAILS, a previous task broke back-compat — fix that task before continuing.

- [ ] **Step 3: Run the full proofs suite**

Run: `python -m pytest tests/proofs/ -q`
Expected: PASS (all existing + new tests).

- [ ] **Step 4: Commit**

```bash
git add tests/proofs/test_store.py
git commit -m "test(proofs): regression guard for theorem layer under v2 schema"
```

---

## Task 8: `proofgraph` package + `segment(text)` (deterministic, LLM-free)

**Files:**
- Create: `src/paper_distiller/proofgraph/__init__.py`
- Create: `src/paper_distiller/proofgraph/reader.py`
- Create: `tests/proofgraph/__init__.py` (empty)
- Create: `tests/proofgraph/test_segment.py`

- [ ] **Step 1: Write the failing test** — create `tests/proofgraph/__init__.py` (empty file) and `tests/proofgraph/test_segment.py`:

```python
"""Tests for proofgraph.reader.segment — deterministic paper segmentation."""
from __future__ import annotations

SAMPLE = """\
1 Introduction
We study the convergence of estimators under sub-Gaussian noise.

2 Main Result
Theorem 4.3. For all f, ||f|| <= C n^{-1/2}.
Proof. By Bernstein's inequality we bound the tail. Applying Dudley chaining
to the empirical process yields the claim. □

3 Discussion
Future work remains.
"""


def test_segment_splits_by_section_and_marks_proof_block():
    from paper_distiller.proofgraph.reader import segment
    segs = segment(SAMPLE)
    # every segment carries the text it covers + offsets within the source
    assert all(s.text == SAMPLE[s.char_start:s.char_end] for s in segs)
    # at least one proof block detected (the "Proof. ... □" region)
    proofs = [s for s in segs if s.is_proof_block]
    assert len(proofs) == 1
    assert "Bernstein" in proofs[0].text
    # a theorem-statement segment is detected
    assert any(s.kind_hint == "theorem" for s in segs)
    # coverage: concatenated segment text reconstructs (modulo splits) the source
    assert sum(len(s.text) for s in segs) > 0


def test_segment_empty_input_returns_empty():
    from paper_distiller.proofgraph.reader import segment
    assert segment("") == []
    assert segment("   \n  ") == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/proofgraph/test_segment.py -v`
Expected: FAIL (`ModuleNotFoundError: paper_distiller.proofgraph`).

- [ ] **Step 3: Create the package + implementation.** Create `src/paper_distiller/proofgraph/__init__.py`:

```python
"""Proof-graph subsystem: deterministic reading (segmentation + grounding gate),
extraction, cross-paper linking, and review. Phase 1-2 ships reader.py only."""
```

Create `src/paper_distiller/proofgraph/reader.py`:

```python
"""Deterministic, LLM-free reading primitives for the proof-graph pipeline.

- segment(text): split a paper's plain text into ordered Segments, flagging
  theorem-statement and proof-block regions. This is the coverage denominator
  for "don't skim" — every segment must later be visited.
- verify_quote(quote, segment_text): the grounding gate (Task 9).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Segment:
    id: int
    kind_hint: str          # "prose" | "theorem" | "proof" | "definition" | "heading"
    section: str | None     # nearest preceding heading label, e.g. "2 Main Result"
    text: str
    char_start: int
    char_end: int
    is_proof_block: bool


# A heading line: "1 Introduction", "2.1 Setup", "Appendix A", etc.
_HEADING_RE = re.compile(r"^\s*(\d+(\.\d+)*\s+\S.*|Appendix\s+\S.*)$")
# Start of a theorem-like statement.
_THEOREM_RE = re.compile(
    r"^\s*(Theorem|Lemma|Proposition|Corollary|Claim|Definition)\b", re.IGNORECASE)
_PROOF_START_RE = re.compile(r"^\s*Proof\b", re.IGNORECASE)
# End-of-proof markers: QED box, "QED", "q.e.d."
_PROOF_END_RE = re.compile(r"(□|\bQED\b|\bq\.?e\.?d\.?\b)", re.IGNORECASE)


def _classify(block: str) -> str:
    head = block.lstrip()
    if _PROOF_START_RE.match(head):
        return "proof"
    if _THEOREM_RE.match(head):
        first = head.split(None, 1)[0].lower()
        return "definition" if first.startswith("defin") else "theorem"
    if _HEADING_RE.match(head.splitlines()[0] if head else ""):
        return "heading"
    return "prose"


def segment(text: str) -> list[Segment]:
    """Split into structural blocks — a new block starts at each heading /
    Theorem-like / Proof line, and at blank-line paragraph breaks. Classify
    each block and record char offsets so downstream code can reconstruct and
    ground to source. This list is the coverage denominator for "don't skim"."""
    if not text or not text.strip():
        return []
    lines = text.splitlines(keepends=True)
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln)

    segments: list[Segment] = []
    cur: list[int] = []
    state = {"sid": 0, "section": None}

    def _is_boundary(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        return bool(_HEADING_RE.match(s) or _THEOREM_RE.match(s)
                    or _PROOF_START_RE.match(s))

    def _flush() -> None:
        if not cur:
            return
        start = offsets[cur[0]]
        end = offsets[cur[-1]] + len(lines[cur[-1]])
        block = text[start:end]
        cur.clear()
        if not block.strip():
            return
        kind = _classify(block)
        if kind == "heading":
            state["section"] = block.strip().splitlines()[0].strip()
        segments.append(Segment(
            id=state["sid"], kind_hint=kind, section=state["section"],
            text=block, char_start=start, char_end=end,
            is_proof_block=bool(_PROOF_START_RE.match(block.lstrip())),
        ))
        state["sid"] += 1

    for i, ln in enumerate(lines):
        if not ln.strip():
            _flush()
            continue
        if cur and _is_boundary(ln):
            _flush()
        cur.append(i)
    _flush()
    return segments
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/proofgraph/test_segment.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/paper_distiller/proofgraph/__init__.py src/paper_distiller/proofgraph/reader.py tests/proofgraph/__init__.py tests/proofgraph/test_segment.py
git commit -m "feat(proofgraph): deterministic paper segmentation (segment)"
```

---

## Task 9: Grounding gate `verify_quote(quote, segment_text)`

**Files:**
- Modify: `src/paper_distiller/proofgraph/reader.py`
- Test: `tests/proofgraph/test_grounding_gate.py` (create)

- [ ] **Step 1: Write the failing test** — create `tests/proofgraph/test_grounding_gate.py`:

```python
"""Tests for the grounding gate — fabricated quotes must be rejected."""
from __future__ import annotations

SEG = ("By Hölder's inequality, we have   ||f g||_1  <=  ||f||_p ||g||_q, "
       "which after applying Dudley's chaining bounds the empirical process.")


def test_exact_quote_accepted():
    from paper_distiller.proofgraph.reader import verify_quote
    r = verify_quote("By Hölder's inequality", SEG)
    assert r.ok and r.score == 1.0


def test_whitespace_normalized_quote_accepted():
    from paper_distiller.proofgraph.reader import verify_quote
    # collapsed multiple spaces vs. source's double spaces
    r = verify_quote("||f g||_1 <= ||f||_p ||g||_q", SEG)
    assert r.ok


def test_ocr_noise_quote_fuzzy_accepted():
    from paper_distiller.proofgraph.reader import verify_quote
    # one transposed/garbled char ("Holder" missing umlaut) still passes fuzzy
    r = verify_quote("By Holder's inequality", SEG)
    assert r.ok and r.score >= 0.85


def test_fabricated_quote_rejected():
    from paper_distiller.proofgraph.reader import verify_quote
    r = verify_quote("By the Central Limit Theorem we conclude normality", SEG)
    assert not r.ok and r.score < 0.85


def test_empty_quote_rejected():
    from paper_distiller.proofgraph.reader import verify_quote
    assert not verify_quote("", SEG).ok
    assert not verify_quote("   ", SEG).ok
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/proofgraph/test_grounding_gate.py -v`
Expected: FAIL (`cannot import name 'verify_quote'`).

- [ ] **Step 3: Add `GateResult` + `verify_quote`** to `src/paper_distiller/proofgraph/reader.py` (append; add `from difflib import SequenceMatcher` to the imports at top):

```python
@dataclass
class GateResult:
    ok: bool
    score: float            # 1.0 = exact (after whitespace norm); else best fuzzy ratio
    matched_span: str | None  # the source substring that best matches, if any


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def verify_quote(
    quote: str, segment_text: str, threshold: float = 0.85,
) -> GateResult:
    """The grounding gate. Returns ok=True iff `quote` is found in
    `segment_text` exactly (after whitespace normalization) or with a best
    fuzzy ratio >= threshold over a sliding window. Fabricated quotes that
    aren't really in the source score low and are rejected — this is what
    structurally keeps hallucinated nodes out of the graph.
    """
    q = _norm_ws(quote)
    if not q:
        return GateResult(ok=False, score=0.0, matched_span=None)
    hay = _norm_ws(segment_text)
    if q in hay:
        return GateResult(ok=True, score=1.0, matched_span=q)
    # Fuzzy: slide a window the length of the quote across the haystack.
    words = hay.split(" ")
    qlen = len(q)
    best = 0.0
    best_span: str | None = None
    # Build candidate windows by character length around the quote length.
    for i in range(len(words)):
        window = ""
        j = i
        while j < len(words) and len(window) < qlen + 20:
            window = (window + " " + words[j]).strip()
            j += 1
            ratio = SequenceMatcher(None, q, window).ratio()
            if ratio > best:
                best, best_span = ratio, window
    return GateResult(ok=best >= threshold, score=round(best, 3),
                      matched_span=best_span if best >= threshold else None)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/proofgraph/test_grounding_gate.py -v`
Expected: PASS (5 tests). If `test_ocr_noise_quote_fuzzy_accepted` is borderline, the threshold 0.85 is intentional; "By Holder's inequality" vs "By Hölder's inequality" differs by ~1 char over ~22 → ratio ≈ 0.95, comfortably above.

- [ ] **Step 5: Commit**

```bash
git add src/paper_distiller/proofgraph/reader.py tests/proofgraph/test_grounding_gate.py
git commit -m "feat(proofgraph): grounding gate verify_quote (anti-hallucination)"
```

---

## Task 10: Full-suite green + version bump note

**Files:** none (verification + housekeeping)

- [ ] **Step 1: Run the entire suite**

Run: `python -m pytest -q`
Expected: PASS — all prior tests (437) plus the new graph-store + proofgraph tests; total grows by ~16 (T1:3, T2-T7:6, T8:2, T9:5).

- [ ] **Step 2: Lint the touched files**

Run: `python -m ruff check src/paper_distiller/proofs/store.py src/paper_distiller/proofgraph/`
Expected: `All checks passed!`

- [ ] **Step 3: (No version bump in this plan.)** These are additive internal modules; no PyPI release here. A release happens after a user-facing surface lands (phase 6). If a checkpoint release is desired, follow the 3-file bump in `CONTRIBUTING.md`.

---

## Notes for the implementer

- **Edge direction is fixed**: `src --rel--> dst` means "src depends on / uses dst". `dependency_walk(thm)` therefore returns the things the theorem rests on.
- **`ord` is a column name** (step order); it is not a reserved word in SQLite and is safe unquoted.
- **`techniques` on `Node`** are written to `node_techniques` by `add_node` and re-read by `_row_to_node` — never stored as JSON on the node row (unlike the legacy `theorems.techniques_used`).
- **Migration runs on every open** but is guarded: `_backfill_theorems_to_nodes` skips `(paper, label)` pairs that already exist, and `schema_version` is set to `2` afterward. Safe to re-run.
- **Nothing here calls an LLM or the network.** All tests are deterministic.
