"""Tests for proofs.store — SQLite + FTS5 of theorems / techniques."""

from __future__ import annotations


def _sample_sidecar():
    from paper_distiller.proofs.store import ProofSidecar
    return ProofSidecar(
        theorems=[
            {
                "name": "Theorem 4.3",
                "statement": "For all $f \\in \\mathcal{F}$, "
                             "$\\|f\\|_\\infty \\leq C n^{-1/2}$",
                "proof_sketch": "Apply Bernstein's concentration + chaining.",
                "techniques_used": ["Bernstein concentration", "Dudley chaining"],
            },
            {
                "name": "Lemma 5.1",
                "statement": "If $X, Y$ are sub-Gaussian, then "
                             "$\\mathbb{E}[XY] \\leq \\|X\\|_{\\psi_2}\\|Y\\|_{\\psi_2}$.",
                "proof_sketch": "Apply Hölder for Orlicz spaces.",
                "techniques_used": ["Hölder", "Orlicz norm"],
            },
        ],
        key_definitions=[
            {"name": "IPM", "statement": "$d_\\mathcal{F}(\\mu,\\nu) = \\sup_{f \\in \\mathcal{F}}|\\mathbb{E}_\\mu f - \\mathbb{E}_\\nu f|$"}
        ],
        key_techniques=["Bernstein concentration", "Dudley chaining", "Hölder", "Orlicz norm", "ReLU approximation"],
    )


def test_store_creates_schema(tmp_path):
    from paper_distiller.proofs.store import ProofStore

    store = ProofStore(tmp_path / "proofs.db")
    tables = {row[0] for row in store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert "theorems" in tables
    assert "techniques" in tables
    assert "theorems_fts" in tables
    assert store.theorem_count() == 0
    assert store.technique_count() == 0
    store.close()


def test_ingest_sidecar(tmp_path):
    from paper_distiller.proofs.store import ProofStore

    store = ProofStore(tmp_path / "proofs.db")
    result = store.ingest_sidecar(_sample_sidecar(), "2110.12319", paper_slug="bigan-bounds")

    assert result["theorems_inserted"] == 2
    assert result["techniques_new"] == 5  # 5 distinct techniques
    assert store.theorem_count() == 2
    assert store.technique_count() == 5
    assert store.paper_count() == 1
    store.close()


def test_ingest_is_idempotent_per_paper(tmp_path):
    from paper_distiller.proofs.store import ProofStore

    store = ProofStore(tmp_path / "proofs.db")
    store.ingest_sidecar(_sample_sidecar(), "2110.12319")
    store.ingest_sidecar(_sample_sidecar(), "2110.12319")
    # Re-ingest of same paper → 2 theorems (no doubling)
    assert store.theorem_count() == 2
    store.close()


def test_theorems_using_technique(tmp_path):
    from paper_distiller.proofs.store import ProofStore

    store = ProofStore(tmp_path / "proofs.db")
    store.ingest_sidecar(_sample_sidecar(), "2110.12319")

    results = store.theorems_using_technique("Hölder")
    assert len(results) == 1
    assert results[0].name == "Lemma 5.1"

    results2 = store.theorems_using_technique("Bernstein concentration")
    assert len(results2) == 1
    assert results2[0].name == "Theorem 4.3"

    # Nonexistent technique → empty
    assert store.theorems_using_technique("Quantum Tunneling") == []
    store.close()


def test_search_theorems_fts(tmp_path):
    from paper_distiller.proofs.store import ProofStore

    store = ProofStore(tmp_path / "proofs.db")
    store.ingest_sidecar(_sample_sidecar(), "2110.12319")

    # Statement matches
    rs = store.search_theorems("sub-Gaussian")
    assert len(rs) == 1
    assert rs[0].name == "Lemma 5.1"

    # Proof sketch matches
    rs = store.search_theorems("chaining")
    assert len(rs) == 1
    assert rs[0].name == "Theorem 4.3"

    # No match → empty
    assert store.search_theorems("nonsense") == []
    store.close()


def test_retrieve_relevant_dedups(tmp_path):
    from paper_distiller.proofs.store import ProofStore

    store = ProofStore(tmp_path / "proofs.db")
    store.ingest_sidecar(_sample_sidecar(), "2110.12319")

    # Both technique names point to the SAME theorem (Lemma 5.1)
    out = store.retrieve_relevant(["Hölder", "Orlicz norm"], limit_per_technique=5)
    # Should dedupe — Lemma 5.1 appears once
    assert len(out) == 1
    assert out[0].name == "Lemma 5.1"
    store.close()


def test_retrieve_relevant_caps_total(tmp_path):
    from paper_distiller.proofs.store import ProofStore, ProofSidecar

    store = ProofStore(tmp_path / "proofs.db")
    # Ingest 5 distinct papers, each with 1 theorem using "Hölder"
    for i in range(5):
        sidecar = ProofSidecar(
            theorems=[{
                "name": f"Theorem {i}",
                "statement": "$x \\leq y$",
                "proof_sketch": "trivial",
                "techniques_used": ["Hölder"],
            }],
            key_techniques=["Hölder"],
        )
        store.ingest_sidecar(sidecar, f"2110.000{i}")

    out = store.retrieve_relevant(["Hölder"], limit_per_technique=10, max_total=3)
    assert len(out) == 3
    store.close()


def test_open_for_vault_creates_subdir(tmp_path):
    from paper_distiller.proofs.store import open_for_vault

    vault = tmp_path / "myvault"
    vault.mkdir()
    store = open_for_vault(vault)
    assert (vault / ".proof_store" / "proofs.db").exists()
    store.close()


def test_proof_sidecar_from_json_robust():
    from paper_distiller.proofs.store import ProofSidecar

    # Empty / missing fields
    assert ProofSidecar.from_json({}).theorems == []
    assert ProofSidecar.from_json({}).key_techniques == []
    # Wrong type
    assert ProofSidecar.from_json("not a dict").theorems == []
    assert ProofSidecar.from_json(None).theorems == []
    # Partial
    s = ProofSidecar.from_json({"theorems": [{"name": "x"}]})
    assert len(s.theorems) == 1


def test_techniques_first_seen_arxiv_id(tmp_path):
    from paper_distiller.proofs.store import ProofStore, ProofSidecar

    store = ProofStore(tmp_path / "proofs.db")
    s1 = ProofSidecar(theorems=[], key_techniques=["Hölder"])
    store.ingest_sidecar(s1, "2020.001")
    # Second paper uses same technique — first_seen stays as 2020.001
    s2 = ProofSidecar(theorems=[], key_techniques=["Hölder"])
    store.ingest_sidecar(s2, "2026.002")

    techs = store.list_techniques()
    holder = [t for t in techs if t.name == "Hölder"][0]
    assert holder.first_seen_arxiv_id == "2020.001"
    store.close()


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
    ProofStore(db).close()   # second open: version already 2 -> backfill skipped
    s = ProofStore(db)       # third open
    theorem_nodes = s._conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE kind='theorem'").fetchone()[0]
    assert theorem_nodes == 2  # not 4, not 6
    s.close()


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
