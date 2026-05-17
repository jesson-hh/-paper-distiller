from pathlib import Path
from paper_distiller.vault.store import VaultStore
from paper_distiller.vault.crosslink import load_index


def test_load_index_empty_vault(tmp_vault: Path):
    store = VaultStore(tmp_vault)
    idx = load_index(store)
    assert idx.entries == []


def test_load_index_populated(tmp_vault: Path):
    store = VaultStore(tmp_vault)
    store.save_entry(title="A paper", category="articles",
                     body="x", tags=["t1"])
    store.save_entry(title="A technique", category="techniques", body="y")
    idx = load_index(store)
    assert len(idx.entries) == 2
    slugs = {e["slug"] for e in idx.entries}
    assert "a-paper" in slugs
    assert "a-technique" in slugs
