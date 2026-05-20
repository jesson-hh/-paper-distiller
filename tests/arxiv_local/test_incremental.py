"""Tests for arxiv_local.incremental — OAI-PMH sync with mocked Sickle."""

from __future__ import annotations

from unittest.mock import MagicMock


class _FakeRecord:
    """Minimal sickle.Record stand-in matching pyoai's parsed structure.

    pyoai flattens nested <author> elements into sibling lists at the same
    level: metadata["keyname"] = [...], metadata["forenames"] = [...].
    """
    def __init__(self, arxiv_id, title="T", abstract="A", deleted=False,
                 categories=("cs.LG",), authors_keyname="Smith",
                 authors_forenames="Alice"):
        self.deleted = deleted
        if deleted:
            self.metadata = {}
            return
        # Support either single-author (str) or multi-author (list/tuple).
        if isinstance(authors_keyname, (list, tuple)):
            keynames = list(authors_keyname)
            forenames = list(authors_forenames)
        else:
            keynames = [authors_keyname]
            forenames = [authors_forenames]
        self.metadata = {
            "id": [arxiv_id],
            "title": [title],
            "abstract": [abstract],
            "categories": [" ".join(categories)],
            "author": [None] * len(keynames),
            "keyname": keynames,
            "forenames": forenames,
            "created": ["2024-01-01"],
        }


def test_record_to_paper_skips_deleted():
    from paper_distiller.arxiv_local.oai_record import record_to_paper
    r = _FakeRecord("2401.0", deleted=True)
    assert record_to_paper(r) is None


def test_record_to_paper_flattens_authors():
    from paper_distiller.arxiv_local.oai_record import record_to_paper
    r = _FakeRecord("2401.0", title="Diffusion",
                    authors_keyname="Smith", authors_forenames="Alice")
    p = record_to_paper(r)
    assert p.arxiv_id == "2401.0"
    assert p.title == "Diffusion"
    assert p.authors == ["Alice Smith"]
    assert p.source == "oai-pmh"


def test_record_to_paper_multiple_authors():
    """The real Sickle parsed structure for arxiv has sibling lists keyname
    and forenames, not nested author dicts. Verify we zip them correctly."""
    from paper_distiller.arxiv_local.oai_record import record_to_paper
    r = _FakeRecord(
        "2401.0",
        authors_keyname=["Mészáros", "Micek", "Jiao"],
        authors_forenames=["Tamás", "Piotr", "Yuling"],
    )
    p = record_to_paper(r)
    assert p.authors == ["Tamás Mészáros", "Piotr Micek", "Yuling Jiao"]


def test_flatten_authors_handles_missing_forenames():
    """Some records have only keyname (single-name authors / institutional)."""
    from paper_distiller.arxiv_local.oai_record import _flatten_authors

    md = {
        "keyname": ["OnlySurname"],
        "forenames": [""],
    }
    assert _flatten_authors(md) == ["OnlySurname"]


def test_flatten_authors_zip_handles_uneven_lists():
    """If one list is longer than the other (shouldn't happen but defend)."""
    from paper_distiller.arxiv_local.oai_record import _flatten_authors

    md = {
        "keyname": ["A", "B", "C"],
        "forenames": ["X", "Y"],
    }
    # zip stops at shorter list — 2 authors, last one dropped silently
    assert _flatten_authors(md) == ["X A", "Y B"]


def test_record_to_paper_skips_no_id_no_title():
    from paper_distiller.arxiv_local.oai_record import record_to_paper
    r = _FakeRecord("2401.0")
    r.metadata.pop("id", None)
    assert record_to_paper(r) is None
    r2 = _FakeRecord("2401.0")
    r2.metadata["title"] = [""]
    assert record_to_paper(r2) is None


def test_sync_inserts_records(tmp_path):
    from paper_distiller.arxiv_local.incremental import sync
    from paper_distiller.arxiv_local.store import Store

    store = Store(tmp_path / "arxiv.db")

    fake_client = MagicMock()
    fake_client.ListRecords.return_value = iter([
        _FakeRecord("2401.0", title="Paper Zero"),
        _FakeRecord("2401.1", title="Paper One"),
    ])

    result = sync(store, since=None, sickle_client=fake_client)
    assert result.n_seen == 2
    assert result.n_inserted == 2
    assert store.paper_count() == 2
    store.close()


def test_sync_passes_from_date_to_sickle(tmp_path):
    from paper_distiller.arxiv_local.incremental import sync
    from paper_distiller.arxiv_local.store import Store

    store = Store(tmp_path / "arxiv.db")
    fake_client = MagicMock()
    fake_client.ListRecords.return_value = iter([])

    sync(store, since="2026-05-01", sickle_client=fake_client)
    call_kwargs = fake_client.ListRecords.call_args.kwargs
    assert call_kwargs["from"] == "2026-05-01"


def test_sync_skips_deleted_records(tmp_path):
    from paper_distiller.arxiv_local.incremental import sync
    from paper_distiller.arxiv_local.store import Store

    store = Store(tmp_path / "arxiv.db")
    fake_client = MagicMock()
    fake_client.ListRecords.return_value = iter([
        _FakeRecord("2401.0", title="Live"),
        _FakeRecord("2401.1", deleted=True),
    ])

    result = sync(store, sickle_client=fake_client)
    assert result.n_seen == 2
    assert result.n_inserted == 1
    assert result.n_deleted == 1
    store.close()


def test_sync_uses_last_sync_from_state_when_since_none(tmp_path):
    from paper_distiller.arxiv_local.incremental import sync
    from paper_distiller.arxiv_local.store import Store

    store = Store(tmp_path / "arxiv.db")
    store.save_state({"last_sync": "2026-04-01T00:00:00+00:00"})
    fake_client = MagicMock()
    fake_client.ListRecords.return_value = iter([])
    sync(store, since=None, sickle_client=fake_client)
    assert fake_client.ListRecords.call_args.kwargs["from"] == "2026-04-01"
    store.close()


def test_sync_updates_last_sync_state(tmp_path):
    from paper_distiller.arxiv_local.incremental import sync
    from paper_distiller.arxiv_local.store import Store

    store = Store(tmp_path / "arxiv.db")
    fake_client = MagicMock()
    fake_client.ListRecords.return_value = iter([_FakeRecord("2401.0")])

    before = store.load_state()
    assert before["last_sync"] is None
    sync(store, sickle_client=fake_client)
    after = store.load_state()
    assert after["last_sync"] is not None
    store.close()
