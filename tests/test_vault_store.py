"""Tests for VaultStore — Obsidian markdown CRUD."""
from pathlib import Path

import pytest

from paper_distiller.vault.store import VaultStore, slugify


def test_slugify_ascii():
    assert slugify("Hello World") == "hello-world"


def test_slugify_cjk_fallback():
    # CJK titles slug-fallback to hashed slug
    s = slugify("跨股相关性扩散增强")
    assert s.startswith("entry-")
    assert len(s) > 6


def test_vault_store_creates_directories(tmp_vault: Path):
    store = VaultStore(tmp_vault)
    assert (tmp_vault / "articles").is_dir()
    assert (tmp_vault / "surveys").is_dir()


def test_save_and_read_entry_roundtrip(tmp_vault: Path):
    store = VaultStore(tmp_vault)
    meta = store.save_entry(
        title="Test Article",
        category="articles",
        body="# Test\n\nBody with [[other-slug]] link.",
        tags=["test", "demo"],
        refs=["arxiv:1234.5678"],
    )
    assert meta["slug"] == "test-article"
    entry = store.read_entry("articles", "test-article")
    assert entry is not None
    assert entry.title == "Test Article"
    assert entry.tags == ["test", "demo"]
    assert "[[other-slug]]" in entry.body


def test_slug_exists(tmp_vault: Path):
    store = VaultStore(tmp_vault)
    assert not store.slug_exists("articles", "test-article")
    store.save_entry(title="Test Article", category="articles", body="x")
    assert store.slug_exists("articles", "test-article")


def test_invalid_category_raises(tmp_vault: Path):
    store = VaultStore(tmp_vault)
    with pytest.raises(ValueError, match="Invalid category"):
        store.save_entry(title="x", category="nonsense", body="x")


def test_list_entries(tmp_vault: Path):
    store = VaultStore(tmp_vault)
    store.save_entry(title="A", category="articles", body="b")
    store.save_entry(title="B", category="articles", body="b")
    store.save_entry(title="T", category="techniques", body="b")
    all_entries = store.list_entries()
    assert len(all_entries) == 3
    only_articles = store.list_entries(category="articles")
    assert len(only_articles) == 2


def test_empty_body_rejected(tmp_vault: Path):
    store = VaultStore(tmp_vault)
    with pytest.raises(ValueError, match="body is required"):
        store.save_entry(title="t", category="articles", body="")
