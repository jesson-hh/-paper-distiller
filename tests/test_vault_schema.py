from paper_distiller.vault.schema import DEFAULT_SCHEMA, CATEGORIES


def test_schema_has_six_categories():
    assert set(CATEGORIES) == {"articles", "techniques", "directions",
                                "open-problems", "authors", "surveys"}


def test_schema_descriptions_complete():
    for cat in CATEGORIES:
        assert cat in DEFAULT_SCHEMA
        assert DEFAULT_SCHEMA[cat]  # non-empty description
