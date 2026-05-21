"""Smoke test: package imports, version is set, and the public API resolves."""
import paper_distiller


def test_package_imports():
    assert paper_distiller.__version__ == "1.12.0"


def test_public_api_resolves():
    """Every name advertised in __all__ must resolve (guards the lazy PEP 562 exports)."""
    for name in paper_distiller.__all__:
        assert getattr(paper_distiller, name) is not None, name
