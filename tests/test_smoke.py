"""Smoke test: package imports and version is set."""
import paper_distiller


def test_package_imports():
    assert paper_distiller.__version__ == "1.2.0"
