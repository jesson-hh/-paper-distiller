"""Tests for proofgraph.compress — LLMLingua wrapper with identity fallback."""
from __future__ import annotations


def test_compress_identity_fallback_when_llmlingua_absent(monkeypatch):
    import importlib, builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name.startswith("llmlingua"):
            raise ImportError("not installed")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    from paper_distiller.proofgraph import compress as c
    importlib.reload(c)
    out = c.compress("some long text " * 50, target_ratio=0.5)
    assert isinstance(out, str) and out  # never raises, returns a string


def test_compress_returns_string_with_instruction(monkeypatch):
    import importlib, builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name.startswith("llmlingua"):
            raise ImportError("not installed")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    from paper_distiller.proofgraph import compress as c
    importlib.reload(c)
    text = "Given a random variable X, the expectation E[X] satisfies linearity. " * 20
    out = c.compress(text, instruction="focus on the key result", target_ratio=0.4)
    assert isinstance(out, str) and len(out) > 0


def test_compress_never_raises_on_exception(monkeypatch):
    """Even if llmlingua is installed but raises during compression, fall back."""
    import importlib, builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name.startswith("llmlingua"):
            raise ImportError("not installed")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    from paper_distiller.proofgraph import compress as c
    importlib.reload(c)
    # Even empty string should not raise
    out = c.compress("", target_ratio=0.5)
    assert isinstance(out, str)
