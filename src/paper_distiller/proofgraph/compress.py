"""LLMLingua-based prompt compression with a transparent identity fallback.

LLMLingua is an optional dependency (``pip install paper-distiller[compress]``).
If it is absent — or if any error occurs during compression — the function
returns the original text unchanged.  Callers must never handle ImportError;
they just get a string back.

Usage::

    from paper_distiller.proofgraph.compress import compress
    shorter = compress(memory_text, target_ratio=0.5)
"""
from __future__ import annotations

# Module-level singleton so we pay the model-loading cost once per process.
_compressor = None
_compressor_loaded = False  # True once we attempted to load (even if it failed)


def _get_compressor():
    global _compressor, _compressor_loaded
    if _compressor_loaded:
        return _compressor
    _compressor_loaded = True
    try:
        from llmlingua import PromptCompressor  # type: ignore[import-not-found]
        _compressor = PromptCompressor()
    except Exception:
        _compressor = None
    return _compressor


def compress(
    text: str,
    instruction: str | None = None,
    target_ratio: float = 0.5,
) -> str:
    """Compress *text* to approximately *target_ratio* of its original length.

    Falls back to returning *text* unchanged if LLMLingua is unavailable or
    raises any exception.  Never raises.

    Only the running memory / injected context should be compressed — the
    segment being read must stay verbatim for the grounding gate.
    """
    if not text:
        return text
    try:
        compressor = _get_compressor()
        if compressor is None:
            return text
        kwargs: dict = {"rate": target_ratio, "force_tokens": ["\n"]}
        if instruction:
            kwargs["instruction"] = instruction
        result = compressor.compress_prompt(text, **kwargs)
        if isinstance(result, dict):
            compressed = result.get("compressed_prompt", text)
        elif isinstance(result, str):
            compressed = result
        else:
            compressed = text
        return compressed if compressed else text
    except Exception:
        return text
