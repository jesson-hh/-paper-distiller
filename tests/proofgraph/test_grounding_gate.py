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
