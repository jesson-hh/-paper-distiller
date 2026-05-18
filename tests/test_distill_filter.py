import json
from unittest.mock import MagicMock

from paper_distiller.sources.arxiv import ArxivPaper
from paper_distiller.distill.filter import rank


def _papers(n):
    return [
        ArxivPaper(source="arxiv", paper_id=f"25{i:02d}.00001",
                   arxiv_id=f"25{i:02d}.00001", title=f"P{i}", authors=[],
                   abstract=f"abstract {i}", pdf_url="", published="2025",
                   categories=[])
        for i in range(n)
    ]


def test_rank_filters_to_top_n():
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "selected": [
            {"arxiv_id": "2502.00001", "relevance_score": 9.0, "reason": "best"},
            {"arxiv_id": "2500.00001", "relevance_score": 7.5, "reason": "ok"},
        ]
    })
    top = rank(_papers(5), "test topic", top_n=2, llm=llm)
    assert len(top) == 2
    assert top[0].arxiv_id == "2502.00001"
    assert top[1].arxiv_id == "2500.00001"
    llm.complete.assert_called_once()


def test_rank_skips_invented_arxiv_ids():
    """If LLM hallucinates an arxiv_id not in candidates, drop it."""
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "selected": [
            {"arxiv_id": "2502.00001", "relevance_score": 9.0, "reason": "real"},
            {"arxiv_id": "9999.99999", "relevance_score": 8.0, "reason": "fake"},
        ]
    })
    top = rank(_papers(5), "topic", top_n=2, llm=llm)
    assert len(top) == 1  # invented id dropped
    assert top[0].arxiv_id == "2502.00001"
