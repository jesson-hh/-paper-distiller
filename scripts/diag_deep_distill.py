"""Diagnostic: pick one paper from the local arxiv mirror, distill it, and
report stats so we can verify v1.7's deep distillation produces 12-section
3-6k char output as expected.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Force UTF-8 on Windows so CJK and ¥ glyphs print without GBK errors
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Load .env for API key
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from paper_distiller.arxiv_local.store import Store, _default_dir
from paper_distiller.arxiv_local.search import search as local_search_fn
from paper_distiller.distill.article import distill
from paper_distiller.llm.openai_compatible import LLMClient
from paper_distiller.pipeline import fetch_with_fallback
from paper_distiller.config import load_config
from paper_distiller.vault.crosslink import WikiIndex


def main():
    # 1. Find a paper in the local mirror — prefer Yuling Jiao's known paper
    store = Store(_default_dir() / "arxiv.db")
    print(f"local DB: {store.paper_count():,} papers")

    candidates = local_search_fn(store, "Yuling Jiao", n=3)
    if not candidates:
        print("no Yuling Jiao papers in local DB — falling back to diffusion")
        candidates = local_search_fn(store, "diffusion models", n=3)

    if not candidates:
        print("no papers found locally — abort")
        return

    paper = candidates[0]
    print()
    print(f"chosen paper: {paper.arxiv_id}  {paper.published}")
    print(f"  title: {paper.title}")
    print(f"  authors: {', '.join(paper.authors[:5])}")
    print()
    store.close()

    # 2. Download PDF + extract text
    cfg = load_config(
        vault_path=r"G:\Math research Agent\wiki",
        topic="diffusion",
        source="arxiv",
    )
    print("downloading PDF + extracting text...")
    t0 = time.time()
    tmpdir = Path(tempfile.mkdtemp(prefix="diag-distill-"))
    full_text = fetch_with_fallback(paper, cfg, tmpdir)
    pdf_time = time.time() - t0
    print(f"  PDF + extract: {pdf_time:.1f}s")
    print(f"  text length: {len(full_text):,} chars")
    print()

    if len(full_text) < 1000:
        print(f"text too short ({len(full_text)} chars), aborting")
        return

    # 3. Build LLM + empty wiki index, run distill
    llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model)
    print(f"using model: {cfg.model}")
    wiki_index = WikiIndex(entries=[])  # empty — fresh test

    print("calling distill (LLM)...")
    t0 = time.time()
    article = distill(paper, full_text, wiki_index, llm)
    distill_time = time.time() - t0
    print(f"  distill LLM call: {distill_time:.1f}s")
    print(f"  tokens in/out: {llm.total_tokens_in:,} / {llm.total_tokens_out:,}")
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(f"  estimated cost: CNY {llm.estimated_cost_cny:.4f}")
    print()

    # 4. Inspect the output
    body = article.body
    print(f"=== ARTICLE OUTPUT ({len(body):,} chars) ===")
    print(f"title:  {article.title}")
    print(f"slug:   {article.slug}")
    print(f"tags:   {article.tags}")
    print(f"refs:   {article.refs}")
    print(f"depth:  {article.depth}")
    print()

    # Count sections
    section_count = body.count("\n## ")
    print(f"## sections found: {section_count}")
    expected_sections = [
        "TL;DR", "问题动因", "设定", "核心方法", "关键定理",
        "实验设置", "关键结果", "消融", "局限", "已有 wiki",
        "复现要点", "我的 take", "引用网络",
    ]
    for s in expected_sections:
        present = s in body
        marker = "✓" if present else "✗"
        print(f"  {marker} {s}")
    print()

    # Save to file for inspection
    out = Path(__file__).parent / "diag_distill_output.md"
    out.write_text(body, encoding="utf-8")
    print(f"full body saved to: {out}")
    print()
    print("=== FIRST 1500 CHARS OF BODY ===")
    print(body[:1500])
    print("...")
    print()
    print("=== LAST 500 CHARS OF BODY ===")
    print(body[-500:])


if __name__ == "__main__":
    main()
