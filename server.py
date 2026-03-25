"""FastAPI server for Math Research Agent — replaces Gradio."""

import os
import json

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import time
import io
import base64
import re
import httpx
import networkx as nx

from agent import MathResearchAgent
from tools.arxiv_tool import arxiv_search, arxiv_author_search
from llm import get_client
from tracking import get_experiment_log
from reporting.notebook_generator import NotebookGenerator
from reporting.report_generator import ReportGenerator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = FastAPI(title="Math Research Agent")
agent = MathResearchAgent()
experiment_log = get_experiment_log()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/api/chat")
async def chat(request: Request):
    """SSE streaming chat endpoint."""
    body = await request.json()
    message = body.get("message", "")
    domain = body.get("domain", "number theory")
    max_papers = body.get("max_papers", 8)

    def event_stream():
        last_text = ""
        for history, scratchpad, images in agent.stream_response(
            message, [], domain, max_papers
        ):
            # Extract latest assistant text
            current_text = ""
            if history:
                for msg in reversed(history):
                    if msg.get("role") == "assistant":
                        current_text = msg.get("content", "")
                        break

            # Send text delta
            if current_text and current_text != last_text:
                yield f"data: {json.dumps({'type': 'text', 'content': current_text}, ensure_ascii=False)}\n\n"
                last_text = current_text

            # Send images
            for img_b64 in images:
                yield f"data: {json.dumps({'type': 'image', 'content': img_b64})}\n\n"

            # Send scratchpad updates
            if scratchpad:
                yield f"data: {json.dumps({'type': 'scratchpad', 'content': scratchpad}, ensure_ascii=False)}\n\n"

        # Final done event
        yield f"data: {json.dumps({'type': 'done', 'content': last_text}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/experiments")
async def get_experiments():
    rows = experiment_log.to_dataframe_rows()
    stats = experiment_log.summary_stats()
    return JSONResponse({
        "rows": rows,
        "total": stats.get("total", 0),
        "success_rate": stats.get("success_rate", 0),
    })


@app.get("/api/report")
async def generate_report_endpoint(fmt: str = "markdown"):
    gen = ReportGenerator(experiment_log)
    filepath = gen.generate(fmt=fmt)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return JSONResponse({"filepath": filepath, "content": content})


@app.get("/api/notebook")
async def generate_notebook_endpoint():
    gen = NotebookGenerator(experiment_log)
    filepath = gen.generate()
    return JSONResponse({"filepath": filepath})


@app.get("/api/download")
async def download(path: str):
    if os.path.exists(path):
        return FileResponse(path, filename=os.path.basename(path))
    return JSONResponse({"error": "File not found"}, status_code=404)


@app.post("/api/stop-research")
async def stop_research():
    agent.stop_autonomous()
    return JSONResponse({"status": "stopped"})


@app.post("/api/autonomous")
async def autonomous_research_endpoint(request: Request):
    """Autonomous research mode — SSE streaming progress."""
    body = await request.json()
    goal = body.get("goal", "")
    domain = body.get("domain", "mathematics")
    max_iterations = int(body.get("max_iterations", 20))
    max_time = int(body.get("max_time", 600))

    if not goal:
        return JSONResponse({"error": "goal is required"}, status_code=400)

    def event_stream():
        last_text = ""
        last_plan = ""
        for phase, status_msg, plan_text, history, scratchpad, images in agent.autonomous_research(
            goal, domain, max_iterations, max_time
        ):
            # Phase/progress update
            yield f"data: {json.dumps({'type': 'phase', 'phase': phase, 'status': status_msg}, ensure_ascii=False)}\n\n"

            # Plan update
            if plan_text and plan_text != last_plan:
                yield f"data: {json.dumps({'type': 'plan', 'content': plan_text}, ensure_ascii=False)}\n\n"
                last_plan = plan_text

            # Text update
            current_text = ""
            if history:
                for msg in reversed(history):
                    if msg.get("role") == "assistant":
                        current_text = msg.get("content", "")
                        break
            if current_text and current_text != last_text:
                yield f"data: {json.dumps({'type': 'text', 'content': current_text}, ensure_ascii=False)}\n\n"
                last_text = current_text

            # Scratchpad
            if scratchpad:
                yield f"data: {json.dumps({'type': 'scratchpad', 'content': scratchpad}, ensure_ascii=False)}\n\n"

            # Images
            for img_b64 in images:
                yield f"data: {json.dumps({'type': 'image', 'content': img_b64})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'content': last_text}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ════════════════════════════════════════
# Paper Discovery + Idea Generation APIs
# ════════════════════════════════════════

@app.post("/api/papers")
async def search_papers(request: Request):
    """Search arXiv for papers by keyword."""
    body = await request.json()
    query = body.get("query", "")
    max_results = body.get("max_results", 10)
    sort_by = body.get("sort_by", "submittedDate")
    if not query.strip():
        return JSONResponse({"papers": [], "error": "Empty query"})
    result = arxiv_search(query=query, max_results=max_results, sort_by=sort_by)
    return JSONResponse(result)


@app.post("/api/auto-select")
async def auto_select(request: Request):
    """Let LLM select relevant papers from a list."""
    body = await request.json()
    papers = body.get("papers", [])
    prompt = body.get("prompt", "Select the most interesting and interconnected papers for generating novel research ideas.")

    paper_summaries = "\n".join(
        f"{i+1}. [{p['arxiv_id']}] {p['title']}\n   {p['abstract'][:200]}..."
        for i, p in enumerate(papers)
    )

    system = "You are a research advisor. Select papers from the list that have the most potential for generating novel research ideas when combined. Return ONLY a JSON object: {\"selected\": [list of arxiv_id strings], \"reasoning\": \"brief explanation\"}"
    messages = [{"role": "user", "content": f"Papers:\n{paper_summaries}\n\nSelection criteria: {prompt}"}]

    client = get_client()
    result = client.chat(system=system, messages=messages, max_tokens=1000)
    text = ""
    for block in result["content_blocks"]:
        if block["type"] == "text":
            text += block["text"]

    # Parse JSON from response
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = {"selected": [], "reasoning": text}
    except json.JSONDecodeError:
        data = {"selected": [], "reasoning": text}

    return JSONResponse(data)


@app.post("/api/generate-ideas")
async def generate_ideas(request: Request):
    """Generate research ideas based on selected papers. Streams SSE."""
    body = await request.json()
    papers = body.get("papers", [])
    user_prompt = body.get("prompt", "")

    paper_details = "\n\n".join(
        f"**[{p['arxiv_id']}] {p['title']}**\n"
        f"Authors: {', '.join(p.get('authors', []))}\n"
        f"Abstract: {p.get('abstract', '')}"
        for p in papers
    )

    system = (
        "You are a creative mathematical research advisor. Based on the provided papers, "
        "propose novel research ideas that connect, extend, or combine their findings in unexpected ways. "
        "For each idea:\n"
        "1. Give it a clear title\n"
        "2. Explain the key insight and why it's novel\n"
        "3. Outline a possible approach\n"
        "4. Identify potential challenges\n"
        "5. Rate feasibility (high/medium/low)\n\n"
        "Use LaTeX for mathematical notation. Be specific and creative."
    )

    user_msg = f"## Selected Papers\n\n{paper_details}"
    if user_prompt:
        user_msg += f"\n\n## Additional Guidance\n{user_prompt}"

    messages = [{"role": "user", "content": user_msg}]

    def event_stream():
        client = get_client()
        result_holder = {}
        for partial_text in client.stream_chat(
            system=system,
            messages=messages,
            max_tokens=4096,
            result_holder=result_holder,
        ):
            if partial_text:
                yield f"data: {json.dumps({'type': 'text', 'content': partial_text}, ensure_ascii=False)}\n\n"

        # Final
        final_text = ""
        for block in result_holder.get("blocks", []):
            if block.get("type") == "text":
                final_text += block["text"]
        yield f"data: {json.dumps({'type': 'done', 'content': final_text}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/generate-note")
async def generate_note(request: Request):
    """Generate a complete LaTeX research note from papers and ideas. Streams SSE."""
    body = await request.json()
    papers = body.get("papers", [])
    ideas = body.get("ideas", "")
    instruction = body.get("instruction", "")

    paper_details = "\n\n".join(
        f"**[{p.get('arxiv_id','')}] {p.get('title','')}**\n"
        f"Authors: {', '.join(p.get('authors', []))}\n"
        f"Abstract: {p.get('abstract', '')}"
        for p in papers
    )

    system = (
        "You are an expert mathematical research writer. Based on the provided papers and research ideas, "
        "generate a COMPLETE LaTeX document for a research note. The document must be self-contained and compilable.\n\n"
        "Required structure:\n"
        "1. \\documentclass{article} with packages: amsmath, amsthm, amssymb, hyperref\n"
        "2. \\newtheorem definitions for theorem, lemma, proposition, corollary, definition, remark\n"
        "3. \\title, \\author, \\date, \\maketitle\n"
        "4. \\begin{abstract}...\\end{abstract}\n"
        "5. \\section{Introduction} — motivation, context, main contributions\n"
        "6. \\section{Background} — definitions, known results\n"
        "7. \\section{Main Results} — theorems with full proofs\n"
        "8. \\section{Numerical Experiments} — if applicable, describe computational evidence\n"
        "9. \\section{Conclusion}\n"
        "10. \\begin{thebibliography} with references to the input papers\n\n"
        "Write mathematically rigorous content with proper LaTeX notation. "
        "Include detailed proofs, not just proof sketches. "
        "Output ONLY the LaTeX source code, no markdown wrapping or explanation."
    )

    user_msg = ""
    if paper_details:
        user_msg += f"## Reference Papers\n\n{paper_details}\n\n"
    if ideas:
        user_msg += f"## Research Ideas\n\n{ideas}\n\n"
    if instruction:
        user_msg += f"## Additional Instructions\n\n{instruction}\n\n"
    if not user_msg:
        user_msg = "Generate a sample research note on an interesting topic in number theory."

    messages = [{"role": "user", "content": user_msg}]

    def event_stream():
        client = get_client()
        result_holder = {}
        for partial_text in client.stream_chat(
            system=system,
            messages=messages,
            max_tokens=8192,
            result_holder=result_holder,
        ):
            if partial_text:
                yield f"data: {json.dumps({'type': 'text', 'content': partial_text}, ensure_ascii=False)}\n\n"

        final_text = ""
        for block in result_holder.get("blocks", []):
            if block.get("type") == "text":
                final_text += block["text"]
        yield f"data: {json.dumps({'type': 'done', 'content': final_text}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/edit-note")
async def edit_note(request: Request):
    """Edit a LaTeX note based on user instruction. Streams SSE with the full updated LaTeX."""
    body = await request.json()
    latex = body.get("latex", "")
    instruction = body.get("instruction", "").strip()
    history = body.get("history", [])

    if not latex or not instruction:
        return JSONResponse({"error": "LaTeX source and instruction are required"})

    system = (
        "You are an expert LaTeX editor for mathematical research papers. "
        "The user will provide the current LaTeX source and an editing instruction. "
        "Apply the requested changes and return the COMPLETE modified LaTeX document. "
        "Do not omit any parts — return the full document even if only a small section changed. "
        "Output ONLY the LaTeX source code, no markdown wrapping, no explanation, no ```latex blocks."
    )

    messages = list(history)
    user_msg = f"## Current LaTeX Source\n\n```\n{latex}\n```\n\n## Edit Instruction\n\n{instruction}"
    messages.append({"role": "user", "content": user_msg})

    def event_stream():
        client = get_client()
        result_holder = {}
        for partial_text in client.stream_chat(
            system=system,
            messages=messages,
            max_tokens=8192,
            result_holder=result_holder,
        ):
            if partial_text:
                yield f"data: {json.dumps({'type': 'text', 'content': partial_text}, ensure_ascii=False)}\n\n"

        final_text = ""
        for block in result_holder.get("blocks", []):
            if block.get("type") == "text":
                final_text += block["text"]
        yield f"data: {json.dumps({'type': 'done', 'content': final_text}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/design-experiment")
async def design_experiment(request: Request):
    """Design numerical/computational experiments based on research ideas or note. Streams SSE."""
    body = await request.json()
    ideas = body.get("ideas", "")
    note_latex = body.get("note_latex", "")
    instruction = body.get("instruction", "")

    system = (
        "You are an expert in designing computational and numerical experiments for mathematical research. "
        "Based on the provided research ideas and/or LaTeX note, design concrete, executable experiments.\n\n"
        "For each experiment, provide:\n"
        "1. **Experiment Title**\n"
        "2. **Objective** — What are we trying to verify or explore?\n"
        "3. **Hypothesis** — What do we expect to find?\n"
        "4. **Method** — Step-by-step procedure (algorithms, computations)\n"
        "5. **Parameters & Data** — Specific ranges, sample sizes, inputs\n"
        "6. **Expected Results** — What plots, tables, or metrics to produce\n"
        "7. **Success Criteria** — How to judge if the experiment succeeded\n\n"
        "Be specific and practical. Use concrete numbers and ranges. "
        "These experiments should be implementable in Python using numpy, scipy, matplotlib, sympy. "
        "Use LaTeX notation ($...$) for mathematical expressions."
    )

    user_msg = ""
    if ideas:
        user_msg += f"## Research Ideas\n\n{ideas}\n\n"
    if note_latex:
        user_msg += f"## Research Note (LaTeX)\n\n{note_latex[:3000]}\n\n"
    if instruction:
        user_msg += f"## Additional Instructions\n\n{instruction}\n\n"
    if not user_msg:
        user_msg = "Design a sample experiment for exploring prime number distribution."

    messages = [{"role": "user", "content": user_msg}]

    def event_stream():
        client = get_client()
        result_holder = {}
        for partial_text in client.stream_chat(
            system=system, messages=messages, max_tokens=4096, result_holder=result_holder,
        ):
            if partial_text:
                yield f"data: {json.dumps({'type': 'text', 'content': partial_text}, ensure_ascii=False)}\n\n"
        final_text = ""
        for block in result_holder.get("blocks", []):
            if block.get("type") == "text":
                final_text += block["text"]
        yield f"data: {json.dumps({'type': 'done', 'content': final_text}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/generate-code-plan")
async def generate_code_plan(request: Request):
    """Generate executable Python code for experiments. Streams SSE."""
    body = await request.json()
    experiment_plan = body.get("experiment_plan", "")
    note_latex = body.get("note_latex", "")
    instruction = body.get("instruction", "")

    system = (
        "You are an expert Python scientific computing developer. "
        "Based on the provided experiment plan, generate complete, executable Python code.\n\n"
        "Requirements:\n"
        "1. Start with a requirements section: list all pip packages needed\n"
        "2. Write well-structured, documented Python code\n"
        "3. Use numpy, scipy, matplotlib, sympy as primary libraries\n"
        "4. Include data visualization (matplotlib plots with labels, titles, legends)\n"
        "5. Include progress output (print statements for intermediate results)\n"
        "6. Handle edge cases and add input validation where needed\n"
        "7. Use functions and clear variable names\n"
        "8. Add a `if __name__ == '__main__':` block\n\n"
        "Format your output as:\n"
        "### requirements.txt\n```\nnumpy\nscipy\n...\n```\n\n"
        "### experiment.py\n```python\n# main experiment code\n```\n\n"
        "### utils.py (if needed)\n```python\n# helper functions\n```\n\n"
        "### README.md\n```\n# How to run\n...\n```\n\n"
        "Make the code ready to run with `python experiment.py`."
    )

    user_msg = ""
    if experiment_plan:
        user_msg += f"## Experiment Plan\n\n{experiment_plan}\n\n"
    if note_latex:
        user_msg += f"## Research Context (LaTeX)\n\n{note_latex[:2000]}\n\n"
    if instruction:
        user_msg += f"## Additional Instructions\n\n{instruction}\n\n"
    if not user_msg:
        user_msg = "Generate a sample experiment script for exploring prime numbers."

    messages = [{"role": "user", "content": user_msg}]

    def event_stream():
        client = get_client()
        result_holder = {}
        for partial_text in client.stream_chat(
            system=system, messages=messages, max_tokens=8192, result_holder=result_holder,
        ):
            if partial_text:
                yield f"data: {json.dumps({'type': 'text', 'content': partial_text}, ensure_ascii=False)}\n\n"
        final_text = ""
        for block in result_holder.get("blocks", []):
            if block.get("type") == "text":
                final_text += block["text"]
        yield f"data: {json.dumps({'type': 'done', 'content': final_text}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/analyze-paper")
async def analyze_paper(request: Request):
    """Generate a technical analysis report for a single paper. Streams SSE."""
    body = await request.json()
    paper = body.get("paper", {})

    if not paper.get("title"):
        return JSONResponse({"error": "No paper provided"})

    system = (
        "You are an expert mathematical research analyst. Given a paper's title, authors, abstract, "
        "and categories, produce a detailed technical analysis report covering:\n"
        "1. **Core Research Problem** — What question does this paper address?\n"
        "2. **Main Methods & Techniques** — What mathematical tools and approaches are used?\n"
        "3. **Key Innovations** — What is novel about this work?\n"
        "4. **Relation to Prior Work** — How does it differ from or build on existing results?\n"
        "5. **Potential Applications & Extensions** — Where could these results be applied or extended?\n"
        "6. **Limitations & Open Questions** — What remains unresolved?\n\n"
        "Use LaTeX notation (with $...$ and $$...$$) for mathematical expressions. "
        "Be thorough, specific, and insightful."
    )

    authors_str = ", ".join(paper.get("authors", []))
    cats_str = ", ".join(paper.get("categories", []))
    user_msg = (
        f"## Paper\n\n"
        f"**Title:** {paper.get('title', '')}\n"
        f"**Authors:** {authors_str}\n"
        f"**arXiv ID:** {paper.get('arxiv_id', '')}\n"
        f"**Categories:** {cats_str}\n\n"
        f"**Abstract:**\n{paper.get('abstract', '')}"
    )

    messages = [{"role": "user", "content": user_msg}]

    def event_stream():
        client = get_client()
        result_holder = {}
        for partial_text in client.stream_chat(
            system=system,
            messages=messages,
            max_tokens=4096,
            result_holder=result_holder,
        ):
            if partial_text:
                yield f"data: {json.dumps({'type': 'text', 'content': partial_text}, ensure_ascii=False)}\n\n"

        final_text = ""
        for block in result_holder.get("blocks", []):
            if block.get("type") == "text":
                final_text += block["text"]
        yield f"data: {json.dumps({'type': 'done', 'content': final_text}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/paper-qa")
async def paper_qa(request: Request):
    """Answer questions about a specific paper. Supports multi-turn conversation. Streams SSE."""
    body = await request.json()
    paper = body.get("paper", {})
    question = body.get("question", "").strip()
    history = body.get("history", [])

    if not paper.get("title") or not question:
        return JSONResponse({"error": "Paper and question are required"})

    authors_str = ", ".join(paper.get("authors", []))
    cats_str = ", ".join(paper.get("categories", []))
    system = (
        "You are an expert mathematician answering questions about a specific paper. "
        "Here is the paper's information:\n\n"
        f"**Title:** {paper.get('title', '')}\n"
        f"**Authors:** {authors_str}\n"
        f"**arXiv ID:** {paper.get('arxiv_id', '')}\n"
        f"**Categories:** {cats_str}\n"
        f"**Abstract:** {paper.get('abstract', '')}\n\n"
        "Answer the user's questions based on the paper's content. "
        "Use LaTeX notation ($...$ and $$...$$) for mathematical expressions. "
        "Be precise and informative. If the abstract doesn't contain enough information "
        "to fully answer, say so and provide your best analysis based on what is available."
    )

    messages = list(history) + [{"role": "user", "content": question}]

    def event_stream():
        client = get_client()
        result_holder = {}
        for partial_text in client.stream_chat(
            system=system,
            messages=messages,
            max_tokens=2048,
            result_holder=result_holder,
        ):
            if partial_text:
                yield f"data: {json.dumps({'type': 'text', 'content': partial_text}, ensure_ascii=False)}\n\n"

        final_text = ""
        for block in result_holder.get("blocks", []):
            if block.get("type") == "text":
                final_text += block["text"]
        yield f"data: {json.dumps({'type': 'done', 'content': final_text}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════

def compute_graph_layout(G) -> dict:
    """Compute spring layout and return graph as JSON with coordinates."""
    if len(G.nodes) == 0:
        return {"nodes": [], "edges": [], "stats": {"total_nodes": 0, "total_edges": 0}}

    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

    nodes = []
    for node_id in G.nodes:
        data = G.nodes[node_id]
        x, y = pos[node_id]
        nodes.append({
            "id": node_id,
            "name": data.get("name", node_id),
            "papers": data.get("papers", []),
            "paper_count": len(data.get("papers", [])),
            "x": float(x),
            "y": float(y),
            "is_center": data.get("is_center", False),
        })

    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "weight": data.get("weight", 1),
            "shared_papers": data.get("shared_papers", []),
            "directed": G.is_directed(),
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "most_connected": max(nodes, key=lambda n: n["paper_count"])["name"] if nodes else "",
        },
    }


def fetch_semantic_scholar(arxiv_id: str) -> dict | None:
    """Fetch paper data from Semantic Scholar API."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{arxiv_id}"
    fields = "title,authors,citations.title,citations.authors,citations.externalIds,references.title,references.authors,references.externalIds"
    try:
        resp = httpx.get(url, params={"fields": fields}, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# ════════════════════════════════════════
# Technology Learning APIs
# ════════════════════════════════════════

@app.post("/api/author-papers")
async def author_papers(request: Request):
    """Search arXiv for papers by a specific author."""
    body = await request.json()
    author = body.get("author", "").strip()
    max_results = body.get("max_results", 15)
    if not author:
        return JSONResponse({"papers": [], "error": "Empty author name"})
    result = arxiv_author_search(author=author, max_results=max_results)
    return JSONResponse(result)


@app.post("/api/analyze-techniques")
async def analyze_techniques(request: Request):
    """Analyze an author's proof techniques based on selected papers. Streams SSE."""
    body = await request.json()
    author = body.get("author", "")
    papers_data = body.get("papers", [])

    paper_details = "\n\n".join(
        f"**[{p['arxiv_id']}] {p['title']}** ({p.get('published', '')})\n"
        f"Authors: {', '.join(p.get('authors', []))}\n"
        f"Abstract: {p.get('abstract', '')}"
        for p in papers_data
    )

    system = (
        "You are a mathematical research analyst. Based on the provided papers by the same author (or research group), "
        "analyze their research methodology and techniques in depth. Your analysis should cover:\n\n"
        "1. **Recurring Proof Techniques**: Identify proof strategies that appear across multiple papers\n"
        "2. **Key Mathematical Tools**: Specific theorems, lemmas, or frameworks they frequently employ\n"
        "3. **Methodological Evolution**: How their approach has evolved over time\n"
        "4. **Signature Approaches**: Unique or distinctive methods that characterize this researcher\n"
        "5. **Technical Strengths**: Areas where their techniques are particularly powerful\n\n"
        "Use LaTeX for mathematical notation. Be specific — cite particular papers when discussing techniques."
    )

    user_msg = f"## Papers by {author}\n\n{paper_details}"
    messages = [{"role": "user", "content": user_msg}]

    def event_stream():
        client = get_client()
        result_holder = {}
        for partial_text in client.stream_chat(
            system=system,
            messages=messages,
            max_tokens=4096,
            result_holder=result_holder,
        ):
            if partial_text:
                yield f"data: {json.dumps({'type': 'text', 'content': partial_text}, ensure_ascii=False)}\n\n"

        final_text = ""
        for block in result_holder.get("blocks", []):
            if block.get("type") == "text":
                final_text += block["text"]
        yield f"data: {json.dumps({'type': 'done', 'content': final_text}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ════════════════════════════════════════
# Academic Relations APIs
# ════════════════════════════════════════

@app.post("/api/coauthor-network")
async def coauthor_network(request: Request):
    """Build a co-author network from a list of papers."""
    body = await request.json()
    center_author = body.get("author", "")
    papers_data = body.get("papers", [])

    G = nx.Graph()

    for paper in papers_data:
        authors = paper.get("authors", [])
        paper_info = {"title": paper.get("title", ""), "arxiv_id": paper.get("arxiv_id", ""), "published": paper.get("published", "")}

        for author in authors:
            if not G.has_node(author):
                G.add_node(author, name=author, papers=[], is_center=(author == center_author))
            G.nodes[author]["papers"].append(paper_info)

        # Add edges between all co-authors
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                if G.has_edge(authors[i], authors[j]):
                    G[authors[i]][authors[j]]["weight"] += 1
                    G[authors[i]][authors[j]]["shared_papers"].append(paper_info["title"])
                else:
                    G.add_edge(authors[i], authors[j], weight=1, shared_papers=[paper_info["title"]])

    graph_json = compute_graph_layout(G)
    return JSONResponse(graph_json)


@app.post("/api/citation-network")
async def citation_network(request: Request):
    """Build a citation network using Semantic Scholar API."""
    body = await request.json()
    arxiv_ids = body.get("arxiv_ids", [])

    G = nx.DiGraph()
    missing = []

    for i, aid in enumerate(arxiv_ids):
        if i > 0:
            time.sleep(3)  # rate limit

        data = fetch_semantic_scholar(aid)
        if not data:
            missing.append(aid)
            continue

        title = data.get("title", aid)
        authors = [a.get("name", "") for a in data.get("authors", [])]
        node_id = aid
        paper_info = {"title": title, "arxiv_id": aid}

        if not G.has_node(node_id):
            G.add_node(node_id, name=title, papers=[paper_info], is_center=True)

        # Add citations (papers that cite this one)
        for cite in (data.get("citations") or [])[:10]:
            cite_title = cite.get("title", "unknown")
            cite_ids = cite.get("externalIds") or {}
            cite_arxiv = cite_ids.get("ArXiv", cite_title[:30])
            if not G.has_node(cite_arxiv):
                G.add_node(cite_arxiv, name=cite_title, papers=[{"title": cite_title, "arxiv_id": cite_arxiv}], is_center=False)
            G.add_edge(cite_arxiv, node_id, weight=1, shared_papers=[])  # cite -> this

        # Add references (papers this one cites)
        for ref in (data.get("references") or [])[:10]:
            ref_title = ref.get("title", "unknown")
            ref_ids = ref.get("externalIds") or {}
            ref_arxiv = ref_ids.get("ArXiv", ref_title[:30])
            if not G.has_node(ref_arxiv):
                G.add_node(ref_arxiv, name=ref_title, papers=[{"title": ref_title, "arxiv_id": ref_arxiv}], is_center=False)
            G.add_edge(node_id, ref_arxiv, weight=1, shared_papers=[])  # this -> ref

    graph_json = compute_graph_layout(G)
    graph_json["missing"] = missing
    return JSONResponse(graph_json)


@app.post("/api/add-papers-to-network")
async def add_papers_to_network(request: Request):
    """Add new papers to an existing network by arxiv IDs."""
    body = await request.json()
    arxiv_ids = body.get("arxiv_ids", [])
    existing_graph = body.get("existing_graph", {"nodes": [], "edges": []})
    network_type = body.get("network_type", "coauthor")  # "coauthor" or "citation"

    # Fetch paper info from arXiv
    new_papers = []
    for aid in arxiv_ids:
        result = arxiv_search(query=f"id:{aid}", max_results=1)
        if result.get("papers"):
            new_papers.append(result["papers"][0])

    if not new_papers:
        return JSONResponse({"error": "No papers found for the given IDs", "nodes": existing_graph.get("nodes", []), "edges": existing_graph.get("edges", [])})

    # Rebuild the graph from existing + new
    if network_type == "coauthor":
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    # Restore existing nodes
    for node in existing_graph.get("nodes", []):
        G.add_node(node["id"], name=node["name"], papers=node.get("papers", []), is_center=node.get("is_center", False))

    # Restore existing edges
    for edge in existing_graph.get("edges", []):
        G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1), shared_papers=edge.get("shared_papers", []))

    if network_type == "coauthor":
        # Add new papers as co-author connections
        for paper in new_papers:
            authors = paper.get("authors", [])
            paper_info = {"title": paper.get("title", ""), "arxiv_id": paper.get("arxiv_id", ""), "published": paper.get("published", "")}
            for author in authors:
                if not G.has_node(author):
                    G.add_node(author, name=author, papers=[], is_center=False)
                G.nodes[author]["papers"].append(paper_info)
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    if G.has_edge(authors[i], authors[j]):
                        G[authors[i]][authors[j]]["weight"] += 1
                        G[authors[i]][authors[j]]["shared_papers"].append(paper_info["title"])
                    else:
                        G.add_edge(authors[i], authors[j], weight=1, shared_papers=[paper_info["title"]])
    else:
        # Citation: fetch from Semantic Scholar
        for paper in new_papers:
            aid = paper.get("arxiv_id", "")
            if not aid:
                continue
            time.sleep(3)
            data = fetch_semantic_scholar(aid)
            if not data:
                continue
            title = data.get("title", aid)
            if not G.has_node(aid):
                G.add_node(aid, name=title, papers=[{"title": title, "arxiv_id": aid}], is_center=False)
            for cite in (data.get("citations") or [])[:10]:
                ct = cite.get("title", "unknown")
                cids = cite.get("externalIds") or {}
                ca = cids.get("ArXiv", ct[:30])
                if not G.has_node(ca):
                    G.add_node(ca, name=ct, papers=[{"title": ct, "arxiv_id": ca}], is_center=False)
                G.add_edge(ca, aid, weight=1, shared_papers=[])
            for ref_item in (data.get("references") or [])[:10]:
                rt = ref_item.get("title", "unknown")
                rids = ref_item.get("externalIds") or {}
                ra = rids.get("ArXiv", rt[:30])
                if not G.has_node(ra):
                    G.add_node(ra, name=rt, papers=[{"title": rt, "arxiv_id": ra}], is_center=False)
                G.add_edge(aid, ra, weight=1, shared_papers=[])

    graph_json = compute_graph_layout(G)
    return JSONResponse(graph_json)


if __name__ == "__main__":
    api_key = os.environ.get("API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    model = os.environ.get("MODEL") or os.environ.get("ANTHROPIC_MODEL", "")
    if not api_key or not model:
        print("ERROR: Missing API_KEY or MODEL in .env")
        exit(1)

    provider = os.environ.get("LLM_PROVIDER", "openai")
    base_url = (os.environ.get("BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL", "")).strip()
    print(f"Provider: {provider}")
    print(f"Model:    {model}")
    print(f"Base URL: {base_url or '(default)'}")
    print(f"\n  Open browser: http://127.0.0.1:7861\n")

    uvicorn.run(app, host="127.0.0.1", port=7861)
