from .arxiv_tool import arxiv_search
from .sympy_tool import symbolic_compute
from .proof_tool import proof_assist
from .code_tool import run_code
from .log_tool import log_experiment
from .report_tool import generate_report

TOOL_DEFINITIONS = [
    {
        "name": "arxiv_search",
        "description": (
            "Search arxiv.org for academic mathematics papers. "
            "Returns titles, authors, abstracts, dates, and URLs. "
            "Use this to survey a research domain, find recent work, "
            "or locate specific papers by topic. Always use this first "
            "when exploring a new mathematical domain."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query. E.g., 'Langlands program automorphic forms' or 'Riemann hypothesis zeros'"
                },
                "domain": {
                    "type": "string",
                    "description": (
                        "Math domain name or arxiv category code. "
                        "Domain names: 'algebraic topology', 'number theory', 'differential geometry', "
                        "'complex analysis', 'combinatorics', 'representation theory', "
                        "'functional analysis', 'category theory', 'probability theory', "
                        "'partial differential equations', 'algebraic geometry', 'logic'. "
                        "Or use arxiv codes directly: math.AT, math.NT, math.DG, etc. "
                        "Use empty string to search all math."
                    ),
                    "default": ""
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of papers to return (1-20)",
                    "default": 8
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                    "description": "How to sort results. Use 'submittedDate' to find the most recent work.",
                    "default": "relevance"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "symbolic_compute",
        "description": (
            "Perform exact symbolic mathematics using SymPy. "
            "Can differentiate, integrate, solve equations, expand, factor, "
            "simplify, compute limits, series expansions, and evaluate expressions. "
            "Always prefer this over approximate numerical computation for exact results. "
            "Returns both the result as a string and in LaTeX format."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Mathematical expression in Python/SymPy syntax. "
                        "Use ** for powers, * for multiplication. "
                        "E.g., 'x**3 * sin(x)', 'exp(-x**2)', 'x**2 + 2*x + 1', "
                        "'Matrix([[1, 2], [3, 4]])'"
                    )
                },
                "operation": {
                    "type": "string",
                    "enum": [
                        "differentiate",
                        "integrate",
                        "definite_integral",
                        "solve",
                        "simplify",
                        "expand",
                        "factor",
                        "limit",
                        "series",
                        "evaluate"
                    ],
                    "description": "The symbolic operation to perform"
                },
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Variable names to use. E.g., ['x'], ['x', 'y']. Defaults to ['x']."
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Operation-specific parameters: "
                        "differentiate: {'order': 2, 'wrt': 'x'}. "
                        "definite_integral: {'lower': 0, 'upper': 'oo', 'wrt': 'x'}. "
                        "limit: {'point': 0, 'direction': '+'}. "
                        "series: {'point': 0, 'order': 6}. "
                        "solve: {'equation': 'x**2 - 4'} (or leave empty to solve expression=0). "
                        "evaluate: {'at': 3.14}."
                    )
                }
            },
            "required": ["expression", "operation"]
        }
    },
    {
        "name": "proof_assist",
        "description": (
            "Assist with mathematical proof construction. "
            "Decomposes theorems into proof steps, suggests proof strategies "
            "(direct, contradiction, induction, contrapositive, construction), "
            "traces logical dependencies, and identifies needed lemmas. "
            "Returns a structured proof outline with step-by-step reasoning. "
            "Note: provides structured reasoning assistance, not formal verification."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "theorem": {
                    "type": "string",
                    "description": "The mathematical statement to prove or analyze. State it precisely."
                },
                "context": {
                    "type": "string",
                    "description": "Known definitions, axioms, previously proven lemmas, and mathematical context relevant to the proof.",
                    "default": ""
                },
                "strategy": {
                    "type": "string",
                    "enum": ["auto", "direct", "contradiction", "induction", "contrapositive", "construction", "exhaustion"],
                    "description": "Preferred proof strategy. Use 'auto' to let the system choose.",
                    "default": "auto"
                },
                "mode": {
                    "type": "string",
                    "enum": ["outline", "detailed", "lemmas"],
                    "description": "outline=high-level steps only, detailed=full proof attempt, lemmas=identify needed lemmas",
                    "default": "detailed"
                }
            },
            "required": ["theorem"]
        }
    },
    {
        "name": "run_code",
        "description": (
            "Execute Python code for numerical experiments, data analysis, and visualization. "
            "Available libraries: numpy, scipy, matplotlib, sympy, mpmath, networkx, pandas, math, cmath. "
            "Use for: numerical root finding, plotting functions, testing conjectures numerically, "
            "high-precision arithmetic (mpmath), graph theory computations. "
            "For plots: save to buffer and print as 'IMG:<base64>' to display inline. "
            "Code runs in a restricted sandbox with a timeout."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python code to execute. Use print() for text output. "
                        "For matplotlib plots use: "
                        "import io, base64; buf=io.BytesIO(); plt.savefig(buf, format='png', dpi=100, bbox_inches='tight'); buf.seek(0); print('IMG:' + base64.b64encode(buf.getvalue()).decode())"
                    )
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds (1-30)",
                    "default": 15
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "log_experiment",
        "description": (
            "Log a research experiment or significant finding for tracking. "
            "Call this after completing a meaningful research step: a literature search, "
            "a proof attempt, a computation, or a code experiment. "
            "This builds a record of the research session for later analysis and report generation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The research question or task that was investigated"
                },
                "method": {
                    "type": "string",
                    "description": "The approach used: 'literature_review', 'symbolic_computation', 'proof_attempt', 'numerical_experiment', 'conjecture_test', etc."
                },
                "tools_used": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tool names used: ['arxiv_search'], ['symbolic_compute', 'run_code'], etc."
                },
                "result_summary": {
                    "type": "string",
                    "description": "Concise summary of what was found or achieved. Be specific about mathematical results."
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the research step achieved its goal"
                },
                "domain": {
                    "type": "string",
                    "description": "Mathematical domain, e.g., 'number theory', 'algebraic topology'"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorization, e.g., ['open_problem', 'conjecture', 'verified']",
                    "default": []
                }
            },
            "required": ["question", "method", "result_summary", "success"]
        }
    },
    {
        "name": "generate_report",
        "description": (
            "Generate a structured research report summarizing all experiments, "
            "literature, computations, and proofs from the current session. "
            "Call this after completing a research investigation to produce a "
            "publishable-quality markdown or LaTeX report."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Report title. Auto-generated from session data if omitted."
                },
                "format": {
                    "type": "string",
                    "enum": ["markdown", "latex"],
                    "description": "Output format",
                    "default": "markdown"
                },
                "include_code": {
                    "type": "boolean",
                    "description": "Whether to include code experiments inline",
                    "default": True
                }
            },
            "required": []
        }
    }
]

TOOL_DISPATCH = {
    "arxiv_search": arxiv_search,
    "symbolic_compute": symbolic_compute,
    "proof_assist": proof_assist,
    "run_code": run_code,
    "log_experiment": log_experiment,
    "generate_report": generate_report,
}


def dispatch_tool(name: str, inputs: dict) -> dict:
    if name not in TOOL_DISPATCH:
        return {"error": f"Unknown tool: {name}"}
    try:
        return TOOL_DISPATCH[name](**inputs)
    except Exception as e:
        return {"error": str(e), "tool": name}
