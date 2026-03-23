import ast
import io
import traceback
import threading
from contextlib import redirect_stdout, redirect_stderr

ALLOWED_IMPORTS = {
    "numpy", "scipy", "matplotlib", "sympy", "mpmath",
    "math", "cmath", "itertools", "functools", "collections",
    "networkx", "pandas", "json", "re", "io", "base64",
    "fractions", "decimal", "random", "statistics", "typing",
    "numbers", "operator", "heapq", "bisect",
}


def run_code(code: str, timeout: int = 15) -> dict:
    timeout = min(max(1, timeout), 30)

    # Validate imports via AST
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}", "output": "", "success": False}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top and top not in ALLOWED_IMPORTS:
                    return {
                        "error": f"Import '{top}' is not allowed in sandbox.",
                        "allowed_imports": sorted(ALLOWED_IMPORTS),
                        "output": "",
                        "success": False,
                    }
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top and top not in ALLOWED_IMPORTS:
                return {
                    "error": f"Import '{top}' is not allowed in sandbox.",
                    "allowed_imports": sorted(ALLOWED_IMPORTS),
                    "output": "",
                    "success": False,
                }

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    exec_error = [None]

    def execute():
        try:
            exec_globals = {"__builtins__": __builtins__, "__name__": "__main__"}
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(code, "<math_agent>", "exec"), exec_globals)
        except Exception:
            exec_error[0] = traceback.format_exc()

    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return {
            "error": f"Execution timed out after {timeout}s",
            "output": stdout_buf.getvalue(),
            "success": False,
            "timed_out": True,
        }

    raw_output = stdout_buf.getvalue()
    stderr_output = stderr_buf.getvalue()
    error = exec_error[0] or stderr_output

    # Separate text lines from embedded images
    images = []
    text_lines = []
    for line in raw_output.split("\n"):
        if line.startswith("IMG:"):
            images.append(line[4:])
        else:
            text_lines.append(line)

    return {
        "output": "\n".join(text_lines).strip(),
        "stderr": error.strip() if error else "",
        "images": images,
        "success": not bool(exec_error[0]),
    }
