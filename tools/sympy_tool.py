import sympy as sp
from sympy import (
    symbols, sympify, diff, integrate, solve, simplify,
    expand, factor, limit, series, oo, latex, pi, E, I,
)


def symbolic_compute(
    expression: str,
    operation: str,
    variables: list = None,
    params: dict = None,
) -> dict:
    params = params or {}
    var_names = variables or ["x"]

    # Create symbol objects
    sym_vars = {name: symbols(name) for name in var_names}
    x = sym_vars.get("x", symbols("x"))

    # Namespace for safe sympify parsing
    local_ns = {**sym_vars, "oo": oo, "pi": pi, "E": E, "I": I}
    # Add common sympy functions
    for fn_name in [
        "sin", "cos", "tan", "exp", "log", "sqrt", "abs",
        "sinh", "cosh", "tanh", "asin", "acos", "atan",
        "factorial", "binomial", "gamma", "zeta",
    ]:
        if hasattr(sp, fn_name):
            local_ns[fn_name] = getattr(sp, fn_name)

    try:
        expr = sympify(expression, locals=local_ns)
    except Exception as e:
        return {"error": f"Could not parse expression '{expression}': {e}"}

    try:
        result = _dispatch_operation(expr, operation, sym_vars, x, local_ns, params)
    except Exception as e:
        return {"error": str(e), "expression": expression, "operation": operation}

    if result is None:
        return {"error": f"Unknown operation: {operation}"}

    try:
        result_simplified = str(simplify(result))
    except Exception:
        result_simplified = str(result)

    return {
        "expression": expression,
        "operation": operation,
        "result": str(result),
        "latex": latex(result),
        "simplified": result_simplified,
    }


def _dispatch_operation(expr, operation, sym_vars, x, local_ns, params):
    if operation == "differentiate":
        order = params.get("order", 1)
        wrt = sym_vars.get(params.get("wrt", "x"), x)
        return diff(expr, wrt, order)

    elif operation == "integrate":
        wrt = sym_vars.get(params.get("wrt", "x"), x)
        return integrate(expr, wrt)

    elif operation == "definite_integral":
        wrt = sym_vars.get(params.get("wrt", "x"), x)
        lower = sympify(str(params.get("lower", 0)), locals=local_ns)
        upper = sympify(str(params.get("upper", 1)), locals=local_ns)
        return integrate(expr, (wrt, lower, upper))

    elif operation == "solve":
        eq_str = params.get("equation")
        if eq_str:
            eq_expr = sympify(eq_str, locals=local_ns)
            return solve(eq_expr, list(sym_vars.values()))
        else:
            return solve(expr, x)

    elif operation == "simplify":
        return simplify(expr)

    elif operation == "expand":
        return expand(expr)

    elif operation == "factor":
        return factor(expr)

    elif operation == "limit":
        point = sympify(str(params.get("point", 0)), locals=local_ns)
        direction = params.get("direction", "+")
        return limit(expr, x, point, direction)

    elif operation == "series":
        point = sympify(str(params.get("point", 0)), locals=local_ns)
        order = params.get("order", 6)
        return series(expr, x, point, order)

    elif operation == "evaluate":
        at = params.get("at")
        if at is not None:
            return expr.subs(x, at).evalf()
        return expr.evalf()

    return None
