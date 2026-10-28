"""Safe replacement of MATLAB's ``str2num`` for the numeric text fields.

``str2num`` runs ``eval`` on the text, so anything typed in a field is EXECUTED as MATLAB code. Here only
numbers, ``+ - * / ^``, parentheses, ``pi`` / ``inf`` / ``nan`` and lists of such values are understood.
Invalid text gives an empty list, like ``str2num``; non-finite results are rejected too (a NaN or
infinite background, pixel size or filter order is meaningless).
"""
from __future__ import annotations

import ast
import math
import operator
import re

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CONST = {"pi": math.pi, "inf": math.inf, "nan": math.nan}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id.lower() in _CONST:
        return _CONST[node.id.lower()]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def _value(text: str) -> float:
    return _eval(ast.parse(text.replace("^", "**").strip(), mode="eval"))


def str2num(text: str) -> list[float]:
    """Numbers contained in ``text`` (``[]`` if it is not numeric). ``'2*3'`` -> [6.0]; ``'1 2 3'`` -> [1, 2, 3]."""
    text = text.strip().strip("[]").strip()
    if not text:
        return []
    try:
        values = [_value(text)]
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError, MemoryError):
        try:
            values = [_value(t) for t in re.split(r"[\s,;]+", text) if t]
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError, MemoryError):
            return []
    return values if values and all(math.isfinite(v) for v in values) else []
