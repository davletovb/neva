"""Mathematical tool implementation."""

from __future__ import annotations

import ast
import logging
import operator
from typing import Callable, Dict

from neva.agents.base import Tool
from neva.utils.exceptions import ToolExecutionError

from ._builtin_schemas import BUILTIN_TEXT_ARGUMENT_SCHEMA

logger = logging.getLogger(__name__)


_ALLOWED_OPERATORS: Dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


_MAX_ABS_BASE = 1e6
_MAX_ABS_EXPONENT = 32


class MathTool(Tool):
    """Safely evaluate basic mathematical expressions."""

    def __init__(self) -> None:
        super().__init__(
            "calculator",
            "performs arithmetic expressions",
            capabilities=["math", "calculation"],
            argument_schema=BUILTIN_TEXT_ARGUMENT_SCHEMA,
        )

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPERATORS:
                raise ToolExecutionError(f"Unsupported operator: {op_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if op_type is ast.Pow and (abs(left) > _MAX_ABS_BASE or abs(right) > _MAX_ABS_EXPONENT):
                raise ToolExecutionError("Exponentiation exceeds the allowed range")
            return _ALLOWED_OPERATORS[op_type](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._eval_node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise ToolExecutionError("Unsupported expression")

    def use(self, task: str) -> str:
        try:
            expression = ast.parse(task, mode="eval")
            result = self._eval_node(expression.body)
            return str(result)
        except ToolExecutionError:
            raise
        except Exception as exc:
            logger.warning("MathTool evaluation failed: %s", exc)
            raise ToolExecutionError(f"Failed to use MathTool: {exc}") from exc
