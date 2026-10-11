import ast
import operator
from typing import Union


Number = Union[int, float]

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate(node: ast.AST) -> Number:
    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise ValueError("Only numeric values are allowed.")
        return node.value

    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported arithmetic operator.")

        left = _evaluate(node.left)
        right = _evaluate(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is too large.")

        result = operation(left, right)

        if isinstance(result, (int, float)) and abs(result) > 1e100:
            raise ValueError("Result is too large.")

        return result

    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported unary operator.")
        return operation(_evaluate(node.operand))

    raise ValueError("Invalid arithmetic expression.")


def calculate(expression: str) -> Number:
    """Safely evaluate a basic arithmetic expression."""
    if not expression.strip():
        raise ValueError("Expression cannot be empty.")

    if len(expression) > 200:
        raise ValueError("Expression is too long.")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree.body)
    except (SyntaxError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("Invalid or undefined arithmetic expression.") from exc

    if isinstance(result, float) and not __import__("math").isfinite(result):
        raise ValueError("Result must be finite.")

    return result
