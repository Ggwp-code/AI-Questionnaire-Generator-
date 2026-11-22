"""Check an independent solver's arithmetic against the writer's answer."""

from __future__ import annotations

import ast
import operator
import re
from fractions import Fraction

from tribunal.generation.draft import (
    NUMERICAL_ANSWER_TYPES,
    Draft,
    DraftInvalid,
    parse_json_object,
    parse_number,
)

COMPUTATIONAL_TYPES = NUMERICAL_ANSWER_TYPES | {"trace", "computational_mcq"}

_OPERATIONS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _calculate(expression: str) -> Fraction:
    if not isinstance(expression, str) or len(expression) > 256:
        raise ValueError
    expression = expression.strip()
    if not re.fullmatch(r"[0-9eE.+*/%() \t-]+", expression):
        raise ValueError
    tree = ast.parse(expression, mode="eval")
    nodes = list(ast.walk(tree))
    if len(nodes) > 64 or not any(isinstance(node, ast.BinOp) for node in nodes):
        raise ValueError

    def calculate(node: ast.expr) -> Fraction:
        if isinstance(node, ast.Constant):
            return parse_number(ast.get_source_segment(expression, node) or "")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = calculate(node.operand)
            return -operand if isinstance(node.op, ast.USub) else operand
        if not isinstance(node, ast.BinOp):
            raise ValueError
        left, right = calculate(node.left), calculate(node.right)
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right) > 64:
                raise ValueError
            value = left ** int(right)
        else:
            operation = _OPERATIONS.get(type(node.op))
            if operation is None:
                raise ValueError
            value = operation(left, right)
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > 1024:
            raise ValueError
        return Fraction(value)

    return calculate(tree.body)


def verify_solution(reply_text: str, draft: Draft, question_type: str, tolerance: float) -> tuple[str, str]:
    """Compare plain numeric values with relative tolerance against the solver's result."""
    if len(reply_text) > 8192:
        return "invalid", "Verifier reply exceeds the size limit."
    try:
        reply = parse_json_object(reply_text)
        if reply.keys() == {"unsupported"} and reply["unsupported"] is True:
            return "unsupported", "Verifier could not express the solution as bounded arithmetic."
        if "expression" not in reply or reply.keys() - {"expression", "option_index"}:
            raise ValueError
        is_mcq = question_type == "computational_mcq"
        if "option_index" in reply and (not is_mcq or type(reply["option_index"]) is not int):
            raise ValueError
        result = _calculate(reply["expression"])
        relative_tolerance = parse_number(str(tolerance))
        if relative_tolerance < 0:
            raise ValueError
        margin = abs(result) * relative_tolerance
    except (DraftInvalid, ValueError, SyntaxError, ArithmeticError, RecursionError):
        return "invalid", "Invalid verifier reply or arithmetic calculation."

    if question_type not in COMPUTATIONAL_TYPES:
        return "unsupported", "Question type does not support arithmetic verification."
    try:
        answer = parse_number(draft.answer)
    except ValueError:
        if question_type == "trace":
            return "unsupported", "Only a trace with a single numeric output can be verified."
        return "mismatch", "Expected a plain numeric answer."

    if is_mcq:
        try:
            options = [parse_number(option) for option in draft.options or []]
        except ValueError:
            return "mismatch", "Computational options must be plain numbers."
        matching = [i for i, option in enumerate(options) if abs(result - option) <= margin]
        if len(matching) != 1:
            return "mismatch", "The calculation must match exactly one option."
        index = matching[0]
        if answer != options[index]:
            return "mismatch", "The numeric answer differs from the correct option."
        if reply.get("option_index", index) != index:
            return "invalid", "Verifier option_index disagrees with its own calculation."
        if draft.correct_option != index:
            return "mismatch", "The selected option disagrees with the calculation."
    if abs(result - answer) > margin:
        return "mismatch", "The answer disagrees with the calculation."
    return "pass", "The independent calculation agrees with the answer."
