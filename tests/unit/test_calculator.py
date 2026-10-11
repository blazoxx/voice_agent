import pytest

from app.tools.calculator import calculate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3", 5),
        ("10 - 4", 6),
        ("6 * 7", 42),
        ("20 / 4", 5),
        ("2 ** 3", 8),
        ("(2 + 3) * 4", 20),
        ("-5 + 2", -3),
    ],
)
def test_calculate_valid_expressions(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "1 / 0",
        "import os",
        "__import__('os').system('whoami')",
        "2 ** 1000",
        "True + 1",
    ],
)
def test_calculate_rejects_invalid_or_unsafe_expressions(expression):
    with pytest.raises(ValueError):
        calculate(expression)
