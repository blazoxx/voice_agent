import pytest

from app.agent.tools import ToolRegistry
from app.tools.calculator_tool import register_calculator


def test_calculator_registers_with_registry():
    registry = ToolRegistry()

    register_calculator(registry)

    definitions = registry.get_definitions()

    assert len(definitions) == 1
    assert definitions[0].name == "calculator"
    assert "arithmetic expression" in definitions[0].description


def test_registered_calculator_executes_expression():
    registry = ToolRegistry()
    register_calculator(registry)

    result = registry.execute(
        "calculator",
        {"expression": "(10 + 5) * 2"},
    )

    assert result == 30


def test_registered_calculator_rejects_unsafe_expression():
    registry = ToolRegistry()
    register_calculator(registry)

    with pytest.raises(ValueError):
        registry.execute(
            "calculator",
            {"expression": "__import__('os').system('whoami')"},
        )


def test_registered_calculator_rejects_unknown_arguments():
    registry = ToolRegistry()
    register_calculator(registry)

    with pytest.raises(TypeError):
        registry.execute(
            "calculator",
            {"expression": "2 + 2", "unexpected": True},
        )
