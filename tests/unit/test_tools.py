import pytest

from app.agent.tools import ToolDefinition, ToolRegistry


def test_register_and_execute_tool():
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="greet",
            description="Returns a greeting.",
            parameters={"name": "string"},
        ),
        lambda name: f"Hello, {name}!",
    )

    result = registry.execute("greet", {"name": "Bloxi"})

    assert result == "Hello, Bloxi!"


def test_registered_tool_definition_is_available():
    registry = ToolRegistry()

    definition = ToolDefinition(
        name="calculator",
        description="Performs calculations.",
        parameters={"expression": "string"},
    )

    registry.register(definition, lambda expression: expression)

    definitions = registry.get_definitions()

    assert len(definitions) == 1
    assert definitions[0].name == "calculator"
    assert definitions[0].description == "Performs calculations."


def test_duplicate_tool_registration_is_rejected():
    registry = ToolRegistry()
    definition = ToolDefinition(
        name="greet",
        description="Returns a greeting.",
    )

    registry.register(definition, lambda: "Hello")

    with pytest.raises(ValueError, match="Tool already registered"):
        registry.register(definition, lambda: "Another greeting")


def test_unknown_tool_execution_is_rejected():
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Unknown tool"):
        registry.execute("missing_tool", {})


def test_tool_arguments_are_forwarded_to_handler():
    registry = ToolRegistry()

    def add(a: int, b: int) -> int:
        return a + b

    registry.register(
        ToolDefinition(
            name="add",
            description="Adds two integers.",
        ),
        add,
    )

    result = registry.execute("add", {"a": 7, "b": 5})

    assert result == 12
