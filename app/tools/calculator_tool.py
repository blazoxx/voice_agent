from app.agent.tools import ToolDefinition, ToolRegistry
from app.tools.calculator import calculate


def register_calculator(registry: ToolRegistry) -> None:
    """Register the calculator as a Bloxi agent tool."""

    registry.register(
        ToolDefinition(
            name="calculator",
            description=(
                "Evaluate a basic arithmetic expression. "
                "Supports addition, subtraction, multiplication, division, "
                "modulo, floor division, exponentiation, and parentheses."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        ),
        calculate,
    )
