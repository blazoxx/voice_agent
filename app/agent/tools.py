from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """Metadata and execution handler for an agent tool."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolRegistry:
    """Registers and executes tools by name."""

    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}

    def register(
        self,
        definition: ToolDefinition,
        handler: Callable[..., Any],
    ) -> None:
        """Register a tool and its execution handler."""
        if definition.name in self._definitions:
            raise ValueError(f"Tool already registered: {definition.name}")

        self._definitions[definition.name] = definition
        self._handlers[definition.name] = handler

    def get_definitions(self) -> list[ToolDefinition]:
        """Return definitions of all registered tools."""
        return list(self._definitions.values())

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute a registered tool by name."""
        handler = self._handlers.get(name)

        if handler is None:
            raise ValueError(f"Unknown tool: {name}")

        return handler(**arguments)
