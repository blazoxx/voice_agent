from abc import ABC, abstractmethod
from typing import Any

from groq import Groq

from app.config import settings


class LLMProvider(ABC):
    """Interface for language model providers."""

    @abstractmethod
    def generate(self, messages: list[dict[str, Any]]) -> str:
        """Generate a text response from the language model."""
        raise NotImplementedError

    @abstractmethod
    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """Generate a response that may contain tool calls."""
        raise NotImplementedError


class GroqProvider(LLMProvider):
    """Groq implementation of the LLM provider."""

    def __init__(self) -> None:
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.llm_model

    def generate(self, messages: list[dict[str, Any]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return response.choices[0].message.content or ""

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """Return the model's message, including any requested tool calls."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        return response.choices[0].message
