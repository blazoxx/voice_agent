from abc import ABC, abstractmethod

from groq import Groq

from app.config import settings


class LLMProvider(ABC):
    """Interface for language model providers."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Generate a response from the language model."""
        raise NotImplementedError


class GroqProvider(LLMProvider):
    """Groq implementation of the LLM provider."""

    def __init__(self) -> None:
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.llm_model

    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return response.choices[0].message.content or ""