"""Abstract base class for LLM backends."""

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Interface for LLM API calls."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send messages to an LLM and return the text response."""
        ...
