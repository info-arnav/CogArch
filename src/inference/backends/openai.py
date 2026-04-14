"""OpenAI API backend implementation."""

import os

from openai import AsyncOpenAI

from src.inference.backends.base import LLMBackend


class OpenAIBackend(LLMBackend):
    """OpenAI API adapter for GPT models."""

    def __init__(self, api_key: str | None = None):
        self.client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
