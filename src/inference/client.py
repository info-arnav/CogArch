"""Client for an OpenAI-compatible vLLM server with runtime LoRA loading.

The server runs with `--enable-lora` and VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
(see scripts/serve_vllm.sh). Every fine-tuned adapter is served under its own
name, so each specialist and coordinator runs its own weights on one base model.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

import httpx

from src.models.code import Message


class ChatModel(Protocol):
    async def chat(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> str: ...


class InferenceClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 300.0,
        max_concurrent_requests: int = 64,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout_seconds
        )
        self._slots = asyncio.Semaphore(max_concurrent_requests)

    async def chat(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        async with self._slots:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": model,
                    "messages": [m.model_dump() for m in messages],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "seed": seed,
                },
            )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"] or "")

    async def load_adapter(self, name: str, path: Path) -> None:
        """Register a LoRA adapter under `name`. `path` must exist on the server host."""
        response = await self._client.post(
            "/load_lora_adapter",
            json={"lora_name": name, "lora_path": str(path.resolve())},
        )
        if response.is_error:
            raise RuntimeError(f"vLLM refused adapter {name}: {response.text}")

    async def unload_adapter(self, name: str) -> None:
        response = await self._client.post(
            "/unload_lora_adapter", json={"lora_name": name}
        )
        if response.is_error:
            raise RuntimeError(f"vLLM could not unload adapter {name}: {response.text}")

    async def aclose(self) -> None:
        await self._client.aclose()
