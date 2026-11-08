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

TRANSPORT_RETRIES = 5


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
        # Expire idle connections before the server's 5s keep-alive does, so a
        # request is never written to a socket the server is closing.
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            limits=httpx.Limits(keepalive_expiry=2.0),
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
        payload = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        async with self._slots:
            response = await self._post_with_retry("/chat/completions", payload)
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"] or "")

    async def complete_after(
        self,
        messages: list[Message],
        prefill: str,
        model: str,
        max_tokens: int,
    ) -> str:
        """Greedy continuation of an assistant message that starts with `prefill`
        (vLLM's continue_final_message), as EvalPlus does for chat models."""
        payload = {
            "model": model,
            "messages": [m.model_dump() for m in messages]
            + [{"role": "assistant", "content": prefill}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "continue_final_message": True,
            "add_generation_prompt": False,
        }
        async with self._slots:
            response = await self._post_with_retry("/chat/completions", payload)
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"] or "")

    async def _post_with_retry(self, path: str, payload: dict) -> httpx.Response:
        """POST, retrying dropped connections. Seeded requests are idempotent."""
        for attempt in range(TRANSPORT_RETRIES):
            try:
                return await self._client.post(path, json=payload)
            except httpx.TransportError:
                if attempt == TRANSPORT_RETRIES - 1:
                    raise
                await asyncio.sleep(0.5 * 2**attempt)
        raise AssertionError("unreachable")

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
