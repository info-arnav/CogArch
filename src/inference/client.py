"""Client for an OpenAI-compatible server with runtime LoRA loading.

On the GPU server this is vLLM with `--enable-lora` and
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True (see scripts/serve_vllm.sh); on a laptop
it is src/inference/local_server.py. Every fine-tuned adapter is served under
its own name, so each specialist and coordinator runs its own weights on one
base model. With `on_delta`, the reply is streamed and passed on as it grows.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

import httpx

from src import calllog
from src.models.core import Message

SERVER_WAIT_SECONDS = 900.0
MAX_BACKOFF_SECONDS = 10.0


def _backoff(attempt: int) -> float:
    return float(min(MAX_BACKOFF_SECONDS, 0.5 * 2**attempt))


OnDelta = Callable[[str], None]


class ChatModel(Protocol):
    async def chat(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int,
        on_delta: OnDelta | None = None,
        choices: list[str] | None = None,
    ) -> str: ...


class InferenceClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 300.0,
        max_concurrent_requests: int = 64,
        api_key: str | None = None,
    ) -> None:
        # Expire idle connections before the server's 5s keep-alive does, so a
        # request is never written to a socket the server is closing.
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            limits=httpx.Limits(keepalive_expiry=2.0),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
        )
        self._slots = asyncio.Semaphore(max_concurrent_requests)
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def _count(self, usage: dict | None) -> dict | None:
        self.usage["calls"] += 1
        for k in ("prompt_tokens", "completion_tokens"):
            self.usage[k] += int((usage or {}).get(k) or 0)
        return usage

    async def _logged(
        self,
        payload: dict[str, Any],
        call: Awaitable[tuple[str, dict[str, Any]]],
    ) -> str:
        """Run a request and add it, input and output, to the run's call log."""
        start = time.perf_counter()
        try:
            text, details = await call
        except BaseException as error:
            calllog.record(
                "model",
                server=str(self._client.base_url),
                request=payload,
                error=repr(error),
                seconds=round(time.perf_counter() - start, 3),
            )
            raise
        calllog.record(
            "model",
            server=str(self._client.base_url),
            request=payload,
            output=text,
            **details,
            seconds=round(time.perf_counter() - start, 3),
        )
        return text

    async def chat(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int,
        on_delta: OnDelta | None = None,
        choices: list[str] | None = None,
    ) -> str:
        """With `choices`, the reply is constrained to exactly one of them
        (vLLM guided decoding) and the model's probabilities for its first
        token are logged."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        if choices is not None:
            payload |= {"guided_choice": choices, "logprobs": True, "top_logprobs": 10}
            on_delta = None
        return await self._logged(payload, self._chat(payload, on_delta))

    async def _chat(
        self, payload: dict, on_delta: OnDelta | None
    ) -> tuple[str, dict[str, Any]]:
        async with self._slots:
            if on_delta is not None:
                return await self._stream(payload, on_delta)
            response = await self._post_with_retry("/chat/completions", payload)
        return self._reply(response)

    def _reply(self, response: httpx.Response) -> tuple[str, dict[str, Any]]:
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        details: dict[str, Any] = {"usage": self._count(body.get("usage"))}
        if choice.get("logprobs"):
            details["logprobs"] = choice["logprobs"]
        return str(choice["message"]["content"] or ""), details

    async def _stream(
        self, payload: dict, on_delta: OnDelta
    ) -> tuple[str, dict[str, Any]]:
        """The reply as server-sent events. A dropped connection before the
        first token is retried; after it, the request fails."""
        waited, attempt = 0.0, 0
        while True:
            parts: list[str] = []
            try:
                usage: dict | None = None
                async with self._client.stream(
                    "POST",
                    "/chat/completions",
                    json={
                        **payload,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        chunk = json.loads(line[6:])
                        usage = chunk.get("usage") or usage
                        choices = chunk.get("choices") or []
                        delta = (
                            choices[0].get("delta", {}).get("content")
                            if choices
                            else None
                        )
                        if delta:
                            parts.append(delta)
                            on_delta(delta)
                return "".join(parts), {"usage": self._count(usage)}
            except httpx.TransportError:
                if parts or waited >= SERVER_WAIT_SECONDS:
                    raise
                waited += _backoff(attempt)
                await asyncio.sleep(_backoff(attempt))
                attempt += 1

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
        return await self._logged(payload, self._complete(payload))

    async def _complete(self, payload: dict) -> tuple[str, dict[str, Any]]:
        async with self._slots:
            response = await self._post_with_retry("/chat/completions", payload)
        return self._reply(response)

    async def _post_with_retry(self, path: str, payload: dict) -> httpx.Response:
        """POST, waiting up to SERVER_WAIT_SECONDS for a server that is down or
        restarting (dropped connections, 502/503). Seeded requests are idempotent."""
        waited, attempt = 0.0, 0
        while True:
            try:
                response = await self._client.post(path, json=payload)
                if (
                    response.status_code not in (502, 503)
                    or waited >= SERVER_WAIT_SECONDS
                ):
                    return response
            except httpx.TransportError:
                if waited >= SERVER_WAIT_SECONDS:
                    raise
            waited += _backoff(attempt)
            await asyncio.sleep(_backoff(attempt))
            attempt += 1

    async def wait_ready(self, timeout: float = SERVER_WAIT_SECONDS) -> None:
        """Wait until the server answers /models, or fail with a clear error."""
        waited, attempt = 0.0, 0
        while True:
            try:
                if (await self._client.get("/models")).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            if waited >= timeout:
                raise RuntimeError(f"No model server at {self._client.base_url}")
            waited += _backoff(attempt)
            await asyncio.sleep(_backoff(attempt))
            attempt += 1

    async def load_adapter(self, name: str, path: Path) -> None:
        """Register a LoRA adapter under `name`. `path` must exist on the server host."""
        response = await self._client.post(
            "/load_lora_adapter",
            json={"lora_name": name, "lora_path": str(path.resolve())},
        )
        calllog.record(
            "adapter_load",
            name=name,
            path=str(path.resolve()),
            status=response.status_code,
        )
        if response.is_error:
            raise RuntimeError(f"vLLM refused adapter {name}: {response.text}")

    async def unload_adapter(self, name: str) -> None:
        """An adapter the server does not have (a resumed run's adapters from
        before a server restart) is already unloaded."""
        response = await self._client.post(
            "/unload_lora_adapter", json={"lora_name": name}
        )
        calllog.record("adapter_unload", name=name, status=response.status_code)
        if response.is_error and response.status_code != 404:
            raise RuntimeError(f"vLLM could not unload adapter {name}: {response.text}")

    async def aclose(self) -> None:
        await self._client.aclose()
