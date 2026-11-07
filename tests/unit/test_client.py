"""Inference client: dropped connections are retried."""

import httpx

from src.inference.client import InferenceClient
from src.models.code import Message


async def test_chat_retries_dropped_connection():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ReadError("connection reset", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = InferenceClient("http://test/v1")
    client._client = httpx.AsyncClient(
        base_url="http://test/v1", transport=httpx.MockTransport(handler)
    )
    reply = await client.chat([Message(role="user", content="hi")], "m", 0.0, 8, 0)
    assert reply == "ok" and calls["n"] == 3
