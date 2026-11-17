"""Inference client: dropped connections are retried."""

import httpx
import pytest

from src import calllog
from src.inference.client import InferenceClient
from src.models.core import Message


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


async def test_every_call_is_logged_with_input_output_and_tags(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        if '"boom"' in body:
            return httpx.Response(500, json={})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    client = InferenceClient("http://test/v1")
    client._client = httpx.AsyncClient(
        base_url="http://test/v1", transport=httpx.MockTransport(handler)
    )
    path = tmp_path / "calls.jsonl.gz"
    calllog.open_log(path)
    with calllog.tagged(phase="wake", task_id="t1", role="logical"):
        await client.chat([Message(role="user", content="hi")], "m", 0.6, 8, 7)
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat([Message(role="user", content="boom")], "m", 0.6, 8, 7)
    calllog.close()
    ok, failed = list(calllog.read(path))
    assert ok["kind"] == "model" and ok["output"] == "ok"
    assert ok["request"]["messages"] == [{"role": "user", "content": "hi"}]
    assert ok["request"]["seed"] == 7 and ok["usage"]["prompt_tokens"] == 3
    assert ok["phase"] == "wake" and ok["task_id"] == "t1" and ok["role"] == "logical"
    assert "error" in failed and "output" not in failed


def test_log_survives_a_crash_mid_write(tmp_path):
    path = tmp_path / "calls.jsonl.gz"
    log = calllog.CallLog(path)
    for i in range(50):
        log.record("x", i=i)
    log._handle.flush()
    crashed = tmp_path / "crashed.jsonl.gz"
    crashed.write_bytes(path.read_bytes())
    assert [r["i"] for r in calllog.read(crashed)] == list(range(50))
    log.close()
    resumed = calllog.CallLog(path)
    resumed.record("x", i=50)
    resumed.close()
    assert [r["i"] for r in calllog.read(path)][-1] == 50
