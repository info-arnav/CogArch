"""Speed of each shared resource before the full run: a model server under
load, the grader over the network, and the code sandbox. Prints one JSON object.

    GRADER_URL=... VERIFIER_API_KEY=... python3 scripts/runpod/speedtest.py
"""

import asyncio
import json
import os
import statistics
import sys
import time

import httpx

sys.path.insert(0, ".")
from src.config import FeedbackLevel
from src.execution.sandbox import Sandbox
from src.models.core import Task

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SERVER = os.environ.get("SERVER_URL", "http://localhost:8200/v1")
GRADER = os.environ.get("GRADER_URL", "http://localhost:8300/v1")
KEY = os.environ.get("VERIFIER_API_KEY", "")


async def timed(
    client: httpx.AsyncClient, url: str, body: dict, headers: dict
) -> tuple[float, int]:
    start = time.perf_counter()
    r = await client.post(url, json=body, headers=headers)
    r.raise_for_status()
    return time.perf_counter() - start, r.json()["usage"]["completion_tokens"]


async def load(url: str, body: dict, headers: dict, n: int) -> dict:
    async with httpx.AsyncClient(timeout=300) as client:
        start = time.perf_counter()
        results = await asyncio.gather(
            *(timed(client, url, {**body, "seed": i}, headers) for i in range(n))
        )
        wall = time.perf_counter() - start
    lat = sorted(r[0] for r in results)
    tokens = sum(r[1] for r in results)
    return {
        "requests": n,
        "wall_s": round(wall, 1),
        "req_per_s": round(n / wall, 1),
        "tokens_per_s": round(tokens / wall),
        "p50_s": round(statistics.median(lat), 2),
        "p95_s": round(lat[int(0.95 * n) - 1], 2),
    }


async def server_test() -> dict:
    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Solve step by step: a shop sells pens at 3 for $2. How much do 12 pens cost?",
            }
        ],
        "max_tokens": 300,
        "temperature": 0.6,
    }
    return {
        f"concurrent_{n}": await load(f"{SERVER}/chat/completions", body, {}, n)
        for n in (16, 64, 256)
    }


async def grader_test() -> dict:
    prompt = "You are a diligent and precise assistant tasked with evaluating the correctness of responses.\nQuestion: How much do 12 pens cost at 3 for $2?\nOutput sentence: The answer is $8.\nCorrect answer: 8\nJudgement:"
    body = {
        "model": "IAAR-Shanghai/xVerify-3B-Ia",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {KEY}"} if KEY else {}
    return {
        f"concurrent_{n}": await load(f"{GRADER}/chat/completions", body, headers, n)
        for n in (16, 128, 512)
    }


async def sandbox_test() -> dict:
    sandbox = Sandbox(FeedbackLevel.BINARY, timeout_seconds=10)
    task = Task(
        task_id="speed/add",
        domain="code",
        kind="code",
        prompt="add",
        entry_point="add",
        tests=["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
    )
    code = "def add(a, b):\n    return a + b"
    out = {}
    for n in (16, 64, 256):
        start = time.perf_counter()
        await asyncio.gather(*(sandbox.run_tests(code, task) for _ in range(n)))
        wall = time.perf_counter() - start
        out[f"concurrent_{n}"] = {
            "runs": n,
            "wall_s": round(wall, 2),
            "runs_per_s": round(n / wall, 1),
        }
    return out


async def main() -> None:
    print(
        json.dumps(
            {
                "server": await server_test(),
                "grader": await grader_test(),
                "sandbox": await sandbox_test(),
                "cpus": os.cpu_count(),
            },
            indent=1,
        )
    )


asyncio.run(main())
