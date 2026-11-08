"""Server smoke test: every external piece the experiment depends on.

1. vLLM answers a chat request on the base model.
2. The sandbox runs code and reports pass/fail.
3. A DPO job trains a LoRA adapter on the GPU.
4. vLLM loads that adapter at runtime and answers with it.
5. Throughput: how many concurrent requests the server sustains.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from rich.console import Console

from src.config import Config
from src.execution.sandbox import Sandbox
from src.inference.client import InferenceClient
from src.models.core import Message, PreferencePair, Task
from src.training.trainer import make_job, run_jobs, write_dataset

_PROMPT = [
    Message(role="system", content="You write Python."),
    Message(role="user", content="Write a function add(a, b) that returns a + b."),
]


async def run(cfg: Config, console: Console) -> None:
    client = InferenceClient(
        cfg.inference.base_url,
        cfg.inference.request_timeout_seconds,
        max_concurrent_requests=512,
    )
    base = cfg.model.served_name
    work = cfg.run_dir / "smoke"
    work.mkdir(parents=True, exist_ok=True)

    console.print("1. Base model chat")
    reply = await client.chat(_PROMPT, base, 0.0, 128, seed=0)
    console.print(f"   ok: {reply[:80]!r}")

    console.print("2. Sandbox")
    sandbox = Sandbox(cfg.environment.feedback)
    problem = Task(
        task_id="t",
        domain="code",
        kind="code",
        prompt="",
        entry_point="add",
        tests=["assert add(1, 2) == 3"],
    )
    good = await sandbox.run_tests("def add(a, b):\n    return a + b", problem)
    bad = await sandbox.run_tests("def add(a, b):\n    return a - b", problem)
    assert good.passed and not bad.passed and bad.feedback == "Your solution is wrong."
    console.print("   ok")

    console.print("3. DPO training job")
    pairs = [
        PreferencePair(
            task_id=f"p{i}",
            domain="code",
            agent="a",
            role="logical",
            prompt=_PROMPT,
            chosen="```python\ndef add(a, b):\n    return a + b\n```",
            rejected="```python\ndef add(a, b):\n    return a - b\n```",
            source="self",
        )
        for i in range(8)
    ]
    dataset = work / "pairs.jsonl"
    write_dataset(pairs, dataset)
    adapter = work / "adapter"
    start = time.perf_counter()
    await run_jobs(
        {
            "smoke": make_job(
                cfg.training, cfg.model.base, None, dataset, adapter, seed=0
            )
        },
        cfg.training.slots_dir,
        cfg.training.global_slots,
    )
    console.print(f"   ok ({time.perf_counter() - start:.0f}s)")

    console.print("4. Runtime LoRA loading")
    name = f"{cfg.experiment.name}-smoke-adapter"
    await client.load_adapter(name, Path(adapter))
    reply = await client.chat(_PROMPT, name, 0.0, 128, seed=0)
    await client.unload_adapter(name)
    console.print(f"   ok: {reply[:80]!r}")

    console.print("5. Throughput")
    for concurrency in (8, 32, 64, 128, 256):
        start = time.perf_counter()
        replies = await asyncio.gather(
            *(
                client.chat(_PROMPT, base, 0.7, cfg.model.max_tokens, seed=i)
                for i in range(concurrency)
            )
        )
        elapsed = time.perf_counter() - start
        console.print(
            f"   {concurrency:>3} concurrent: {elapsed:5.1f}s "
            f"({concurrency / elapsed:5.1f} req/s, "
            f"{sum(len(r) for r in replies) / 4 / elapsed:7.0f} ~tok/s)"
        )
    await client.aclose()
