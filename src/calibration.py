"""Model calibration: how well an untrained team does, before any experiment.

Used to pick the model. The community only gets learning signal when an
agent sometimes fails and its partner sometimes succeeds, so the best model
solves roughly 40–60% of training problems, still has headroom on the
benchmarks, and follows the coordinator and probe formats reliably.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from src.config import Config
from src.eval.benchmarks import mbpp
from src.execution.sandbox import Sandbox
from src.inference.agent import Agent, Role
from src.inference.client import InferenceClient
from src.inference.prompts import load_coordinator, load_specialist


async def calibrate(cfg: Config, problems: int, console: Console) -> Path:
    client = InferenceClient(
        cfg.inference.base_url,
        cfg.inference.request_timeout_seconds,
        cfg.inference.max_concurrent_requests,
    )
    sandbox = Sandbox(
        cfg.environment.feedback,
        cfg.environment.execution_timeout_seconds,
        cfg.environment.memory_limit_mb,
    )
    base = cfg.model.served_name
    agent = Agent(
        name="a",
        seed=cfg.agents.seeds[cfg.agents.names[0]],
        specialists={n: Role(load_specialist(n), base) for n in cfg.agents.specialists},
        coordinator=Role(load_coordinator(), base),
        client=client,
        sandbox=sandbox,
        memory=None,
        max_attempts=cfg.environment.max_attempts,
        max_tokens=cfg.model.max_tokens,
        probes_per_problem=cfg.environment.probes_per_problem,
    )
    train, validation = mbpp.split(
        mbpp.load_pool(), cfg.data.validation_size, cfg.data.split_seed
    )
    sample = train[:problems]

    start = time.perf_counter()
    results = await asyncio.gather(*(agent.solve(p) for p in sample))
    solve_seconds = time.perf_counter() - start
    probes = await asyncio.gather(*(agent.write_probes(p) for p in sample))
    answers = await asyncio.gather(*(agent.answer(p) for p in validation))
    verdicts = await asyncio.gather(
        *(sandbox.run_tests(code, p) for code, p in zip(answers, validation))
    )

    decisions = [d for r in results for d in r.decisions]
    report: dict[str, Any] = {
        "model": cfg.model.base,
        "problems": len(sample),
        "first_attempt_solved": sum(r.decisions[0].result.passed for r in results)
        / len(results),
        "solved_within_attempts": sum(r.solved for r in results) / len(results),
        "coordinator_valid_format": sum(d.choice is not None for d in decisions)
        / len(decisions),
        "coordinator_combined": sum(d.choice == "combined" for d in decisions)
        / len(decisions),
        "probes_parsed": sum(bool(p) for p, _ in probes) / len(probes),
        "validation_pass_at_1": sum(v.passed for v in verdicts) / len(verdicts),
        "seconds_per_problem": solve_seconds / len(sample),
    }
    out = Path("data/calibration")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{cfg.model.base.replace('/', '__')}.json"
    path.write_text(json.dumps(report, indent=2))
    console.print_json(data=report)
    await client.aclose()
    return path
