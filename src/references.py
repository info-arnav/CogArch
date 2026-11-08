"""Reference systems on the same benchmarks, for comparison with the brain.

  plain     a single model, greedy, standard prompts (e.g. Qwen2.5-3B / 7B,
            the parameter-matched comparison)
  majority  the 1.5B model answering N times with majority vote
            (self-consistency; N = 9 matches the brain's 9 calls per problem)

Results accumulate in <output_dir>/references/report.json, one entry per system;
re-running a system replaces its entry.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from src.config import Config
from src.eval import scoring
from src.inference.client import InferenceClient
from src.models.core import BenchmarkScores, ExperimentReport
from src.tasks import BENCHMARKS


async def evaluate_reference(
    cfg: Config,
    label: str,
    model: str,
    base_url: str,
    majority: int,
    console: Console,
) -> Path:
    client = InferenceClient(
        base_url,
        cfg.inference.request_timeout_seconds,
        cfg.inference.max_concurrent_requests,
    )
    mode = f"majority@{majority}" if majority > 1 else "plain"
    answerer = (
        scoring.majority_answerer(client, model, cfg.model.max_tokens, majority)
        if majority > 1
        else scoring.plain_answerer(client, model, cfg.model.max_tokens)
    )
    run_dir = cfg.experiment.output_dir / "references"
    entry = BenchmarkScores(system=label, mode=mode)
    for bench in cfg.evaluation.benchmarks:
        tasks = BENCHMARKS[bench]()
        console.print(f"  {label} · {mode} · {bench} ({len(tasks)})")
        entry.scores[bench] = await scoring.score(
            bench,
            tasks,
            answerer,
            1,
            run_dir
            / "benchmarks"
            / f"{label}_{mode}".replace("/", "_")
            / f"{bench}.jsonl",
        )
    await client.aclose()

    path = run_dir / "report.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    report = (
        ExperimentReport.model_validate_json(path.read_text())
        if path.exists()
        else ExperimentReport(config=cfg.model_dump(mode="json"))
    )
    report.baseline_benchmarks = [
        e for e in report.baseline_benchmarks if (e.system, e.mode) != (label, mode)
    ] + [entry]
    path.write_text(report.model_dump_json(indent=2))
    console.print_json(data=json.loads(entry.model_dump_json()))
    return path
