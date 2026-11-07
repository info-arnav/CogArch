"""Sleep-phase fine-tuning: writes datasets and runs DPO jobs in parallel."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from src.config import TrainingConfig
from src.models.code import PreferencePair
from src.training.dpo_job import DPOJob

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_dataset(pairs: list[PreferencePair], path: Path) -> None:
    """TRL conversational preference format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for pair in pairs:
            row = {
                "prompt": [m.model_dump() for m in pair.prompt],
                "chosen": [{"role": "assistant", "content": pair.chosen}],
                "rejected": [{"role": "assistant", "content": pair.rejected}],
            }
            f.write(json.dumps(row) + "\n")


def make_job(
    cfg: TrainingConfig,
    base_model: str,
    init_adapter: Path | None,
    dataset: Path,
    output_dir: Path,
    seed: int,
) -> DPOJob:
    return DPOJob(
        base_model=base_model,
        init_adapter=str(init_adapter) if init_adapter else None,
        dataset=str(dataset),
        output_dir=str(output_dir),
        seed=seed,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        grad_accumulation=cfg.grad_accumulation,
        max_length=cfg.max_length,
    )


class TrainingError(RuntimeError):
    pass


async def run_jobs(jobs: dict[str, DPOJob], parallel: int) -> None:
    """Run each job in its own process, at most `parallel` at a time."""
    slots = asyncio.Semaphore(parallel)

    async def run_one(key: str, job: DPOJob) -> None:
        out = Path(job.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        job_file = out / "job.json"
        job_file.write_text(job.model_dump_json(indent=2))
        async with slots:
            with open(out / "train.log", "w") as log:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "src.training.dpo_job",
                    str(job_file),
                    cwd=PROJECT_ROOT,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                )
                code = await proc.wait()
        if code != 0:
            raise TrainingError(
                f"DPO job {key} failed (exit {code}); see {out / 'train.log'}"
            )

    await asyncio.gather(*(run_one(k, j) for k, j in jobs.items()))
