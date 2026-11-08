"""Sleep-phase fine-tuning: writes datasets and runs DPO jobs in parallel."""

from __future__ import annotations

import asyncio
import fcntl
import json
import sys
from pathlib import Path
from typing import Any

from src.config import TrainingConfig
from src.models.core import PreferencePair
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
        nll_weight=cfg.nll_weight,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        grad_accumulation=cfg.grad_accumulation,
        max_length=cfg.max_length,
    )


class TrainingError(RuntimeError):
    pass


class TrainingSlot:
    """One of N machine-wide training slots, held via an exclusive file lock.

    Runs are separate processes sharing one GPU; the lock files make sure no
    more than N DPO jobs train at once, whichever runs they belong to.
    """

    def __init__(self, slots_dir: Path, slots: int) -> None:
        self.slots_dir = slots_dir
        self.slots = slots
        self._handle: Any = None

    async def __aenter__(self) -> TrainingSlot:
        self.slots_dir.mkdir(parents=True, exist_ok=True)
        while True:
            for i in range(self.slots):
                handle = open(self.slots_dir / f"slot{i}.lock", "w")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                self._handle = handle
                return self
            await asyncio.sleep(2)

    async def __aexit__(self, *exc: object) -> None:
        fcntl.flock(self._handle, fcntl.LOCK_UN)
        self._handle.close()


async def run_jobs(jobs: dict[str, DPOJob], slots_dir: Path, slots: int) -> None:
    """Run each job in its own process, within the machine-wide slot limit."""

    async def run_one(key: str, job: DPOJob) -> None:
        out = Path(job.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        job_file = out / "job.json"
        job_file.write_text(job.model_dump_json(indent=2))
        async with TrainingSlot(slots_dir, slots):
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


def training_summary(output_dir: Path, pairs: int) -> dict[str, Any]:
    """Final DPO statistics from a finished job's log history."""
    history = json.loads((output_dir / "train_metrics.json").read_text())
    steps = [h for h in history if "loss" in h]
    last = steps[-1] if steps else {}
    return {
        "pairs": pairs,
        "steps": max((h.get("step", 0) for h in history), default=0),
        "loss": last.get("loss"),
        "reward_accuracy": last.get("rewards/accuracies"),
        "reward_margin": last.get("rewards/margins"),
        "log": steps,
    }
