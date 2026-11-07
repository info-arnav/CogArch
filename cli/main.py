"""CogArch CLI."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from src.config import Config, load_config

app = typer.Typer(
    name="cogarch", help="CogArch — a community of agents learning from 'wrong'"
)


def _replicate(cfg: Config, replicate: int) -> Config:
    """Give replicate N its own run name and agent seeds."""
    if replicate > 1:
        cfg.agents.seeds = {
            n: s + 1000 * (replicate - 1) for n, s in cfg.agents.seeds.items()
        }
    cfg.experiment.name = f"{cfg.experiment.name}-r{replicate}"
    return cfg


@app.command()
def run(
    config: str = typer.Argument(..., help="Path to an experiment config"),
    replicate: int = typer.Option(1, min=1, help="Replicate number (changes seeds)"),
) -> None:
    """Run an experiment: baseline, cycles of compete → sleep → validate, final eval."""
    from src.experiment import Experiment

    cfg = _replicate(load_config(config), replicate)
    asyncio.run(Experiment(cfg, Console()).run())


@app.command()
def smoke(config: str = typer.Argument("config/smoke.yaml")) -> None:
    """Check the server: vLLM, sandbox, DPO training, runtime LoRA, throughput."""
    from src import smoke as smoke_test

    asyncio.run(smoke_test.run(load_config(config), Console()))


@app.command()
def dashboard(
    runs: str = typer.Option("data/runs", help="Directory containing run folders"),
    port: int = typer.Option(8080),
) -> None:
    """Serve the live dashboard on 127.0.0.1 (use an SSH tunnel to view it)."""
    from pathlib import Path

    from src.dashboard import serve

    serve(Path(runs), port=port)


if __name__ == "__main__":
    app()
