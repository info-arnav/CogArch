"""CogArch CLI."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from src.config import Config, load_config

app = typer.Typer(
    name="cogarch", help="CogArch — a community of agents learning from 'wrong'"
)


def _with_model(cfg: Config, model: str | None) -> Config:
    """Override the base model (served by vLLM under its Hugging Face id)."""
    if model:
        cfg.model.base = model
        cfg.model.served_name = model
    return cfg


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
    model: str = typer.Option("", help="Override the base model"),
) -> None:
    """Run an experiment: baseline, cycles of compete → sleep → validate, final eval."""
    from src.experiment import Experiment

    cfg = _replicate(_with_model(load_config(config), model or None), replicate)
    asyncio.run(Experiment(cfg, Console()).run())


@app.command()
def smoke(config: str = typer.Argument("config/smoke.yaml")) -> None:
    """Check the server: vLLM, sandbox, DPO training, runtime LoRA, throughput."""
    from src import smoke as smoke_test

    asyncio.run(smoke_test.run(load_config(config), Console()))


@app.command()
def calibrate(
    config: str = typer.Argument("config/pilot.yaml"),
    model: str = typer.Option("", help="Override the base model"),
    problems: int = typer.Option(100, help="Training problems to measure on"),
) -> None:
    """Measure an untrained team: solve rate, validation, output-format compliance."""
    from src.calibration import calibrate as run_calibration

    cfg = _with_model(load_config(config), model or None)
    print(asyncio.run(run_calibration(cfg, problems, Console())))


@app.command()
def dashboard(
    runs: str = typer.Option("data/runs", help="Directory containing run folders"),
    port: int = typer.Option(8080),
) -> None:
    """Serve the live dashboard on 127.0.0.1 (public link: scripts/tunnel.sh)."""
    from pathlib import Path

    from src.dashboard import serve

    serve(Path(runs), port=port)


@app.command()
def analyze(
    runs: str = typer.Option("data/runs", help="Directory containing run folders"),
    out: str = typer.Option("data/analysis", help="Where to write tables and figures"),
) -> None:
    """Aggregate finished runs into tables (summary.md) and figures."""
    from pathlib import Path

    from src.analysis import analyze as run_analysis

    print(run_analysis(Path(runs), Path(out)))


if __name__ == "__main__":
    app()
