"""CogArch CLI — code competition self-improvement pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from dotenv import load_dotenv

if TYPE_CHECKING:
    from src.inference.backends.base import LLMBackend

app = typer.Typer(name="cogarch", help="CogArch — Code Competition Self-Improvement")


def _build_specialists(cfg: dict, backend: LLMBackend) -> dict:
    from src.inference.code_specialist import CodeSpecialist

    model = cfg["model"]["ollama_id"]
    max_attempts = cfg["specialists"]["max_attempts_per_problem"]
    return {
        name: CodeSpecialist(
            name=name, backend=backend, model=model, max_attempts=max_attempts
        )
        for name in cfg["specialists"]["enabled"]
    }


@app.command()
def code_eval(
    benchmark: str = typer.Argument("humaneval", help="Benchmark to evaluate on"),
    config_path: str = typer.Option("config/default.yaml", "--config", "-c"),
    limit: int = typer.Option(
        0, "--limit", "-n", help="Cap number of problems (0 = all)"
    ),
) -> None:
    """Evaluate current model on HumanEval (Pass@1). No training, no feedback."""
    load_dotenv()
    import asyncio

    from rich.console import Console

    from src.config import load_config
    from src.eval.benchmarks import humaneval, mbpp
    from src.eval.code_experiment import CodeExperimentRunner
    from src.inference.backends.ollama import OllamaBackend
    from src.inference.code_orchestrator import CodeOrchestrator

    console = Console()
    cfg = load_config(config_path)
    backend = OllamaBackend(base_url=cfg["ollama"]["base_url"])
    specialists = _build_specialists(cfg, backend)
    agent = CodeOrchestrator(specialists=specialists)

    async def _run() -> None:
        if benchmark == "humaneval":
            problems = humaneval.load(limit=limit or None)
        else:
            problems = mbpp.load(limit=limit or None)

        console.print(
            f"\n[bold]Evaluating on {benchmark}[/bold] ({len(problems)} problems)"
        )
        runner = CodeExperimentRunner(
            agent_a=agent,
            agent_b=agent,
            finetuner=None,
            registry=None,
            console=console,
        )
        score = await runner._eval_humaneval(problems)
        console.print(f"\n[bold green]Pass@1: {score:.3f}[/bold green]")

    asyncio.run(_run())


@app.command()
def code_experiment(
    cycles: int = typer.Option(10, "--cycles", "-n"),
    rounds_per_cycle: int = typer.Option(30, "--rounds", "-r"),
    config_path: str = typer.Option("config/default.yaml", "--config", "-c"),
    output_dir: str = typer.Option("data/experiments/code", "--output", "-o"),
) -> None:
    """Full self-improvement loop: compete on MBPP → DPO fine-tune → eval HumanEval."""
    load_dotenv()
    import asyncio

    from rich.console import Console

    from src.config import load_config
    from src.eval.code_experiment import CodeExperimentRunner
    from src.inference.backends.ollama import OllamaBackend
    from src.inference.code_orchestrator import CodeOrchestrator
    from src.training.finetuner import SpecialistFinetuner
    from src.training.model_registry import ModelRegistry

    console = Console()
    cfg = load_config(config_path)
    backend = OllamaBackend(base_url=cfg["ollama"]["base_url"])

    specialists_a = _build_specialists(cfg, backend)
    specialists_b = _build_specialists(cfg, backend)
    agent_a = CodeOrchestrator(specialists=specialists_a)
    agent_b = CodeOrchestrator(specialists=specialists_b)

    ft_cfg = cfg.get("finetuning", {})
    finetuner: SpecialistFinetuner | None = None
    registry: ModelRegistry | None = None
    if ft_cfg.get("enabled", False):
        finetuner = SpecialistFinetuner(
            base_model=ft_cfg["base_model"],
            lora_rank=ft_cfg.get("lora_rank", 16),
            lora_alpha=ft_cfg.get("lora_alpha", 16),
            epochs=ft_cfg.get("epochs", 3),
            batch_size=ft_cfg.get("batch_size", 2),
            grad_accumulation=ft_cfg.get("grad_accumulation", 8),
            max_seq_length=ft_cfg.get("max_seq_length", 2048),
            min_examples=ft_cfg.get("min_examples", 20),
            output_dir=ft_cfg.get("output_dir", "models"),
            console=console,
        )
        registry = ModelRegistry(ft_cfg.get("registry_path", "models/registry.json"))

    runner = CodeExperimentRunner(
        agent_a=agent_a,
        agent_b=agent_b,
        finetuner=finetuner,
        registry=registry,
        cycles=cycles,
        rounds_per_cycle=rounds_per_cycle,
        output_dir=output_dir,
        console=console,
    )

    asyncio.run(runner.run())


if __name__ == "__main__":
    app()
