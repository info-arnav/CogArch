"""CogArch CLI — main entry point."""

import typer
from dotenv import load_dotenv

app = typer.Typer(name="cogarch", help="CogArch — Parallel Cognitive Architecture")


@app.command()
def infer(
    query: str = typer.Argument(..., help="The question or input to process"),
    no_revision: bool = typer.Option(
        False, "--no-revision", help="Skip Round 2 revision pass"
    ),
    config_path: str = typer.Option(
        "config/default.yaml", "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Run the full inference pipeline on a query."""
    load_dotenv()

    import asyncio

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from src.config import load_all_specialist_configs, load_config
    from src.inference.backends.ollama import OllamaBackend
    from src.inference.coordinator import Coordinator
    from src.inference.orchestrator import Orchestrator
    from src.inference.specialist import Specialist
    from src.memory.experience_log import ExperienceLog
    from src.models.interaction import InteractionRecord

    console = Console()

    async def _run() -> None:
        # Load config
        cfg = load_config(config_path)
        ollama_cfg = cfg.get("ollama", {})
        backend = OllamaBackend(
            base_url=ollama_cfg.get("base_url", "http://localhost:11434")
        )

        # Build specialists
        specialist_names = cfg["specialists"]["enabled"]
        specialist_configs = load_all_specialist_configs(specialist_names)
        default_model = cfg["specialists"]["model_id"]

        specialists = {
            name: Specialist(config, backend, default_model)
            for name, config in specialist_configs.items()
        }

        # Build coordinator
        coordinator = Coordinator(
            model=cfg["coordinator"]["model_id"],
            backend=backend,
            temperature=cfg["coordinator"]["temperature"],
            max_tokens=cfg["coordinator"]["max_tokens"],
        )

        # Build orchestrator
        orchestrator = Orchestrator(
            specialists=specialists,
            coordinator=coordinator,
            enable_revision=not no_revision
            and cfg["inference"]["enable_revision_pass"],
        )

        console.print(f"\n[bold]Query:[/bold] {query}")
        console.print(f"[dim]Specialists: {', '.join(specialist_names)}[/dim]")
        console.print(
            f"[dim]Revision pass: {'disabled' if no_revision else 'enabled'}[/dim]\n"
        )

        with console.status("[bold green]Specialists thinking..."):
            result = await orchestrator.run(query)

        # Display results
        console.print(Panel(result.answer, title="Final Answer", border_style="green"))

        # Attribution table
        table = Table(title="Attribution")
        table.add_column("Specialist", style="cyan")
        table.add_column("Weight", justify="right")
        table.add_column("R1 Confidence", justify="right")
        table.add_column("R2 Confidence", justify="right")
        for name, weight in sorted(
            result.attribution.items(), key=lambda x: x[1], reverse=True
        ):
            r1_conf = (
                f"{result.round1_outputs[name].confidence:.2f}"
                if name in result.round1_outputs
                else "-"
            )
            r2_conf = (
                f"{result.round2_outputs[name].confidence:.2f}"
                if name in result.round2_outputs
                else "-"
            )
            table.add_row(name, f"{weight:.2f}", r1_conf, r2_conf)
        console.print(table)

        console.print(f"\n[bold]System confidence:[/bold] {result.confidence:.2f}")
        console.print(
            f"[bold]Primary specialist:[/bold] {result.coordinator_output.primary_specialist}"
        )
        console.print(
            f"[bold]Agreement:[/bold] {result.coordinator_output.specialist_agreement:.2f}"
        )
        console.print(
            f"\n[dim]Coordinator reasoning:[/dim] {result.coordinator_output.reasoning}\n"
        )

        # Log interaction
        log = ExperienceLog(cfg["experience_log"]["path"])
        record = InteractionRecord(
            input=query,
            round1_outputs=result.round1_outputs,
            round2_outputs=result.round2_outputs,
            coordinator_output=result.coordinator_output,
            attribution=result.attribution,
            primary_specialist=result.coordinator_output.primary_specialist,
            log_priority=result.coordinator_output.log_priority,
        )
        log.append(record)
        console.print(
            f"[dim]Logged to {cfg['experience_log']['path']} (total: {log.count()})[/dim]"
        )

    asyncio.run(_run())


@app.command()
def compete(
    benchmark_path: str = typer.Argument(..., help="Path to benchmark JSONL file"),
    rounds: int = typer.Option(
        50, "--rounds", "-n", help="Number of competitive rounds"
    ),
    config_path: str = typer.Option(
        "config/default.yaml", "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Run a competitive session between two agent instances."""
    load_dotenv()

    import asyncio

    from rich.console import Console
    from rich.table import Table

    from src.config import load_all_specialist_configs, load_config
    from src.eval.benchmarks.jsonl_benchmark import JsonlBenchmark
    from src.eval.scorer import Scorer
    from src.inference.backends.ollama import OllamaBackend
    from src.inference.coordinator import Coordinator
    from src.inference.orchestrator import Orchestrator
    from src.inference.specialist import Specialist
    from src.memory.experience_log import ExperienceLog

    console = Console()

    def _build_orchestrator(cfg: dict, backend: OllamaBackend) -> Orchestrator:
        specialist_names = cfg["specialists"]["enabled"]
        specialist_configs = load_all_specialist_configs(specialist_names)
        default_model = cfg["specialists"]["model_id"]
        specialists = {
            name: Specialist(config, backend, default_model)
            for name, config in specialist_configs.items()
        }
        coordinator = Coordinator(
            model=cfg["coordinator"]["model_id"],
            backend=backend,
            temperature=cfg["coordinator"]["temperature"],
            max_tokens=cfg["coordinator"]["max_tokens"],
        )
        return Orchestrator(
            specialists=specialists,
            coordinator=coordinator,
            enable_revision=cfg["inference"]["enable_revision_pass"],
        )

    async def _run() -> None:
        from src.training.competitive import CompetitiveTrainer

        cfg = load_config(config_path)
        ollama_cfg = cfg.get("ollama", {})
        backend = OllamaBackend(
            base_url=ollama_cfg.get("base_url", "http://localhost:11434")
        )
        agent_a = _build_orchestrator(cfg, backend)
        agent_b = _build_orchestrator(cfg, backend)

        benchmark = JsonlBenchmark(path=benchmark_path)
        scorer = Scorer(backend=backend)
        log = ExperienceLog(cfg["experience_log"]["path"])

        trainer = CompetitiveTrainer(
            agent_a=agent_a,
            agent_b=agent_b,
            benchmark=benchmark,
            scorer=scorer,
            experience_log=log,
        )

        console.print("\n[bold]Competitive Session[/bold]")
        console.print(f"[dim]Benchmark: {benchmark_path}[/dim]")
        console.print(f"[dim]Rounds: {rounds}[/dim]\n")

        with console.status("[bold green]Running competitive rounds..."):
            results = await trainer.run_session(num_rounds=rounds)

        summary = trainer.session_summary(results)

        table = Table(title="Competitive Session Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        for key, val in summary.items():
            table.add_row(key.replace("_", " ").title(), str(val))
        console.print(table)
        console.print(f"\n[dim]Interactions logged: {log.count()}[/dim]")

    asyncio.run(_run())


@app.command()
def sleep(
    config_path: str = typer.Option(
        "config/default.yaml", "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Run a sleep cycle: curate interactions and build training datasets."""
    load_dotenv()

    from rich.console import Console
    from rich.table import Table

    from src.config import load_config
    from src.eval.metrics import MetricsTracker
    from src.memory.experience_log import ExperienceLog
    from src.training.curator import Curator
    from src.training.dataset_builder import DatasetBuilder
    from src.training.sleep_cycle import SleepCycle

    console = Console()
    cfg = load_config(config_path)
    sc_cfg = cfg.get("sleep_cycle", {})
    cur_cfg = sc_cfg.get("curation", {})

    log = ExperienceLog(cfg["experience_log"]["path"])
    curator = Curator(
        max_items=cur_cfg.get("max_items", 200),
        vindication_weight=cur_cfg.get("vindication_weight", 3),
        disagreement_weight=cur_cfg.get("disagreement_weight", 2),
        loss_weight=cur_cfg.get("loss_weight", 3),
    )
    builder = DatasetBuilder()
    metrics = MetricsTracker()

    cycle = SleepCycle(
        experience_log=log,
        curator=curator,
        dataset_builder=builder,
        metrics_tracker=metrics,
    )

    console.print("\n[bold]Sleep Cycle[/bold]\n")

    with console.status("[bold green]Curating and building datasets..."):
        report = cycle.run()

    table = Table(title="Sleep Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Cycle", str(report.sleep_cycle_num))
    table.add_row("Items curated", str(report.items_curated))
    table.add_row("Training examples", str(report.training_examples_generated))
    table.add_row("Vindication cases", str(report.vindication_cases_found))
    table.add_row("Routing accuracy", f"{report.routing_accuracy_before:.2f}")
    table.add_row("Status", report.status)
    console.print(table)

    if report.checkpoints_saved:
        console.print("\n[dim]Datasets/checkpoints saved:[/dim]")
        for p in report.checkpoints_saved:
            console.print(f"  {p}")
    console.print()


@app.command()
def dashboard(
    config_path: str = typer.Option(
        "config/default.yaml", "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Show metrics dashboard from logged interactions."""
    load_dotenv()

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from src.config import load_config
    from src.eval.metrics import MetricsTracker
    from src.memory.experience_log import ExperienceLog

    console = Console()
    cfg = load_config(config_path)

    log = ExperienceLog(cfg["experience_log"]["path"])
    tracker = MetricsTracker()

    records = log.read_all()
    if not records:
        console.print("[yellow]No interactions logged yet.[/yellow]")
        raise typer.Exit()

    metrics = tracker.compute_all(records)

    console.print(Panel("[bold]CogArch Metrics Dashboard[/bold]", border_style="blue"))

    table = Table(title="System Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Total interactions", str(metrics["total_interactions"]))
    table.add_row("Routing accuracy", f"{metrics['routing_accuracy']:.2f}")
    table.add_row("Vindication rate", f"{metrics['vindication_rate']:.2f}")
    table.add_row("Calibration (ECE)", f"{metrics['coordinator_calibration_ece']:.3f}")
    console.print(table)

    cq = metrics["consensus_quality"]
    table2 = Table(title="Consensus Quality")
    table2.add_column("Metric", style="cyan")
    table2.add_column("Value", justify="right")
    table2.add_row("Agreement rate", f"{cq['agreement_rate']:.2f}")
    table2.add_row("Revision rate", f"{cq['revision_rate']:.2f}")
    table2.add_row("Improvement rate", f"{cq['improvement_rate']:.2f}")
    console.print(table2)


@app.command()
def bench(
    benchmark_path: str = typer.Argument(..., help="Path to benchmark JSONL file"),
    metric: str = typer.Option(
        "fuzzy_match",
        "--metric",
        "-m",
        help="Scoring metric: exact_match, fuzzy_match, or llm_judge",
    ),
    config_path: str = typer.Option(
        "config/default.yaml", "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Run a benchmark evaluation suite against the full pipeline."""
    load_dotenv()

    import asyncio

    from rich.console import Console
    from rich.table import Table

    from src.config import load_all_specialist_configs, load_config
    from src.eval.benchmarks.jsonl_benchmark import JsonlBenchmark
    from src.eval.scorer import Scorer
    from src.inference.backends.ollama import OllamaBackend
    from src.inference.coordinator import Coordinator
    from src.inference.orchestrator import Orchestrator
    from src.inference.specialist import Specialist

    console = Console()

    async def _run() -> None:
        cfg = load_config(config_path)
        ollama_cfg = cfg.get("ollama", {})
        backend = OllamaBackend(
            base_url=ollama_cfg.get("base_url", "http://localhost:11434")
        )

        specialist_names = cfg["specialists"]["enabled"]
        specialist_configs = load_all_specialist_configs(specialist_names)
        default_model = cfg["specialists"]["model_id"]
        specialists = {
            name: Specialist(config, backend, default_model)
            for name, config in specialist_configs.items()
        }
        coordinator = Coordinator(
            model=cfg["coordinator"]["model_id"],
            backend=backend,
            temperature=cfg["coordinator"]["temperature"],
            max_tokens=cfg["coordinator"]["max_tokens"],
        )
        orchestrator = Orchestrator(
            specialists=specialists,
            coordinator=coordinator,
            enable_revision=cfg["inference"]["enable_revision_pass"],
        )

        benchmark = JsonlBenchmark(path=benchmark_path, metric=metric)
        scorer = Scorer(backend=backend)
        items = await benchmark.load()

        if not items:
            console.print("[yellow]No benchmark items found.[/yellow]")
            raise typer.Exit()

        console.print(f"\n[bold]Benchmark: {benchmark.name}[/bold]")
        console.print(f"[dim]{len(items)} items · metric: {metric}[/dim]\n")

        results: list[dict] = []
        correct = 0
        total_score = 0.0

        for i, item in enumerate(items):
            with console.status(f"[bold green]Running item {i + 1}/{len(items)}..."):
                result = await orchestrator.run(item.question)

            if metric == "llm_judge":
                score = await scorer.llm_as_judge(
                    item.question, result.answer, item.expected_answer
                )
            elif metric == "fuzzy_match":
                score = scorer.fuzzy_match(result.answer, item.expected_answer)
            else:
                score = scorer.exact_match(result.answer, item.expected_answer)

            total_score += score
            if score >= 0.5:
                correct += 1

            results.append(
                {
                    "question": item.question[:60],
                    "expected": item.expected_answer[:30],
                    "predicted": result.answer[:30],
                    "score": score,
                    "primary": result.coordinator_output.primary_specialist,
                    "category": item.category,
                }
            )

        # Results table
        table = Table(title="Benchmark Results")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Question", style="white", max_width=50)
        table.add_column("Expected", style="green")
        table.add_column("Predicted", style="yellow")
        table.add_column("Score", justify="right")
        table.add_column("Primary", style="cyan")

        for i, r in enumerate(results):
            score_style = "green" if r["score"] >= 0.5 else "red"
            table.add_row(
                str(i + 1),
                r["question"],
                r["expected"],
                r["predicted"],
                f"[{score_style}]{r['score']:.2f}[/{score_style}]",
                r["primary"],
            )
        console.print(table)

        # Summary
        avg_score = total_score / len(items)
        console.print(f"\n[bold]Accuracy:[/bold] {correct}/{len(items)}")
        console.print(f"[bold]Average score:[/bold] {avg_score:.3f}")

        # Per-category breakdown if categories exist
        categories = {r["category"] for r in results if r["category"]}
        if categories:
            cat_table = Table(title="Per-Category Breakdown")
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Items", justify="right")
            cat_table.add_column("Avg Score", justify="right")
            for cat in sorted(categories):
                cat_items = [r for r in results if r["category"] == cat]
                cat_avg = sum(r["score"] for r in cat_items) / len(cat_items)
                cat_table.add_row(cat, str(len(cat_items)), f"{cat_avg:.3f}")
            console.print(cat_table)

    asyncio.run(_run())


@app.command()
def experiment(
    benchmark: str = typer.Argument(
        ...,
        help="Benchmark name (gsm8k, mmlu, truthfulqa) or path to JSONL file",
    ),
    cycles: int = typer.Option(5, "--cycles", "-n", help="Number of training cycles"),
    test_ratio: float = typer.Option(
        0.2, "--test-ratio", help="Fraction of data held out for testing"
    ),
    metric: str = typer.Option(
        "exact_match",
        "--metric",
        "-m",
        help="Scoring metric: exact_match, fuzzy_match, llm_judge",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducible splits"),
    config_path: str = typer.Option(
        "config/default.yaml", "--config", "-c", help="Path to config file"
    ),
    output_dir: str = typer.Option(
        "data/experiments", "--output", "-o", help="Directory for experiment results"
    ),
) -> None:
    """Run a full self-improvement experiment: baseline → train cycles → final eval."""
    load_dotenv()

    import asyncio

    from rich.console import Console

    from src.config import load_all_specialist_configs, load_config
    from src.eval.experiment import ExperimentRunner
    from src.eval.scorer import Scorer
    from src.inference.backends.ollama import OllamaBackend
    from src.inference.coordinator import Coordinator
    from src.inference.orchestrator import Orchestrator
    from src.inference.specialist import Specialist
    from src.models.experiment import ExperimentConfig
    from src.training.finetuner import SpecialistFinetuner
    from src.training.model_registry import ModelRegistry

    console = Console()

    async def _run() -> None:
        cfg = load_config(config_path)
        ollama_cfg = cfg.get("ollama", {})
        backend = OllamaBackend(
            base_url=ollama_cfg.get("base_url", "http://localhost:11434")
        )

        # Build specialists
        specialist_names = cfg["specialists"]["enabled"]
        specialist_configs = load_all_specialist_configs(specialist_names)
        default_model = cfg["specialists"]["model_id"]
        specialists = {
            name: Specialist(config, backend, default_model)
            for name, config in specialist_configs.items()
        }

        # Build coordinator
        coordinator = Coordinator(
            model=cfg["coordinator"]["model_id"],
            backend=backend,
            temperature=cfg["coordinator"]["temperature"],
            max_tokens=cfg["coordinator"]["max_tokens"],
        )

        # Build orchestrator
        orchestrator = Orchestrator(
            specialists=specialists,
            coordinator=coordinator,
            enable_revision=cfg["inference"]["enable_revision_pass"],
        )

        scorer = Scorer(backend=backend)

        ft_cfg = cfg.get("finetuning", {})
        finetuner: SpecialistFinetuner | None = None
        registry: ModelRegistry | None = None
        if ft_cfg.get("enabled", False):
            finetuner = SpecialistFinetuner(
                base_model=ft_cfg.get("base_model", "unsloth/llama-3-8b-bnb-4bit"),
                lora_rank=ft_cfg.get("lora_rank", 16),
                lora_alpha=ft_cfg.get("lora_alpha", 16),
                epochs=ft_cfg.get("epochs", 3),
                batch_size=ft_cfg.get("batch_size", 4),
                grad_accumulation=ft_cfg.get("grad_accumulation", 4),
                max_seq_length=ft_cfg.get("max_seq_length", 2048),
                min_examples=ft_cfg.get("min_examples", 10),
                output_dir=ft_cfg.get("output_dir", "models"),
                console=console,
            )
            registry = ModelRegistry(
                registry_path=ft_cfg.get("registry_path", "models/registry.json")
            )

        exp_config = ExperimentConfig(
            benchmark_name=benchmark,
            num_cycles=cycles,
            test_ratio=test_ratio,
            metric=metric,
            seed=seed,
        )

        runner = ExperimentRunner(
            config=exp_config,
            orchestrator=orchestrator,
            specialists=specialists,
            scorer=scorer,
            experience_log_path=cfg["experience_log"]["path"],
            output_dir=output_dir,
            console=console,
            finetuner=finetuner,
            registry=registry,
        )

        await runner.run()

    asyncio.run(_run())


if __name__ == "__main__":
    app()
