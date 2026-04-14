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
    from src.inference.backends.openai import OpenAIBackend
    from src.inference.coordinator import Coordinator
    from src.inference.orchestrator import Orchestrator
    from src.inference.specialist import Specialist
    from src.memory.experience_log import ExperienceLog
    from src.models.interaction import InteractionRecord

    console = Console()

    async def _run() -> None:
        # Load config
        cfg = load_config(config_path)
        backend = OpenAIBackend()

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


if __name__ == "__main__":
    app()
