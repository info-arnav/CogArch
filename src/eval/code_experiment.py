"""Code self-improvement experiment.

Flow per cycle:
  1. Competitive rounds on MBPP training chunk (agents never see test assertions)
  2. Build DPO pairs from outcomes (even from both-fail rounds)
  3. Fine-tune each specialist via DPO
  4. Evaluate on held-out HumanEval (Pass@1, no feedback, one attempt)
  5. Track improvement; rollback if regression > threshold
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.eval.benchmarks import humaneval, mbpp
from src.execution.code_runner import run_attempt
from src.inference.code_orchestrator import CodeOrchestrator
from src.models.code import (
    CodeExperimentReport,
    CodeProblem,
    CycleMetrics,
)
from src.training.code_competitive import CodeCompetitiveTrainer
from src.training.code_dataset_builder import build_dpo_pairs, save
from src.training.finetuner import SpecialistFinetuner
from src.training.model_registry import ModelRegistry


class CodeExperimentRunner:
    """Runs the full code self-improvement loop."""

    def __init__(
        self,
        agent_a: CodeOrchestrator,
        agent_b: CodeOrchestrator,
        finetuner: SpecialistFinetuner | None,
        registry: ModelRegistry | None,
        cycles: int = 10,
        rounds_per_cycle: int = 30,
        output_dir: str | Path = "data/experiments/code",
        console: Console | None = None,
    ) -> None:
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.finetuner = finetuner
        self.registry = registry
        self.cycles = cycles
        self.rounds_per_cycle = rounds_per_cycle
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.console = console or Console()
        self._best_score = 0.0

    async def run(self) -> CodeExperimentReport:
        model_name = next(iter(self.agent_a.specialists.values())).model
        report = CodeExperimentReport(
            model=model_name,
            train_benchmark="mbpp",
            eval_benchmark="humaneval",
        )

        # Load benchmarks
        self.console.print("\n[bold]Loading benchmarks...[/bold]")
        train_problems = mbpp.load(split="train")
        eval_problems = humaneval.load()
        self.console.print(
            f"[dim]MBPP train: {len(train_problems)} · HumanEval eval: {len(eval_problems)}[/dim]"
        )

        # Baseline
        self.console.print("\n[bold yellow]── Baseline HumanEval ──[/bold yellow]")
        baseline = await self._eval_humaneval(eval_problems)
        report.baseline_pass_at_1 = baseline
        self._best_score = baseline
        self.console.print(f"Baseline Pass@1: {baseline:.3f}\n")

        # Partition training problems into cycle chunks
        chunk_size = max(1, len(train_problems) // self.cycles)
        chunks = [
            train_problems[i : i + chunk_size]
            for i in range(0, len(train_problems), chunk_size)
        ][: self.cycles]

        # Main loop
        trainer = CodeCompetitiveTrainer(agent_a=self.agent_a, agent_b=self.agent_b)

        for cycle_idx, chunk in enumerate(chunks):
            cycle_num = cycle_idx + 1
            self.console.print(
                f"[bold cyan]── Cycle {cycle_num}/{self.cycles} "
                f"({len(chunk)} problems) ──[/bold cyan]"
            )

            metrics = await self._run_cycle(
                cycle_num=cycle_num,
                trainer=trainer,
                train_problems=chunk[: self.rounds_per_cycle],
                eval_problems=eval_problems,
            )
            report.cycles.append(metrics)
            report.total_rounds += metrics.rounds_competed
            report.total_dpo_pairs += metrics.dpo_pairs_generated

            self.console.print(
                f"  HumanEval Pass@1: {metrics.humaneval_pass_at_1:.3f} "
                f"(Δ {metrics.humaneval_pass_at_1 - baseline:+.3f})\n"
            )

        report.final_pass_at_1 = (
            report.cycles[-1].humaneval_pass_at_1 if report.cycles else baseline
        )
        report.improvement = report.final_pass_at_1 - report.baseline_pass_at_1

        self._save_report(report)
        self._display_report(report)
        return report

    async def _run_cycle(
        self,
        cycle_num: int,
        trainer: CodeCompetitiveTrainer,
        train_problems: list[CodeProblem],
        eval_problems: list[CodeProblem],
    ) -> CycleMetrics:
        metrics = CycleMetrics(cycle=cycle_num)

        # Compete
        self.console.print(f"  Competing on {len(train_problems)} problems...")
        results = await trainer.run_session(train_problems)
        summary = trainer.session_summary(results)
        metrics.rounds_competed = summary["rounds"]
        metrics.both_failed_rounds = summary["both_failed"]
        self.console.print(
            f"  [dim]A wins: {summary['agent_a_wins']} · "
            f"B wins: {summary['agent_b_wins']} · "
            f"Ties: {summary['ties']} · "
            f"Both failed: {summary['both_failed']}[/dim]"
        )

        # Build DPO pairs
        datasets = build_dpo_pairs(results)
        metrics.dpo_pairs_generated = sum(len(v) for v in datasets.values())
        self.console.print(
            f"  [dim]{metrics.dpo_pairs_generated} DPO pairs generated[/dim]"
        )

        dpo_paths = save(datasets, self.output_dir / "training")

        # Fine-tune each specialist
        for spec_name in self.agent_a.specialists:
            system_prompt = self.agent_a.specialists[spec_name].system_prompt
            dpo_path = dpo_paths.get(spec_name)
            new_model = None
            if self.finetuner and dpo_path and dpo_path.exists():
                new_model = self.finetuner.run_dpo(spec_name, dpo_path, system_prompt)
            if new_model and self.registry:
                self.registry.register(spec_name, new_model)
                # Update both agents to use the new model
                for agent in (self.agent_a, self.agent_b):
                    agent.specialists[spec_name].model = new_model
                self.console.print(f"  [green]{spec_name} → {new_model}[/green]")

        # Evaluate
        self.console.print(
            f"  Evaluating on {len(eval_problems)} HumanEval problems..."
        )
        pass_at_1 = await self._eval_humaneval(eval_problems)
        metrics.humaneval_pass_at_1 = pass_at_1

        # Rollback if regressed
        if pass_at_1 < self._best_score - 0.05 and self.registry is not None:
            self.console.print(
                f"  [yellow]Regression detected ({pass_at_1:.3f} < {self._best_score:.3f} - 0.05) "
                f"— rolling back[/yellow]"
            )
            for spec_name in self.agent_a.specialists:
                prev = self.registry.rollback(spec_name)
                for agent in (self.agent_a, self.agent_b):
                    agent.specialists[spec_name].model = prev
        else:
            self._best_score = max(self._best_score, pass_at_1)

        return metrics

    async def _eval_humaneval(self, problems: list[CodeProblem]) -> float:
        """Evaluate agents on HumanEval: Pass@1 (single attempt, no feedback)."""
        passed = 0
        for i, problem in enumerate(problems):
            # Single attempt, no error feedback — standard Pass@1 protocol
            result = await self.agent_a.specialists[
                next(iter(self.agent_a.specialists))
            ].backend.generate(
                messages=[
                    {
                        "role": "system",
                        "content": next(
                            iter(self.agent_a.specialists.values())
                        ).system_prompt,
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Problem:\n{problem.prompt}\n\n"
                            f"Implement the function named `{problem.entry_point}`."
                        ),
                    },
                ],
                model=next(iter(self.agent_a.specialists.values())).model,
                temperature=0.0,
                max_tokens=1024,
            )
            from src.inference.code_specialist import _extract_code

            code = _extract_code(result)
            attempt = run_attempt(code, problem.test_assertions, attempt_number=1)
            if attempt.passed_all:
                passed += 1

            if (i + 1) % 20 == 0:
                self.console.print(
                    f"  [{i + 1}/{len(problems)}] pass@1 so far: {passed}/{i + 1}"
                )

        return passed / len(problems) if problems else 0.0

    def _save_report(self, report: CodeExperimentReport) -> None:
        path = self.output_dir / "report.json"
        with open(path, "w") as f:
            f.write(report.model_dump_json(indent=2))
        self.console.print(f"\n[dim]Report saved → {path}[/dim]")

    def _display_report(self, report: CodeExperimentReport) -> None:
        self.console.print("\n[bold green]── Code Experiment Summary ──[/bold green]")
        table = Table(title=f"HumanEval Pass@1 — {report.model}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Baseline Pass@1", f"{report.baseline_pass_at_1:.3f}")
        table.add_row("Final Pass@1", f"{report.final_pass_at_1:.3f}")
        table.add_row(
            "Improvement",
            f"{report.improvement:+.3f} ({report.improvement * 100:+.1f}pp)",
        )
        table.add_row("Total DPO pairs", str(report.total_dpo_pairs))
        table.add_row("Total rounds", str(report.total_rounds))
        self.console.print(table)

        if report.cycles:
            ct = Table(title="Per-Cycle Pass@1")
            ct.add_column("Cycle", justify="right")
            ct.add_column("Pass@1", justify="right")
            ct.add_column("DPO Pairs", justify="right")
            ct.add_column("Both Failed", justify="right")
            for c in report.cycles:
                style = (
                    "green"
                    if c.humaneval_pass_at_1 > report.baseline_pass_at_1
                    else "red"
                )
                ct.add_row(
                    str(c.cycle),
                    f"[{style}]{c.humaneval_pass_at_1:.3f}[/{style}]",
                    str(c.dpo_pairs_generated),
                    str(c.both_failed_rounds),
                )
            self.console.print(ct)
