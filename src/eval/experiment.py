"""Experiment runner — orchestrates the full self-improvement experiment.

Flow:
  1. Load benchmark (GSM8K / MMLU / TruthfulQA or local JSONL)
  2. Split 80/20 train/test
  3. Partition train into N per-cycle chunks (unique questions each cycle)
  4. Baseline: evaluate on held-out test set
  5. For each cycle:
     a. Run competitive rounds on that cycle's training chunk
     b. Sleep cycle: curate → build datasets → compute metrics
     c. Evaluate on held-out test set again
  6. Final comparison: baseline vs. final score
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from src.eval.benchmarks.base import Benchmark
from src.eval.benchmarks.gsm8k import GSM8KBenchmark
from src.eval.benchmarks.jsonl_benchmark import JsonlBenchmark
from src.eval.benchmarks.mmlu import MMLUBenchmark
from src.eval.benchmarks.truthfulqa import TruthfulQABenchmark
from src.eval.scorer import Scorer
from src.eval.splitter import BenchmarkSplitter
from src.models.benchmark import BenchmarkItem
from src.models.experiment import CycleResult, ExperimentConfig, ExperimentReport


def load_benchmark(name: str, limit: int | None = None) -> Benchmark:
    """Factory: create a benchmark loader by name."""
    loaders: dict[str, type] = {
        "gsm8k": GSM8KBenchmark,
        "mmlu": MMLUBenchmark,
        "truthfulqa": TruthfulQABenchmark,
    }
    normalized = name.lower().replace("-", "").replace("_", "")
    if normalized in loaders:
        bench: Benchmark = loaders[normalized](limit=limit)  # type: ignore[call-arg]
        return bench
    # Fallback: treat as a local JSONL file path
    return JsonlBenchmark(path=name)


class ExperimentRunner:
    """Runs the full self-improvement experiment loop.

    Requires an already-configured set of CogArch components:
    - orchestrator (Orchestrator) for inference
    - specialists dict (name → Specialist) for model swapping
    - scorer (Scorer)
    - experience_log_path for logging competitive interactions
    - config (ExperimentConfig) for experiment parameters
    """

    def __init__(
        self,
        config: ExperimentConfig,
        orchestrator: Any,  # Orchestrator, avoid circular
        specialists: dict[str, Any],  # name → Specialist
        scorer: Scorer,
        experience_log_path: str = "data/experience_log/",
        output_dir: str | Path = "data/experiments",
        console: Console | None = None,
    ) -> None:
        self.config = config
        self.orchestrator = orchestrator
        self.specialists = specialists
        self.scorer = scorer
        self.experience_log_path = experience_log_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.console = console or Console()
        self._model_history: list[dict[str, str]] = []

    async def run(self) -> ExperimentReport:
        """Execute the full experiment pipeline."""
        benchmark = load_benchmark(self.config.benchmark_name)
        self.console.print(
            f"\n[bold]Loading benchmark:[/bold] {self.config.benchmark_name}"
        )
        all_items = await benchmark.load()
        self.console.print(f"[dim]Loaded {len(all_items)} items[/dim]")

        # Split train/test
        splitter = BenchmarkSplitter(
            items=all_items,
            test_ratio=self.config.test_ratio,
            num_cycles=self.config.num_cycles,
            seed=self.config.seed,
        )
        summary = splitter.summary()
        self.console.print(
            f"[dim]Test: {len(splitter.test_items)} · "
            f"Train: {splitter.total_train_items} · "
            f"Cycles: {splitter.num_cycles} · "
            f"Per-cycle: {summary['items_per_cycle']}[/dim]\n"
        )

        report = ExperimentReport(
            config=self.config,
            total_items=len(splitter.test_items) + splitter.total_train_items,
            train_items=splitter.total_train_items,
            test_items=len(splitter.test_items),
        )

        # Step 1: Baseline evaluation
        self.console.print("[bold yellow]── Baseline Evaluation ──[/bold yellow]")
        baseline = await self._evaluate(splitter.test_items, benchmark)
        report.baseline_score = baseline["avg_score"]
        report.baseline_correct = baseline["correct"]
        report.baseline_per_category = baseline["per_category"]
        self.console.print(
            f"Baseline: {baseline['correct']}/{len(splitter.test_items)} "
            f"({baseline['avg_score']:.3f})\n"
        )

        # Step 2: Cycle loop
        for cycle_idx in range(splitter.num_cycles):
            cycle_num = cycle_idx + 1
            train_chunk = splitter.get_cycle_items(cycle_idx)
            self.console.print(
                f"[bold cyan]── Cycle {cycle_num}/{splitter.num_cycles} "
                f"({len(train_chunk)} training items) ──[/bold cyan]"
            )

            cycle_result = await self._run_cycle(
                cycle_num=cycle_num,
                train_items=train_chunk,
                test_items=splitter.test_items,
                benchmark=benchmark,
            )
            report.cycles.append(cycle_result)

            self.console.print(
                f"  Test score after cycle {cycle_num}: "
                f"{cycle_result.test_correct}/{cycle_result.test_total} "
                f"({cycle_result.test_score:.3f})\n"
            )

        # Step 3: Final evaluation (should match last cycle's test score)
        self.console.print("[bold yellow]── Final Evaluation ──[/bold yellow]")
        final = await self._evaluate(splitter.test_items, benchmark)
        report.final_score = final["avg_score"]
        report.final_correct = final["correct"]
        report.final_per_category = final["per_category"]
        report.improvement = report.final_score - report.baseline_score
        if report.baseline_score > 0:
            report.improvement_pct = (report.improvement / report.baseline_score) * 100
        else:
            report.improvement_pct = 0.0

        # Save and display
        self._save_report(report)
        self._display_report(report)
        return report

    async def _evaluate(
        self,
        items: list[BenchmarkItem],
        benchmark: Benchmark,
    ) -> dict[str, Any]:
        """Evaluate the orchestrator on a set of items. Returns metrics dict."""
        correct = 0
        total_score = 0.0
        category_scores: dict[str, list[float]] = {}

        for i, item in enumerate(items):
            result = await self.orchestrator.run(item.question)
            score = benchmark.score(result.answer, item.expected_answer)
            total_score += score
            if score >= 0.5:
                correct += 1

            cat = item.category or "general"
            category_scores.setdefault(cat, []).append(score)

            if (i + 1) % 20 == 0 or i == len(items) - 1:
                self.console.print(
                    f"  [{i + 1}/{len(items)}] " f"running score: {correct}/{i + 1}"
                )

        avg_score = total_score / len(items) if items else 0.0
        per_cat = {
            cat: sum(scores) / len(scores) for cat, scores in category_scores.items()
        }

        return {
            "avg_score": avg_score,
            "correct": correct,
            "total": len(items),
            "per_category": per_cat,
        }

    async def _run_cycle(
        self,
        cycle_num: int,
        train_items: list[BenchmarkItem],
        test_items: list[BenchmarkItem],
        benchmark: Benchmark,
    ) -> CycleResult:
        """Run one experiment cycle: compete → sleep → test."""
        from src.eval.benchmarks.jsonl_benchmark import JsonlBenchmark
        from src.eval.metrics import MetricsTracker
        from src.memory.experience_log import ExperienceLog
        from src.training.competitive import CompetitiveTrainer
        from src.training.curator import Curator
        from src.training.dataset_builder import DatasetBuilder
        from src.training.sleep_cycle import SleepCycle

        cycle_result = CycleResult(
            cycle=cycle_num,
            train_items_used=len(train_items),
        )

        # ── Competitive rounds ──
        cycle_jsonl = self.output_dir / f"cycle_{cycle_num}_train.jsonl"
        with open(cycle_jsonl, "w") as f:
            for item in train_items:
                f.write(item.model_dump_json() + "\n")

        cycle_benchmark = JsonlBenchmark(path=cycle_jsonl)
        experience_dir = Path(self.experience_log_path)
        experience_dir.mkdir(parents=True, exist_ok=True)
        exp_log = ExperienceLog(str(experience_dir))

        trainer = CompetitiveTrainer(
            agent_a=self.orchestrator,
            agent_b=self.orchestrator,
            benchmark=cycle_benchmark,
            scorer=self.scorer,
            experience_log=exp_log,
        )

        self.console.print(f"  Competing on {len(train_items)} items...")
        results = await trainer.run_session(num_rounds=len(train_items))
        summary = trainer.session_summary(results)
        cycle_result.competitive_rounds = summary["rounds"]
        cycle_result.agent_a_wins = summary["agent_a_wins"]
        cycle_result.agent_b_wins = summary["agent_b_wins"]
        cycle_result.ties = summary["ties"]

        # ── Sleep cycle (curate + build datasets + metrics) ──
        curator = Curator()
        builder = DatasetBuilder()
        metrics_tracker = MetricsTracker(metrics_dir=str(self.output_dir / "metrics"))

        sleep_cycle = SleepCycle(
            experience_log=exp_log,
            curator=curator,
            dataset_builder=builder,
            metrics_tracker=metrics_tracker,
            cycle_num=cycle_num,
        )

        self.console.print("  Running sleep cycle...")
        sleep_cycle.run()

        # ── Evaluate on held-out test set ──
        self.console.print(f"  Evaluating on {len(test_items)} test items...")
        test_metrics = await self._evaluate(test_items, benchmark)
        cycle_result.test_score = test_metrics["avg_score"]
        cycle_result.test_correct = test_metrics["correct"]
        cycle_result.test_total = test_metrics["total"]
        cycle_result.per_category_scores = test_metrics["per_category"]

        return cycle_result

    def _save_report(self, report: ExperimentReport) -> None:
        """Save the full experiment report as JSON."""
        path = self.output_dir / f"report_{report.config.benchmark_name}.json"
        with open(path, "w") as f:
            f.write(report.model_dump_json(indent=2))
        self.console.print(f"\n[dim]Report saved to {path}[/dim]")

    def _display_report(self, report: ExperimentReport) -> None:
        """Print a summary table."""
        self.console.print("\n[bold green]── Experiment Summary ──[/bold green]")

        table = Table(title=f"Self-Improvement: {report.config.benchmark_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Total items", str(report.total_items))
        table.add_row("Test items", str(report.test_items))
        table.add_row("Cycles", str(len(report.cycles)))
        table.add_row("Baseline score", f"{report.baseline_score:.3f}")
        table.add_row("Final score", f"{report.final_score:.3f}")
        table.add_row(
            "Improvement",
            f"{report.improvement:+.3f} ({report.improvement_pct:+.1f}%)",
        )
        self.console.print(table)

        # Per-cycle progress
        if report.cycles:
            cycle_table = Table(title="Per-Cycle Test Scores")
            cycle_table.add_column("Cycle", justify="right")
            cycle_table.add_column("Train Items", justify="right")
            cycle_table.add_column("Score", justify="right")
            cycle_table.add_column("Correct", justify="right")

            for c in report.cycles:
                score_style = "green" if c.test_score > report.baseline_score else "red"
                cycle_table.add_row(
                    str(c.cycle),
                    str(c.train_items_used),
                    f"[{score_style}]{c.test_score:.3f}[/{score_style}]",
                    f"{c.test_correct}/{c.test_total}",
                )
            self.console.print(cycle_table)
