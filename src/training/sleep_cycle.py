"""Sleep cycle — curate, build datasets, compute metrics, and validate."""

from __future__ import annotations

from src.eval.metrics import MetricsTracker
from src.memory.experience_log import ExperienceLog
from src.models.benchmark import SleepReport
from src.training.curator import Curator
from src.training.dataset_builder import DatasetBuilder


class SleepCycle:
    """Orchestrates the full sleep cycle: curate → assemble → validate.

    Three stages:
      1. Curate high-signal interactions from experience log
      2. Build per-specialist training datasets
      3. Compute metrics and produce a SleepReport
    """

    def __init__(
        self,
        experience_log: ExperienceLog,
        curator: Curator,
        dataset_builder: DatasetBuilder,
        metrics_tracker: MetricsTracker,
        system_prompts: dict[str, str] | None = None,
        cycle_num: int = 1,
    ) -> None:
        self.experience_log = experience_log
        self.curator = curator
        self.dataset_builder = dataset_builder
        self.metrics_tracker = metrics_tracker
        self.system_prompts = system_prompts or {}
        self.cycle_num = cycle_num

    def run(self) -> SleepReport:
        """Execute the full sleep cycle pipeline."""
        # Stage 1: Curate high-signal interactions
        all_records = self.experience_log.read_all()
        curated = self.curator.select(all_records)

        # Stage 2: Build per-specialist training datasets
        datasets = self.dataset_builder.build(curated)
        saved_paths = self.dataset_builder.save(datasets)

        total_examples = sum(len(exs) for exs in datasets.values())

        # Count vindication cases
        vindication_count = sum(
            1 for r in curated if r.vindication and any(r.vindication.values())
        )

        # Stage 3: Compute metrics
        metrics = self.metrics_tracker.compute_all(all_records)
        self.metrics_tracker.save(metrics, label=f"cycle_{self.cycle_num}")
        routing_accuracy = metrics["routing_accuracy"]

        checkpoints = [str(p) for p in saved_paths.values()]

        return SleepReport(
            sleep_cycle_num=self.cycle_num,
            items_curated=len(curated),
            training_examples_generated=total_examples,
            specialist_improvements={name: 0.0 for name in datasets},
            routing_accuracy_before=routing_accuracy,
            routing_accuracy_after=routing_accuracy,
            checkpoints_saved=checkpoints,
            vindication_cases_found=vindication_count,
            status="success",
        )
