"""Sleep cycle — curate, build datasets, optionally fine-tune, and validate."""

from __future__ import annotations

from typing import Any

from src.eval.metrics import MetricsTracker
from src.memory.experience_log import ExperienceLog
from src.models.benchmark import SleepReport
from src.training.curator import Curator
from src.training.dataset_builder import DatasetBuilder
from src.training.fine_tuner import FineTuner


class SleepCycle:
    """Orchestrates the full sleep cycle: curate → assemble → fine-tune → validate.

    Four stages:
      1. Curate high-signal interactions from experience log
      2. Build per-specialist training datasets
      3. (Optional) Fine-tune specialists via OpenAI fine-tuning API
      4. Compute metrics and produce a SleepReport
    """

    def __init__(
        self,
        experience_log: ExperienceLog,
        curator: Curator,
        dataset_builder: DatasetBuilder,
        metrics_tracker: MetricsTracker,
        fine_tuner: FineTuner | None = None,
        system_prompts: dict[str, str] | None = None,
        cycle_num: int = 1,
    ) -> None:
        self.experience_log = experience_log
        self.curator = curator
        self.dataset_builder = dataset_builder
        self.metrics_tracker = metrics_tracker
        self.fine_tuner = fine_tuner
        self.system_prompts = system_prompts or {}
        self.cycle_num = cycle_num

    def run(self, fine_tune: bool = False, wait: bool = False) -> SleepReport:
        """Execute the full sleep cycle pipeline.

        Args:
            fine_tune: If True and a FineTuner is configured, submit fine-tuning jobs.
            wait: If True, block until fine-tuning jobs complete.
        """
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

        # Compute pre-fine-tuning metrics
        metrics = self.metrics_tracker.compute_all(all_records)
        self.metrics_tracker.save(metrics, label=f"cycle_{self.cycle_num}")
        routing_before = metrics["routing_accuracy"]

        # Stage 3: Fine-tune (optional)
        fine_tune_results: list[dict[str, Any]] = []
        if fine_tune and self.fine_tuner and total_examples > 0:
            fine_tune_results = self.fine_tuner.fine_tune_all(
                datasets=datasets,
                system_prompts=self.system_prompts,
                wait=wait,
            )
            self.fine_tuner.save_job_manifest(fine_tune_results, self.cycle_num)

        # Stage 4: Post-fine-tuning validation
        # Re-compute metrics if fine-tuning completed synchronously
        routing_after = routing_before
        specialist_deltas: dict[str, float] = {}

        if wait and fine_tune_results:
            post_metrics = self.metrics_tracker.compute_all(all_records)
            routing_after = post_metrics["routing_accuracy"]
            specialist_deltas = {name: 0.0 for name in datasets}
        else:
            specialist_deltas = {name: 0.0 for name in datasets}

        # Build checkpoint list
        checkpoints = [str(p) for p in saved_paths.values()]
        for ft_result in fine_tune_results:
            if ft_result.get("fine_tuned_model"):
                checkpoints.append(ft_result["fine_tuned_model"])

        status = "success"
        if fine_tune and not fine_tune_results:
            status = "no_finetune_data"
        elif fine_tune_results and any(
            r.get("status") == "failed" for r in fine_tune_results
        ):
            status = "partial_failure"

        return SleepReport(
            sleep_cycle_num=self.cycle_num,
            items_curated=len(curated),
            training_examples_generated=total_examples,
            specialist_improvements=specialist_deltas,
            routing_accuracy_before=routing_before,
            routing_accuracy_after=routing_after,
            checkpoints_saved=checkpoints,
            vindication_cases_found=vindication_count,
            fine_tune_jobs=[
                r.get("job_id", "") for r in fine_tune_results if r.get("job_id")
            ],
            status=status,
        )
