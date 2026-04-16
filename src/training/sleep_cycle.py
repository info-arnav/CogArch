"""Sleep cycle — curate, build datasets, fine-tune specialists, compute metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.eval.metrics import MetricsTracker
from src.memory.experience_log import ExperienceLog
from src.models.benchmark import SleepReport
from src.training.curator import Curator
from src.training.dataset_builder import DatasetBuilder

if TYPE_CHECKING:
    from src.training.finetuner import SpecialistFinetuner
    from src.training.model_registry import ModelRegistry


class SleepCycle:
    """Orchestrates the full sleep cycle: curate → build datasets → fine-tune → metrics.

    Four stages:
      1. Curate high-signal interactions from experience log
      2. Build per-specialist training datasets
      3. QLoRA fine-tune each specialist (if finetuner provided)
      4. Compute metrics and produce a SleepReport
    """

    def __init__(
        self,
        experience_log: ExperienceLog,
        curator: Curator,
        dataset_builder: DatasetBuilder,
        metrics_tracker: MetricsTracker,
        system_prompts: dict[str, str] | None = None,
        cycle_num: int = 1,
        finetuner: SpecialistFinetuner | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.experience_log = experience_log
        self.curator = curator
        self.dataset_builder = dataset_builder
        self.metrics_tracker = metrics_tracker
        self.system_prompts = system_prompts or {}
        self.cycle_num = cycle_num
        self.finetuner = finetuner
        self.registry = registry

    def run(self) -> SleepReport:
        """Execute the full sleep cycle pipeline."""
        all_records = self.experience_log.read_all()
        curated = self.curator.select(all_records)

        datasets = self.dataset_builder.build(curated)
        saved_paths = self.dataset_builder.save(datasets)

        if self.finetuner and self.registry:
            dpo_datasets = self.dataset_builder.build_dpo(curated)
            dpo_paths = self.dataset_builder.save_dpo(dpo_datasets)

            for name, sft_path in saved_paths.items():
                system_prompt = self.system_prompts.get(name, "")
                new_model: str | None = None

                # Prefer DPO when enough preference pairs are available
                dpo_path = dpo_paths.get(name)
                if dpo_path and dpo_path.exists():
                    new_model = self.finetuner.run_dpo(name, dpo_path, system_prompt)

                # Fall back to SFT if DPO was skipped or failed
                if new_model is None:
                    new_model = self.finetuner.run(name, sft_path, system_prompt)

                if new_model:
                    exs = datasets.get(name, [])
                    win_rate = 0.0
                    if exs:
                        wins = sum(1 for e in exs if e.training_signal == "win")
                        vindicated = sum(
                            1 for e in exs if e.training_signal == "vindicated"
                        )
                        win_rate = round((wins + 0.5 * vindicated) / len(exs), 3)
                    self.registry.register(name, new_model, score=win_rate)

        total_examples = sum(len(exs) for exs in datasets.values())
        vindication_count = sum(
            1 for r in curated if r.vindication and any(r.vindication.values())
        )

        metrics_all = self.metrics_tracker.compute_all(all_records)
        self.metrics_tracker.save(metrics_all, label=f"cycle_{self.cycle_num}")
        routing_accuracy_before = metrics_all["routing_accuracy"]

        if curated:
            routing_accuracy_after = self.metrics_tracker.compute_all(curated)[
                "routing_accuracy"
            ]
        else:
            routing_accuracy_after = routing_accuracy_before

        specialist_improvements: dict[str, float] = {}
        for name, exs in datasets.items():
            if exs:
                wins = sum(1 for e in exs if e.training_signal == "win")
                vindicated = sum(1 for e in exs if e.training_signal == "vindicated")
                specialist_improvements[name] = round(
                    (wins + 0.5 * vindicated) / len(exs), 3
                )
            else:
                specialist_improvements[name] = 0.0

        checkpoints = [str(p) for p in saved_paths.values()]

        return SleepReport(
            sleep_cycle_num=self.cycle_num,
            items_curated=len(curated),
            training_examples_generated=total_examples,
            specialist_improvements=specialist_improvements,
            routing_accuracy_before=routing_accuracy_before,
            routing_accuracy_after=routing_accuracy_after,
            checkpoints_saved=checkpoints,
            vindication_cases_found=vindication_count,
            status="success",
        )
