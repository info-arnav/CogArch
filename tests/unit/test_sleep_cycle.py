"""Unit tests for the enhanced sleep cycle with fine-tuning integration."""

import json
from pathlib import Path

from src.eval.metrics import MetricsTracker
from src.memory.experience_log import ExperienceLog
from src.models.coordinator import CoordinatorOutput, SelfState
from src.models.interaction import InteractionRecord
from src.models.specialist import SpecialistOutput
from src.training.curator import Curator
from src.training.dataset_builder import DatasetBuilder
from src.training.sleep_cycle import SleepCycle


def _make_records(count: int = 5) -> list[InteractionRecord]:
    """Create mock interaction records for testing."""
    records = []
    for i in range(count):
        r1 = {
            "logical": SpecialistOutput(
                specialist_name="logical",
                answer=f"answer_{i}",
                reasoning_trace="reasoning",
                confidence=0.8,
            ),
            "creative": SpecialistOutput(
                specialist_name="creative",
                answer=f"alt_{i}",
                reasoning_trace="reasoning",
                confidence=0.6,
            ),
        }
        coord = CoordinatorOutput(
            final_answer=f"answer_{i}",
            attribution={"logical": 0.7, "creative": 0.3},
            primary_specialist="logical",
            confidence=0.8,
            specialist_agreement=0.3 if i % 2 else 0.9,
            reasoning="test",
            should_log=True,
            log_priority="high" if i < 2 else "medium",
            updated_self_state=SelfState(),
        )
        records.append(
            InteractionRecord(
                input=f"question {i}",
                round1_outputs=r1,
                round2_outputs=r1,
                coordinator_output=coord,
                attribution={"logical": 0.7, "creative": 0.3},
                primary_specialist="logical",
                outcome_score=1.0 if i % 2 == 0 else 0.0,
                vindication={"creative": i % 2 == 1},
                log_priority="high" if i < 2 else "medium",
            )
        )
    return records


class TestSleepCycleNoFineTune:
    """Sleep cycle without fine-tuning (dataset assembly only)."""

    def test_run_produces_report(self, tmp_path: Path) -> None:
        log = ExperienceLog(str(tmp_path / "log"))
        for r in _make_records(5):
            log.append(r)

        cycle = SleepCycle(
            experience_log=log,
            curator=Curator(max_items=100),
            dataset_builder=DatasetBuilder(output_dir=tmp_path / "training"),
            metrics_tracker=MetricsTracker(metrics_dir=str(tmp_path / "metrics")),
        )

        report = cycle.run()
        assert report.status == "success"
        assert report.items_curated == 5
        assert report.training_examples_generated > 0
        assert report.fine_tune_jobs == []

    def test_run_empty_log(self, tmp_path: Path) -> None:
        log = ExperienceLog(str(tmp_path / "log"))
        cycle = SleepCycle(
            experience_log=log,
            curator=Curator(),
            dataset_builder=DatasetBuilder(output_dir=tmp_path / "training"),
            metrics_tracker=MetricsTracker(metrics_dir=str(tmp_path / "metrics")),
        )

        report = cycle.run()
        assert report.items_curated == 0
        assert report.training_examples_generated == 0

    def test_fine_tune_flag_without_tuner(self, tmp_path: Path) -> None:
        """fine_tune=True with no FineTuner configured should not crash."""
        log = ExperienceLog(str(tmp_path / "log"))
        for r in _make_records(3):
            log.append(r)

        cycle = SleepCycle(
            experience_log=log,
            curator=Curator(),
            dataset_builder=DatasetBuilder(output_dir=tmp_path / "training"),
            metrics_tracker=MetricsTracker(metrics_dir=str(tmp_path / "metrics")),
            fine_tuner=None,
        )

        # Should not raise even with fine_tune=True
        report = cycle.run(fine_tune=True)
        assert report.status == "no_finetune_data"

    def test_metrics_saved_per_cycle(self, tmp_path: Path) -> None:
        log = ExperienceLog(str(tmp_path / "log"))
        for r in _make_records(5):
            log.append(r)

        metrics_dir = tmp_path / "metrics"
        cycle = SleepCycle(
            experience_log=log,
            curator=Curator(),
            dataset_builder=DatasetBuilder(output_dir=tmp_path / "training"),
            metrics_tracker=MetricsTracker(metrics_dir=str(metrics_dir)),
            cycle_num=3,
        )

        cycle.run()
        assert (metrics_dir / "cycle_3.json").exists()
        data = json.loads((metrics_dir / "cycle_3.json").read_text())
        assert "routing_accuracy" in data

    def test_datasets_written_to_disk(self, tmp_path: Path) -> None:
        log = ExperienceLog(str(tmp_path / "log"))
        for r in _make_records(5):
            log.append(r)

        training_dir = tmp_path / "training"
        cycle = SleepCycle(
            experience_log=log,
            curator=Curator(),
            dataset_builder=DatasetBuilder(output_dir=training_dir),
            metrics_tracker=MetricsTracker(metrics_dir=str(tmp_path / "metrics")),
        )

        report = cycle.run()
        assert report.checkpoints_saved
        for path_str in report.checkpoints_saved:
            assert Path(path_str).exists()

    def test_vindication_counting(self, tmp_path: Path) -> None:
        log = ExperienceLog(str(tmp_path / "log"))
        for r in _make_records(10):
            log.append(r)

        cycle = SleepCycle(
            experience_log=log,
            curator=Curator(max_items=100),
            dataset_builder=DatasetBuilder(output_dir=tmp_path / "training"),
            metrics_tracker=MetricsTracker(metrics_dir=str(tmp_path / "metrics")),
        )

        report = cycle.run()
        assert report.vindication_cases_found > 0
