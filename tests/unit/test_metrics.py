"""Unit tests for MetricsTracker."""

from src.eval.metrics import MetricsTracker
from src.models.coordinator import CoordinatorOutput, SelfState
from src.models.interaction import InteractionRecord
from src.models.specialist import SpecialistOutput


def _make_output(
    name: str, answer: str = "a", confidence: float = 0.8
) -> SpecialistOutput:
    return SpecialistOutput(
        specialist_name=name,
        answer=answer,
        reasoning_trace="reasoning",
        confidence=confidence,
    )


def _make_record(
    outcome: float | None = None,
    vindication: dict[str, bool] | None = None,
    confidence: float = 0.9,
    agreement: float = 0.8,
    r1_answer: str = "a",
    r2_answer: str = "a",
    r2_confidence: float = 0.8,
) -> InteractionRecord:
    r1 = _make_output("logical", r1_answer, 0.8)
    r2 = _make_output("logical", r2_answer, r2_confidence)
    return InteractionRecord(
        input="test",
        round1_outputs={"logical": r1},
        round2_outputs={"logical": r2},
        coordinator_output=CoordinatorOutput(
            final_answer="a",
            attribution={"logical": 1.0},
            primary_specialist="logical",
            confidence=confidence,
            specialist_agreement=agreement,
            reasoning="reason",
            updated_self_state=SelfState(),
            should_log=True,
            log_priority="medium",
        ),
        attribution={"logical": 1.0},
        primary_specialist="logical",
        outcome_score=outcome,
        vindication=vindication,
    )


class TestMetricsTracker:
    def test_empty_records(self, tmp_path: object) -> None:
        tracker = MetricsTracker(metrics_dir=str(tmp_path))
        metrics = tracker.compute_all([])
        assert metrics["total_interactions"] == 0
        assert metrics["routing_accuracy"] == 0.0

    def test_routing_accuracy_all_correct(self, tmp_path: object) -> None:
        tracker = MetricsTracker(metrics_dir=str(tmp_path))
        records = [
            _make_record(outcome=1.0, vindication={"creative": False}),
            _make_record(outcome=0.8, vindication={"creative": False}),
        ]
        assert tracker.routing_accuracy(records) == 1.0

    def test_routing_accuracy_some_vindicated(self, tmp_path: object) -> None:
        tracker = MetricsTracker(metrics_dir=str(tmp_path))
        records = [
            _make_record(outcome=0.5, vindication={"creative": True}),
            _make_record(outcome=0.8, vindication={"creative": False}),
        ]
        assert tracker.routing_accuracy(records) == 0.5

    def test_vindication_rate(self, tmp_path: object) -> None:
        tracker = MetricsTracker(metrics_dir=str(tmp_path))
        records = [
            _make_record(vindication={"creative": True}),
            _make_record(vindication={"creative": False}),
        ]
        assert tracker.vindication_rate(records) == 0.5

    def test_consensus_quality_no_revision(self, tmp_path: object) -> None:
        tracker = MetricsTracker(metrics_dir=str(tmp_path))
        records = [_make_record(r1_answer="a", r2_answer="a")]
        cq = tracker.consensus_quality(records)
        assert cq["agreement_rate"] == 1.0
        assert cq["revision_rate"] == 0.0

    def test_consensus_quality_with_revision(self, tmp_path: object) -> None:
        tracker = MetricsTracker(metrics_dir=str(tmp_path))
        records = [_make_record(r1_answer="a", r2_answer="b", r2_confidence=0.9)]
        cq = tracker.consensus_quality(records)
        assert cq["revision_rate"] == 1.0
        assert cq["improvement_rate"] == 1.0  # confidence went up

    def test_coordinator_calibration_empty(self, tmp_path: object) -> None:
        tracker = MetricsTracker(metrics_dir=str(tmp_path))
        assert tracker.coordinator_calibration([]) == 1.0

    def test_save_metrics(self, tmp_path: object) -> None:
        tracker = MetricsTracker(metrics_dir=str(tmp_path))
        metrics = {"routing_accuracy": 0.8}
        import pathlib

        path = tracker.save(metrics, label="test")
        assert pathlib.Path(path).exists()
