"""Unit tests for the DatasetBuilder."""

from src.models.coordinator import CoordinatorOutput, SelfState
from src.models.interaction import InteractionRecord
from src.models.specialist import SpecialistOutput
from src.training.dataset_builder import DatasetBuilder


def _make_output(name: str, answer: str = "answer") -> SpecialistOutput:
    return SpecialistOutput(
        specialist_name=name,
        answer=answer,
        reasoning_trace="reasoning",
        confidence=0.8,
    )


def _make_record(
    primary: str = "logical",
    outcome: float = 1.0,
    vindication: dict[str, bool] | None = None,
) -> InteractionRecord:
    outputs = {
        "logical": _make_output("logical", "42"),
        "creative": _make_output("creative", "forty-two"),
    }
    return InteractionRecord(
        input="What is 6*7?",
        round1_outputs=outputs,
        round2_outputs=outputs,
        coordinator_output=CoordinatorOutput(
            final_answer="42",
            attribution={"logical": 0.7, "creative": 0.3},
            primary_specialist=primary,
            confidence=0.9,
            specialist_agreement=0.8,
            reasoning="reason",
            updated_self_state=SelfState(),
            should_log=True,
            log_priority="medium",
        ),
        attribution={"logical": 0.7, "creative": 0.3},
        primary_specialist=primary,
        outcome_score=outcome,
        vindication=vindication,
    )


class TestDatasetBuilder:
    def test_empty_input(self, tmp_path: object) -> None:
        builder = DatasetBuilder(output_dir=str(tmp_path))
        datasets = builder.build([])
        assert datasets == {}

    def test_winner_gets_win_signal(self, tmp_path: object) -> None:
        builder = DatasetBuilder(output_dir=str(tmp_path))
        record = _make_record(primary="logical", outcome=1.0)
        datasets = builder.build([record])
        logical_signals = [e.training_signal for e in datasets.get("logical", [])]
        assert "win" in logical_signals

    def test_loser_gets_learn_signal(self, tmp_path: object) -> None:
        builder = DatasetBuilder(output_dir=str(tmp_path))
        record = _make_record(primary="logical", outcome=1.0)
        datasets = builder.build([record])
        creative_signals = [e.training_signal for e in datasets.get("creative", [])]
        assert "learn_from_winner" in creative_signals

    def test_vindicated_gets_vindicated_signal(self, tmp_path: object) -> None:
        builder = DatasetBuilder(output_dir=str(tmp_path))
        record = _make_record(
            primary="logical",
            outcome=1.0,
            vindication={"creative": True},
        )
        datasets = builder.build([record])
        creative_signals = [e.training_signal for e in datasets.get("creative", [])]
        assert "vindicated" in creative_signals

    def test_weights_applied(self, tmp_path: object) -> None:
        builder = DatasetBuilder(
            output_dir=str(tmp_path),
            winner_weight=1.0,
            loser_weight=0.5,
            vindication_weight=2.0,
        )
        record = _make_record(
            primary="logical", outcome=1.0, vindication={"creative": True}
        )
        datasets = builder.build([record])
        for ex in datasets.get("logical", []):
            if ex.training_signal == "win":
                assert ex.weight == 1.0
        for ex in datasets.get("creative", []):
            if ex.training_signal == "vindicated":
                assert ex.weight == 2.0

    def test_summary(self, tmp_path: object) -> None:
        builder = DatasetBuilder(output_dir=str(tmp_path))
        record = _make_record(primary="logical", outcome=1.0)
        datasets = builder.build([record])
        summary = builder.summary(datasets)
        assert "logical" in summary
        assert summary["logical"]["total"] > 0
