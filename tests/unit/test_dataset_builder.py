"""Unit tests for the DatasetBuilder."""

from src.models.coordinator import CoordinatorOutput, SelfState
from src.models.interaction import InteractionRecord
from src.models.specialist import SpecialistOutput
from src.training.dataset_builder import DatasetBuilder


def _make_output(
    name: str, answer: str = "answer", reasoning: str = "reasoning"
) -> SpecialistOutput:
    return SpecialistOutput(
        specialist_name=name,
        answer=answer,
        reasoning_trace=reasoning,
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

    def test_reasoning_trace_propagated_to_winner(self, tmp_path: object) -> None:
        outputs = {
            "logical": _make_output("logical", "42", reasoning="step by step math"),
            "creative": _make_output("creative", "forty-two"),
        }
        record = InteractionRecord(
            input="What is 6*7?",
            round1_outputs=outputs,
            round2_outputs=outputs,
            coordinator_output=CoordinatorOutput(
                final_answer="42",
                attribution={"logical": 0.7, "creative": 0.3},
                primary_specialist="logical",
                confidence=0.92,
                specialist_agreement=0.8,
                reasoning="reason",
                updated_self_state=SelfState(),
                should_log=True,
                log_priority="medium",
            ),
            attribution={"logical": 0.7, "creative": 0.3},
            primary_specialist="logical",
            outcome_score=1.0,
        )
        builder = DatasetBuilder(output_dir=str(tmp_path))
        datasets = builder.build([record])
        win_example = next(e for e in datasets["logical"] if e.training_signal == "win")
        assert win_example.reasoning_trace == "step by step math"
        assert win_example.coordinator_confidence == 0.92

    def test_loser_gets_winners_reasoning_trace(self, tmp_path: object) -> None:
        outputs = {
            "logical": _make_output("logical", "42", reasoning="winner reasoning"),
            "creative": _make_output("creative", "wrong", reasoning="loser reasoning"),
        }
        record = InteractionRecord(
            input="What is 6*7?",
            round1_outputs=outputs,
            round2_outputs=outputs,
            coordinator_output=CoordinatorOutput(
                final_answer="42",
                attribution={"logical": 0.7, "creative": 0.3},
                primary_specialist="logical",
                confidence=0.9,
                specialist_agreement=0.8,
                reasoning="reason",
                updated_self_state=SelfState(),
                should_log=True,
                log_priority="medium",
            ),
            attribution={"logical": 0.7, "creative": 0.3},
            primary_specialist="logical",
            outcome_score=1.0,
        )
        builder = DatasetBuilder(output_dir=str(tmp_path))
        datasets = builder.build([record])
        learn_example = next(
            e for e in datasets["creative"] if e.training_signal == "learn_from_winner"
        )
        # Loser should learn the winner's reasoning trace, not its own
        assert learn_example.reasoning_trace == "winner reasoning"


class TestDPODatasetBuilder:
    def _make_correct_record(self) -> InteractionRecord:
        outputs = {
            "logical": _make_output("logical", "42", reasoning="logical reasoning"),
            "creative": _make_output(
                "creative", "forty-two", reasoning="creative reasoning"
            ),
        }
        return InteractionRecord(
            input="What is 6*7?",
            round1_outputs=outputs,
            round2_outputs=outputs,
            coordinator_output=CoordinatorOutput(
                final_answer="42",
                attribution={"logical": 0.8, "creative": 0.2},
                primary_specialist="logical",
                confidence=0.9,
                specialist_agreement=0.8,
                reasoning="reason",
                updated_self_state=SelfState(),
                should_log=True,
                log_priority="medium",
            ),
            attribution={"logical": 0.8, "creative": 0.2},
            primary_specialist="logical",
            outcome_score=1.0,
        )

    def test_dpo_skips_incorrect_records(self, tmp_path: object) -> None:
        outputs = {
            "logical": _make_output("logical", "wrong"),
            "creative": _make_output("creative", "also wrong"),
        }
        record = InteractionRecord(
            input="Hard question",
            round1_outputs=outputs,
            round2_outputs=outputs,
            coordinator_output=CoordinatorOutput(
                final_answer="wrong",
                attribution={"logical": 0.5, "creative": 0.5},
                primary_specialist="logical",
                confidence=0.3,
                specialist_agreement=0.5,
                reasoning="reason",
                updated_self_state=SelfState(),
                should_log=True,
                log_priority="medium",
            ),
            attribution={"logical": 0.5, "creative": 0.5},
            primary_specialist="logical",
            outcome_score=0.0,
        )
        builder = DatasetBuilder(output_dir=str(tmp_path))
        dpo = builder.build_dpo([record])
        assert all(len(v) == 0 for v in dpo.values())

    def test_dpo_winner_gets_dpo_win_signal(self, tmp_path: object) -> None:
        builder = DatasetBuilder(output_dir=str(tmp_path))
        dpo = builder.build_dpo([self._make_correct_record()])
        logical_signals = [e.training_signal for e in dpo.get("logical", [])]
        assert "dpo_win" in logical_signals

    def test_dpo_loser_gets_dpo_learn_signal(self, tmp_path: object) -> None:
        builder = DatasetBuilder(output_dir=str(tmp_path))
        dpo = builder.build_dpo([self._make_correct_record()])
        creative_signals = [e.training_signal for e in dpo.get("creative", [])]
        assert "dpo_learn" in creative_signals

    def test_dpo_loser_chosen_is_winners_trace(self, tmp_path: object) -> None:
        builder = DatasetBuilder(output_dir=str(tmp_path))
        dpo = builder.build_dpo([self._make_correct_record()])
        learn_ex = next(e for e in dpo["creative"] if e.training_signal == "dpo_learn")
        assert "logical reasoning" in learn_ex.chosen
        assert "42" in learn_ex.chosen

    def test_dpo_loser_rejected_is_own_trace(self, tmp_path: object) -> None:
        builder = DatasetBuilder(output_dir=str(tmp_path))
        dpo = builder.build_dpo([self._make_correct_record()])
        learn_ex = next(e for e in dpo["creative"] if e.training_signal == "dpo_learn")
        assert "creative reasoning" in learn_ex.rejected

    def test_save_dpo_writes_jsonl(self, tmp_path: object) -> None:
        builder = DatasetBuilder(output_dir=str(tmp_path))
        dpo = builder.build_dpo([self._make_correct_record()])
        paths = builder.save_dpo(dpo)
        assert "logical" in paths
        assert paths["logical"].exists()
        lines = paths["logical"].read_text().strip().splitlines()
        assert len(lines) > 0
