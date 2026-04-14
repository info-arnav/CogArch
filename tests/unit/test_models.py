"""Unit tests for Pydantic data models."""

import pytest

from src.models.benchmark import (
    BenchmarkItem,
    CompetitiveResult,
    SleepReport,
    TrainingExample,
)
from src.models.coordinator import CoordinatorOutput, SelfState
from src.models.interaction import InteractionRecord
from src.models.specialist import SpecialistConfig, SpecialistOutput


class TestSpecialistModels:
    def test_specialist_config_defaults(self) -> None:
        cfg = SpecialistConfig(name="test")
        assert cfg.name == "test"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 2048
        assert cfg.model_id is None

    def test_specialist_output_validation(self) -> None:
        out = SpecialistOutput(
            specialist_name="logical",
            answer="42",
            reasoning_trace="Because math",
            confidence=0.9,
        )
        assert out.confidence == 0.9
        assert out.endorsed == []
        assert out.challenged == []

    def test_specialist_output_confidence_bounds(self) -> None:
        with pytest.raises(Exception):
            SpecialistOutput(
                specialist_name="x",
                answer="a",
                reasoning_trace="b",
                confidence=1.5,
            )


class TestCoordinatorModels:
    def test_self_state_defaults(self) -> None:
        state = SelfState()
        assert state.uncertainty == 0.5
        assert state.turn_count == 0
        assert state.recent_routing_history == []

    def test_coordinator_output(self) -> None:
        out = CoordinatorOutput(
            final_answer="Paris",
            attribution={"logical": 0.6, "creative": 0.4},
            primary_specialist="logical",
            confidence=0.95,
            specialist_agreement=0.8,
            reasoning="Both agree",
            updated_self_state=SelfState(),
            should_log=True,
            log_priority="medium",
        )
        assert out.primary_specialist == "logical"
        assert out.confidence == 0.95


class TestBenchmarkModels:
    def test_benchmark_item(self) -> None:
        item = BenchmarkItem(
            question="What is 2+2?",
            expected_answer="4",
        )
        assert item.difficulty == "medium"

    def test_competitive_result(self) -> None:
        result = CompetitiveResult(
            test_item=BenchmarkItem(question="q", expected_answer="a"),
            agent_a_answer="a",
            agent_b_answer="b",
            agent_a_score=1.0,
            agent_b_score=0.0,
            winner="a",
        )
        assert result.winner == "a"

    def test_training_example(self) -> None:
        ex = TrainingExample(
            input="q",
            specialist_name="logical",
            target_output="a",
            training_signal="win",
        )
        assert ex.weight == 1.0

    def test_sleep_report_defaults(self) -> None:
        report = SleepReport(
            sleep_cycle_num=1,
            items_curated=10,
            training_examples_generated=20,
        )
        assert report.status == "success"
        assert report.vindication_cases_found == 0


class TestInteractionRecord:
    def test_record_creation(self) -> None:
        out = SpecialistOutput(
            specialist_name="logical",
            answer="42",
            reasoning_trace="math",
            confidence=0.9,
        )
        coord = CoordinatorOutput(
            final_answer="42",
            attribution={"logical": 1.0},
            primary_specialist="logical",
            confidence=0.95,
            specialist_agreement=1.0,
            reasoning="unanimous",
            updated_self_state=SelfState(),
            should_log=True,
            log_priority="medium",
        )
        record = InteractionRecord(
            input="What is 6*7?",
            round1_outputs={"logical": out},
            round2_outputs={"logical": out},
            coordinator_output=coord,
            attribution={"logical": 1.0},
            primary_specialist="logical",
        )
        assert record.outcome_score is None
        assert record.vindication is None
        assert record.timestamp  # auto-generated
