"""Unit tests for the Curator."""

from src.models.coordinator import CoordinatorOutput, SelfState
from src.models.interaction import InteractionRecord
from src.models.specialist import SpecialistOutput
from src.training.curator import Curator


def _make_output(name: str, confidence: float = 0.8) -> SpecialistOutput:
    return SpecialistOutput(
        specialist_name=name,
        answer="answer",
        reasoning_trace="reasoning",
        confidence=confidence,
    )


def _make_coord(
    agreement: float = 0.8,
    log_priority: str = "medium",
) -> CoordinatorOutput:
    return CoordinatorOutput(
        final_answer="answer",
        attribution={"logical": 1.0},
        primary_specialist="logical",
        confidence=0.9,
        specialist_agreement=agreement,
        reasoning="reason",
        updated_self_state=SelfState(),
        should_log=True,
        log_priority=log_priority,
    )


def _make_record(
    outcome: float | None = None,
    vindication: dict[str, bool] | None = None,
    agreement: float = 0.8,
    priority: str = "medium",
) -> InteractionRecord:
    out = _make_output("logical")
    return InteractionRecord(
        input="test",
        round1_outputs={"logical": out},
        round2_outputs={"logical": out},
        coordinator_output=_make_coord(agreement, priority),
        attribution={"logical": 1.0},
        primary_specialist="logical",
        outcome_score=outcome,
        vindication=vindication,
        log_priority=priority,
    )


class TestCurator:
    def test_empty_records(self) -> None:
        curator = Curator()
        assert curator.select([]) == []

    def test_loss_adds_score(self) -> None:
        curator = Curator()
        record = _make_record(outcome=0.0)
        assert curator.priority_score(record) >= 3

    def test_vindication_adds_score(self) -> None:
        curator = Curator()
        record = _make_record(vindication={"creative": True})
        assert curator.priority_score(record) >= 3

    def test_disagreement_adds_score(self) -> None:
        curator = Curator()
        record = _make_record(agreement=0.2)
        assert curator.priority_score(record) >= 2

    def test_high_priority_adds_score(self) -> None:
        curator = Curator()
        record = _make_record(priority="high")
        assert curator.priority_score(record) >= 2

    def test_clear_signal_adds_score(self) -> None:
        curator = Curator()
        record = _make_record(outcome=0.95)
        assert curator.priority_score(record) >= 1

    def test_max_items_limit(self) -> None:
        curator = Curator(max_items=2)
        records = [_make_record() for _ in range(5)]
        selected = curator.select(records)
        assert len(selected) == 2

    def test_highest_priority_selected_first(self) -> None:
        curator = Curator(max_items=1)
        low = _make_record()
        high = _make_record(
            outcome=0.0, vindication={"creative": True}, priority="high"
        )
        selected = curator.select([low, high])
        assert len(selected) == 1
        # The high-priority record should be selected
        assert selected[0].outcome_score == 0.0
