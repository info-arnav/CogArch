"""Unit tests for ExperienceLog."""

import json

from src.memory.experience_log import ExperienceLog
from src.models.coordinator import CoordinatorOutput, SelfState
from src.models.interaction import InteractionRecord
from src.models.specialist import SpecialistOutput


def _make_record() -> InteractionRecord:
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
    return InteractionRecord(
        input="What is 6*7?",
        round1_outputs={"logical": out},
        round2_outputs={"logical": out},
        coordinator_output=coord,
        attribution={"logical": 1.0},
        primary_specialist="logical",
    )


class TestExperienceLog:
    def test_append_and_count(self, tmp_path: object) -> None:
        log = ExperienceLog(str(tmp_path))
        assert log.count() == 0
        log.append(_make_record())
        assert log.count() == 1
        log.append(_make_record())
        assert log.count() == 2

    def test_read_all(self, tmp_path: object) -> None:
        log = ExperienceLog(str(tmp_path))
        log.append(_make_record())
        log.append(_make_record())
        records = log.read_all()
        assert len(records) == 2
        assert records[0].input == "What is 6*7?"

    def test_read_empty(self, tmp_path: object) -> None:
        log = ExperienceLog(str(tmp_path))
        assert log.read_all() == []

    def test_jsonl_format(self, tmp_path: object) -> None:
        log = ExperienceLog(str(tmp_path))
        log.append(_make_record())
        with open(log.log_file) as f:
            line = f.readline()
        data = json.loads(line)
        assert "input" in data
        assert data["input"] == "What is 6*7?"
