"""Data models for CogArch - specialist outputs, coordinator state, interaction logging."""

from src.models.benchmark import (
    BenchmarkItem,
    CompetitiveResult,
    SleepReport,
    TrainingExample,
)
from src.models.coordinator import CoordinatorOutput, SelfState
from src.models.interaction import InteractionRecord
from src.models.specialist import SpecialistConfig, SpecialistOutput

__all__ = [
    "BenchmarkItem",
    "CompetitiveResult",
    "CoordinatorOutput",
    "InteractionRecord",
    "SelfState",
    "SleepReport",
    "SpecialistConfig",
    "SpecialistOutput",
    "TrainingExample",
]
