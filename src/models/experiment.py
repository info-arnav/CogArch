"""Experiment data models — track self-improvement experiment state and results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CycleResult(BaseModel):
    """Metrics from a single experiment cycle."""

    cycle: int
    train_items_used: int = 0
    competitive_rounds: int = 0
    agent_a_wins: int = 0
    agent_b_wins: int = 0
    ties: int = 0
    fine_tune_jobs: list[str] = Field(default_factory=list)
    fine_tuned_models: dict[str, str] = Field(
        default_factory=dict,
        description="Specialist name → fine-tuned model ID",
    )
    test_score: float = 0.0
    test_correct: int = 0
    test_total: int = 0
    per_category_scores: dict[str, float] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Configuration for a self-improvement experiment."""

    benchmark_name: str
    num_cycles: int = 5
    per_cycle_items: int = 100
    test_ratio: float = 0.2
    metric: str = "exact_match"
    fine_tune: bool = True
    wait_for_fine_tune: bool = True
    base_model: str = "gpt-4o-mini-2024-07-18"
    seed: int = 42


class ExperimentReport(BaseModel):
    """Full experiment report with baseline, per-cycle, and final results."""

    config: ExperimentConfig
    total_items: int = 0
    train_items: int = 0
    test_items: int = 0
    baseline_score: float = 0.0
    baseline_correct: int = 0
    baseline_per_category: dict[str, float] = Field(default_factory=dict)
    cycles: list[CycleResult] = Field(default_factory=list)
    final_score: float = 0.0
    final_correct: int = 0
    final_per_category: dict[str, float] = Field(default_factory=dict)
    improvement: float = 0.0
    improvement_pct: float = 0.0
