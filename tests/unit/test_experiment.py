"""Tests for the ExperimentRunner — end-to-end experiment pipeline with mocks."""

from pathlib import Path

import pytest

from src.eval.benchmarks.jsonl_benchmark import JsonlBenchmark
from src.eval.experiment import load_benchmark
from src.models.benchmark import BenchmarkItem
from src.models.experiment import CycleResult, ExperimentConfig, ExperimentReport


def _write_benchmark_jsonl(path: Path, n: int = 30) -> list[BenchmarkItem]:
    """Write a small benchmark JSONL and return the items."""
    items = []
    for i in range(n):
        item = BenchmarkItem(
            question=f"What is {i} + {i}?",
            expected_answer=str(i + i),
            category=f"cat_{i % 3}",
        )
        items.append(item)
    with open(path, "w") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
    return items


class TestLoadBenchmark:
    def test_load_gsm8k_returns_correct_type(self) -> None:
        from src.eval.benchmarks.gsm8k import GSM8KBenchmark

        bench = load_benchmark("gsm8k")
        assert isinstance(bench, GSM8KBenchmark)

    def test_load_mmlu_returns_correct_type(self) -> None:
        from src.eval.benchmarks.mmlu import MMLUBenchmark

        bench = load_benchmark("mmlu")
        assert isinstance(bench, MMLUBenchmark)

    def test_load_truthfulqa_returns_correct_type(self) -> None:
        from src.eval.benchmarks.truthfulqa import TruthfulQABenchmark

        bench = load_benchmark("truthfulqa")
        assert isinstance(bench, TruthfulQABenchmark)

    def test_load_jsonl_fallback(self) -> None:
        bench = load_benchmark("data/benchmarks/sample.jsonl")
        assert isinstance(bench, JsonlBenchmark)

    def test_case_insensitive(self) -> None:
        from src.eval.benchmarks.gsm8k import GSM8KBenchmark

        bench = load_benchmark("GSM8K")
        assert isinstance(bench, GSM8KBenchmark)


class TestExperimentConfig:
    def test_defaults(self) -> None:
        cfg = ExperimentConfig(benchmark_name="gsm8k")
        assert cfg.num_cycles == 5
        assert cfg.test_ratio == 0.2
        assert cfg.seed == 42

    def test_custom_values(self) -> None:
        cfg = ExperimentConfig(
            benchmark_name="mmlu",
            num_cycles=3,
            test_ratio=0.3,
            seed=123,
        )
        assert cfg.num_cycles == 3
        assert cfg.test_ratio == 0.3


class TestExperimentReport:
    def test_improvement_calculation(self) -> None:
        report = ExperimentReport(
            config=ExperimentConfig(benchmark_name="test"),
            baseline_score=0.5,
            final_score=0.7,
            improvement=0.2,
            improvement_pct=40.0,
        )
        assert report.improvement == pytest.approx(0.2)
        assert report.improvement_pct == pytest.approx(40.0)

    def test_empty_cycles(self) -> None:
        report = ExperimentReport(
            config=ExperimentConfig(benchmark_name="test"),
        )
        assert report.cycles == []
        assert report.baseline_score == 0.0


class TestCycleResult:
    def test_defaults(self) -> None:
        result = CycleResult(cycle=1)
        assert result.train_items_used == 0

    def test_with_data(self) -> None:
        result = CycleResult(
            cycle=2,
            train_items_used=50,
            test_score=0.75,
            test_correct=15,
            test_total=20,
        )
        assert result.test_score == 0.75
