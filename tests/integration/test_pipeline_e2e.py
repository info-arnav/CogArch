"""Integration tests — full pipeline E2E with mock LLM backend."""

import json
import tempfile
from pathlib import Path

import pytest

from src.eval.benchmarks.jsonl_benchmark import JsonlBenchmark
from src.eval.metrics import MetricsTracker
from src.eval.scorer import Scorer
from src.inference.coordinator import Coordinator
from src.inference.orchestrator import Orchestrator
from src.inference.specialist import Specialist
from src.memory.experience_log import ExperienceLog
from src.models.interaction import InteractionRecord
from src.models.specialist import SpecialistConfig
from src.training.competitive import CompetitiveTrainer
from src.training.curator import Curator
from src.training.dataset_builder import DatasetBuilder
from src.training.sleep_cycle import SleepCycle
from tests.conftest import CorrectAnswerBackend, MockBackend


def _make_specialist(name: str, backend: MockBackend) -> Specialist:
    config = SpecialistConfig(
        name=name, system_prompt=f"You are the {name} specialist.", temperature=0.4
    )
    return Specialist(config, backend, default_model="llama3:8b")


def _make_coordinator(backend: MockBackend) -> Coordinator:
    return Coordinator(
        backend=backend,
        model="llama3:8b",
        temperature=0.3,
        max_tokens=1024,
    )


def _make_orchestrator(backend: MockBackend) -> Orchestrator:
    specialists = {
        name: _make_specialist(name, backend)
        for name in ["logical", "creative", "skeptical"]
    }
    coordinator = _make_coordinator(backend)
    return Orchestrator(
        specialists=specialists, coordinator=coordinator, enable_revision=True
    )


class TestFullPipeline:
    """Test the complete inference pipeline with mock backend."""

    @pytest.mark.asyncio
    async def test_basic_inference(self) -> None:
        backend = MockBackend()
        orchestrator = _make_orchestrator(backend)

        result = await orchestrator.run("What is the capital of France?")

        assert result.answer != ""
        assert result.confidence > 0
        assert len(result.round1_outputs) == 3
        assert len(result.round2_outputs) == 3
        assert result.coordinator_output.primary_specialist != ""

    @pytest.mark.asyncio
    async def test_inference_without_revision(self) -> None:
        backend = MockBackend()
        specialists = {
            name: _make_specialist(name, backend) for name in ["logical", "creative"]
        }
        coordinator = _make_coordinator(backend)
        orchestrator = Orchestrator(
            specialists=specialists, coordinator=coordinator, enable_revision=False
        )

        result = await orchestrator.run("What is 2+2?")

        assert len(result.round1_outputs) == 2
        # Without revision, round2 == round1
        assert result.round1_outputs == result.round2_outputs
        # R1 + coordinator = 3 calls total
        assert backend.call_count == 3

    @pytest.mark.asyncio
    async def test_inference_call_count(self) -> None:
        backend = MockBackend()
        orchestrator = _make_orchestrator(backend)

        await orchestrator.run("Test query")

        # 3 specialists R1 + 3 specialists R2 + 1 coordinator = 7
        assert backend.call_count == 7

    @pytest.mark.asyncio
    async def test_attribution_sums_to_one(self) -> None:
        backend = MockBackend()
        orchestrator = _make_orchestrator(backend)

        result = await orchestrator.run("Test")

        total = sum(result.attribution.values())
        assert abs(total - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_experience_logging(self) -> None:
        backend = MockBackend()
        orchestrator = _make_orchestrator(backend)

        result = await orchestrator.run("What causes rain?")

        with tempfile.TemporaryDirectory() as tmpdir:
            log = ExperienceLog(tmpdir)
            record = InteractionRecord(
                input="What causes rain?",
                round1_outputs=result.round1_outputs,
                round2_outputs=result.round2_outputs,
                coordinator_output=result.coordinator_output,
                attribution=result.attribution,
                primary_specialist=result.coordinator_output.primary_specialist,
                log_priority=result.coordinator_output.log_priority,
            )
            log.append(record)
            assert log.count() == 1

            records = log.read_all()
            assert len(records) == 1
            assert records[0].input == "What causes rain?"


class TestCompetitivePipeline:
    """Test the competitive training pipeline end-to-end."""

    @pytest.mark.asyncio
    async def test_competitive_round(self, tmp_path: Path) -> None:
        # Create a small benchmark
        bench_path = tmp_path / "test_bench.jsonl"
        bench_path.write_text(
            '{"question": "What is 2+2?", "expected_answer": "4", "category": "math"}\n'
            '{"question": "Capital of France?", "expected_answer": "Paris", "category": "geo"}\n'
        )

        backend = MockBackend()
        agent_a = _make_orchestrator(backend)
        agent_b = _make_orchestrator(backend)
        benchmark = JsonlBenchmark(path=bench_path)
        scorer = Scorer()
        log = ExperienceLog(str(tmp_path / "log"))

        trainer = CompetitiveTrainer(
            agent_a=agent_a,
            agent_b=agent_b,
            benchmark=benchmark,
            scorer=scorer,
            experience_log=log,
        )

        results = await trainer.run_session(num_rounds=2)

        assert len(results) == 2
        assert all(r.winner in ("a", "b", "tie") for r in results)
        # 2 rounds × 2 agents × 7 calls each = 28
        assert backend.call_count == 28
        # Each round logs 2 interactions (one per agent)
        assert log.count() == 4

    @pytest.mark.asyncio
    async def test_competitive_summary(self, tmp_path: Path) -> None:
        bench_path = tmp_path / "bench.jsonl"
        bench_path.write_text(
            '{"question": "q1", "expected_answer": "a1"}\n'
            '{"question": "q2", "expected_answer": "a2"}\n'
            '{"question": "q3", "expected_answer": "a3"}\n'
        )

        backend = MockBackend()
        agent_a = _make_orchestrator(backend)
        agent_b = _make_orchestrator(backend)
        benchmark = JsonlBenchmark(path=bench_path)
        scorer = Scorer()
        log = ExperienceLog(str(tmp_path / "log"))

        trainer = CompetitiveTrainer(
            agent_a=agent_a,
            agent_b=agent_b,
            benchmark=benchmark,
            scorer=scorer,
            experience_log=log,
        )

        results = await trainer.run_session(num_rounds=3)
        summary = trainer.session_summary(results)

        assert summary["rounds"] == 3
        assert summary["agent_a_wins"] + summary["agent_b_wins"] + summary["ties"] == 3


class TestSleepCyclePipeline:
    """Test the sleep cycle with curated data."""

    def test_sleep_cycle_from_logged_interactions(self, tmp_path: Path) -> None:
        log = ExperienceLog(str(tmp_path / "log"))

        # Create mock interaction records with scores
        for i in range(5):
            from src.models.coordinator import CoordinatorOutput, SelfState
            from src.models.specialist import SpecialistOutput

            r1 = {
                "logical": SpecialistOutput(
                    specialist_name="logical",
                    answer=f"answer_{i}",
                    reasoning_trace="reasoning",
                    confidence=0.8,
                ),
                "creative": SpecialistOutput(
                    specialist_name="creative",
                    answer=f"alt_answer_{i}",
                    reasoning_trace="reasoning",
                    confidence=0.6,
                ),
            }
            coord = CoordinatorOutput(
                final_answer=f"answer_{i}",
                attribution={"logical": 0.7, "creative": 0.3},
                primary_specialist="logical",
                confidence=0.8,
                specialist_agreement=0.4 if i % 2 else 0.9,
                reasoning="test",
                should_log=True,
                log_priority="high" if i < 2 else "medium",
                updated_self_state=SelfState(),
            )
            record = InteractionRecord(
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
            log.append(record)

        curator = Curator(max_items=10)
        builder = DatasetBuilder(output_dir=tmp_path / "training")
        metrics = MetricsTracker(metrics_dir=str(tmp_path / "metrics"))

        cycle = SleepCycle(
            experience_log=log,
            curator=curator,
            dataset_builder=builder,
            metrics_tracker=metrics,
        )

        report = cycle.run()

        assert report.items_curated == 5
        assert report.training_examples_generated > 0
        assert report.vindication_cases_found > 0
        assert report.status == "success"

    def test_sleep_cycle_empty_log(self, tmp_path: Path) -> None:
        log = ExperienceLog(str(tmp_path / "log"))
        curator = Curator()
        builder = DatasetBuilder(output_dir=tmp_path / "training")
        metrics = MetricsTracker(metrics_dir=str(tmp_path / "metrics"))

        cycle = SleepCycle(
            experience_log=log,
            curator=curator,
            dataset_builder=builder,
            metrics_tracker=metrics,
        )

        report = cycle.run()
        assert report.items_curated == 0
        assert report.training_examples_generated == 0


class TestBenchmarkEvaluation:
    """Test benchmark loading and scoring end-to-end."""

    @pytest.mark.asyncio
    async def test_jsonl_benchmark_load(self, tmp_path: Path) -> None:
        bench_path = tmp_path / "bench.jsonl"
        items = [
            {"question": f"q{i}", "expected_answer": f"a{i}", "category": "test"}
            for i in range(10)
        ]
        bench_path.write_text("\n".join(json.dumps(item) for item in items))

        benchmark = JsonlBenchmark(path=bench_path, metric="exact_match")
        loaded = await benchmark.load()
        assert len(loaded) == 10
        assert loaded[0].question == "q0"

    @pytest.mark.asyncio
    async def test_benchmark_with_correct_answers(self, tmp_path: Path) -> None:
        """Test that a backend returning correct answers gets high scores."""
        bench_path = tmp_path / "bench.jsonl"
        bench_path.write_text(
            '{"question": "What is the capital of France?", "expected_answer": "Paris"}\n'
        )

        answer_map = {"What is the capital of France?": "Paris"}
        backend = CorrectAnswerBackend(answer_map=answer_map)

        specialists = {
            name: _make_specialist(name, backend) for name in ["logical", "creative"]
        }
        coordinator = Coordinator(backend=backend, model="llama3:8b", temperature=0.3)
        orchestrator = Orchestrator(
            specialists=specialists, coordinator=coordinator, enable_revision=False
        )

        result = await orchestrator.run("What is the capital of France?")
        scorer = Scorer()
        score = scorer.fuzzy_match(result.answer, "Paris")

        assert score > 0.5


class TestMetricsFromCompetitive:
    """Test that metrics computation works on competitive data."""

    def test_metrics_from_scored_records(self, tmp_path: Path) -> None:
        from src.models.coordinator import CoordinatorOutput, SelfState
        from src.models.specialist import SpecialistOutput

        records: list[InteractionRecord] = []
        for i in range(10):
            r1 = {
                "logical": SpecialistOutput(
                    specialist_name="logical",
                    answer="a",
                    reasoning_trace="r",
                    confidence=0.8,
                ),
                "creative": SpecialistOutput(
                    specialist_name="creative",
                    answer="b",
                    reasoning_trace="r",
                    confidence=0.6,
                ),
            }
            coord = CoordinatorOutput(
                final_answer="a",
                attribution={"logical": 0.7, "creative": 0.3},
                primary_specialist="logical",
                confidence=0.7 + (i * 0.02),
                specialist_agreement=0.6,
                reasoning="test",
                should_log=True,
                log_priority="medium",
                updated_self_state=SelfState(),
            )
            record = InteractionRecord(
                input=f"q{i}",
                round1_outputs=r1,
                round2_outputs=r1,
                coordinator_output=coord,
                attribution={"logical": 0.7, "creative": 0.3},
                primary_specialist="logical",
                outcome_score=1.0 if i % 2 == 0 else 0.0,
                vindication={"creative": i % 3 == 0},
                log_priority="medium",
            )
            records.append(record)

        tracker = MetricsTracker(metrics_dir=str(tmp_path / "metrics"))
        metrics = tracker.compute_all(records)

        assert "routing_accuracy" in metrics
        assert "vindication_rate" in metrics
        assert "coordinator_calibration_ece" in metrics
        assert "consensus_quality" in metrics
        assert metrics["total_interactions"] == 10
        assert 0.0 <= metrics["routing_accuracy"] <= 1.0
        assert 0.0 <= metrics["vindication_rate"] <= 1.0

        # Save and verify
        path = tracker.save(metrics, label="test")
        assert path.exists()
        with open(path) as f:
            saved = json.load(f)
        assert saved["total_interactions"] == 10
