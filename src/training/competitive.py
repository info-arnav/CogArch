"""Competitive trainer — two agent instances compete on benchmark items."""

import asyncio

from src.eval.benchmarks.base import Benchmark
from src.eval.scorer import Scorer
from src.inference.orchestrator import Orchestrator, OrchestratorResult
from src.memory.experience_log import ExperienceLog
from src.models.benchmark import BenchmarkItem, CompetitiveResult
from src.models.interaction import InteractionRecord


class CompetitiveTrainer:
    """Runs competitive sessions between two agent instances."""

    def __init__(
        self,
        agent_a: Orchestrator,
        agent_b: Orchestrator,
        benchmark: Benchmark,
        scorer: Scorer,
        experience_log: ExperienceLog,
    ) -> None:
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.benchmark = benchmark
        self.scorer = scorer
        self.experience_log = experience_log

    async def run_round(self, item: BenchmarkItem) -> CompetitiveResult:
        """Run a single competitive round on one benchmark item."""
        result_a, result_b = await asyncio.gather(
            self.agent_a.run(item.question),
            self.agent_b.run(item.question),
        )

        score_a = self.scorer.exact_match(result_a.answer, item.expected_answer)
        score_b = self.scorer.exact_match(result_b.answer, item.expected_answer)

        if score_a > score_b:
            winner = "a"
        elif score_b > score_a:
            winner = "b"
        else:
            winner = "tie"

        competitive_result = CompetitiveResult(
            test_item=item,
            agent_a_answer=result_a.answer,
            agent_b_answer=result_b.answer,
            agent_a_score=score_a,
            agent_b_score=score_b,
            winner=winner,
            agent_a_primary_specialist=result_a.coordinator_output.primary_specialist,
            agent_b_primary_specialist=result_b.coordinator_output.primary_specialist,
            agent_a_attribution=result_a.attribution,
            agent_b_attribution=result_b.attribution,
        )

        # Log both agents' interactions with competitive context
        self._log_interaction(result_a, item, competitive_result, agent_label="a")
        self._log_interaction(result_b, item, competitive_result, agent_label="b")

        return competitive_result

    async def run_session(self, num_rounds: int = 50) -> list[CompetitiveResult]:
        """Run a full competitive session across benchmark items."""
        items = await self.benchmark.load()
        items = items[:num_rounds]

        results: list[CompetitiveResult] = []
        for item in items:
            result = await self.run_round(item)
            results.append(result)

        return results

    def session_summary(self, results: list[CompetitiveResult]) -> dict:
        """Summarize a competitive session."""
        a_wins = sum(1 for r in results if r.winner == "a")
        b_wins = sum(1 for r in results if r.winner == "b")
        ties = sum(1 for r in results if r.winner == "tie")
        a_avg = sum(r.agent_a_score for r in results) / len(results) if results else 0
        b_avg = sum(r.agent_b_score for r in results) / len(results) if results else 0
        return {
            "rounds": len(results),
            "agent_a_wins": a_wins,
            "agent_b_wins": b_wins,
            "ties": ties,
            "agent_a_avg_score": round(a_avg, 3),
            "agent_b_avg_score": round(b_avg, 3),
        }

    def _log_interaction(
        self,
        result: OrchestratorResult,
        item: BenchmarkItem,
        competitive: CompetitiveResult,
        agent_label: str,
    ) -> None:
        """Log an interaction record enriched with competitive scoring."""
        score = (
            competitive.agent_a_score
            if agent_label == "a"
            else competitive.agent_b_score
        )

        # Compute vindication: which non-primary specialists had the right answer?
        primary = result.coordinator_output.primary_specialist
        vindication: dict[str, bool] = {}
        for name, out in result.round2_outputs.items():
            if name != primary:
                spec_score = self.scorer.exact_match(out.answer, item.expected_answer)
                vindication[name] = spec_score > score

        record = InteractionRecord(
            input=item.question,
            round1_outputs=result.round1_outputs,
            round2_outputs=result.round2_outputs,
            coordinator_output=result.coordinator_output,
            attribution=result.attribution,
            primary_specialist=primary,
            outcome_score=score,
            vindication=vindication,
            log_priority="high" if competitive.winner != "tie" else "medium",
        )
        self.experience_log.append(record)
