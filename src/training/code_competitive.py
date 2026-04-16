"""Code competitive trainer — two agents independently solve problems.

Neither agent sees the other's code. Pass rate on hidden test cases
determines the winner. Even when both fail, partial pass rates create
ranked DPO pairs so no round is wasted.
"""

from __future__ import annotations

import asyncio

from src.inference.code_orchestrator import CodeOrchestrator
from src.models.code import AgentResult, CodeCompetitiveResult, CodeProblem


class CodeCompetitiveTrainer:
    """Runs two independent agents on the same problem; determines winner by pass rate."""

    def __init__(
        self,
        agent_a: CodeOrchestrator,
        agent_b: CodeOrchestrator,
    ) -> None:
        self.agent_a = agent_a
        self.agent_b = agent_b

    async def run_round(self, problem: CodeProblem) -> CodeCompetitiveResult:
        """Run both agents independently on the same problem."""
        result_a, result_b = await asyncio.gather(
            self.agent_a.solve(problem),
            self.agent_b.solve(problem),
        )

        winner = _determine_winner(result_a, result_b)

        return CodeCompetitiveResult(
            problem=problem,
            agent_a=result_a,
            agent_b=result_b,
            winner=winner,
        )

    async def run_session(
        self, problems: list[CodeProblem]
    ) -> list[CodeCompetitiveResult]:
        """Run competitive rounds on a list of problems sequentially."""
        results = []
        for problem in problems:
            result = await self.run_round(problem)
            results.append(result)
        return results

    def session_summary(self, results: list[CodeCompetitiveResult]) -> dict:
        a_wins = sum(1 for r in results if r.winner == "a")
        b_wins = sum(1 for r in results if r.winner == "b")
        ties = sum(1 for r in results if r.winner == "tie")
        both_fail = sum(
            1 for r in results if r.agent_a.pass_rate == 0 and r.agent_b.pass_rate == 0
        )
        avg_a = (
            sum(r.agent_a.pass_rate for r in results) / len(results) if results else 0
        )
        avg_b = (
            sum(r.agent_b.pass_rate for r in results) / len(results) if results else 0
        )
        return {
            "rounds": len(results),
            "agent_a_wins": a_wins,
            "agent_b_wins": b_wins,
            "ties": ties,
            "both_failed": both_fail,
            "agent_a_avg_pass_rate": round(avg_a, 3),
            "agent_b_avg_pass_rate": round(avg_b, 3),
        }


def _determine_winner(a: AgentResult, b: AgentResult) -> str:
    """Compare pass rates; ties broken by number of attempts (fewer = better)."""
    rate_a = a.pass_rate
    rate_b = b.pass_rate

    if abs(rate_a - rate_b) < 1e-6:
        # Same pass rate — fewer attempts wins (more efficient)
        attempts_a = _best_attempts(a)
        attempts_b = _best_attempts(b)
        if attempts_a < attempts_b:
            return "a"
        if attempts_b < attempts_a:
            return "b"
        return "tie"

    return "a" if rate_a > rate_b else "b"


def _best_attempts(agent: AgentResult) -> int:
    best = agent.best_solution
    if best and best.best_attempt:
        return best.best_attempt.attempt_number
    return 99
