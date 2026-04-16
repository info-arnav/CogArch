"""Code orchestrator — runs all specialists and picks the best solution."""

from __future__ import annotations

import asyncio

from src.inference.code_specialist import CodeSpecialist
from src.models.code import AgentResult, CodeProblem, SpecialistSolution


class CodeOrchestrator:
    """Runs all specialists in parallel; coordinator picks the highest pass-rate solution."""

    def __init__(
        self,
        specialists: dict[str, CodeSpecialist],
    ) -> None:
        self.specialists = specialists

    async def solve(self, problem: CodeProblem) -> AgentResult:
        """Run all specialists concurrently; return the agent's best result."""
        tasks = {name: spec.solve(problem) for name, spec in self.specialists.items()}
        results: list[SpecialistSolution] = await asyncio.gather(*tasks.values())
        solutions = dict(zip(tasks.keys(), results))

        # Coordinator: pick specialist with highest pass rate; break ties by fewest attempts
        def _sort_key(n: str) -> tuple[float, int]:
            best = solutions[n].best_attempt
            return (solutions[n].best_pass_rate, -(best.attempt_number if best else 99))

        best_name = max(solutions, key=_sort_key)

        return AgentResult(
            problem_id=problem.task_id,
            solutions=solutions,
            chosen_specialist=best_name,
        )
