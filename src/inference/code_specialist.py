"""Code specialist — generates Python solutions with iterative error feedback."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from src.execution.code_runner import format_feedback, run_attempt
from src.inference.backends.base import LLMBackend
from src.models.code import AttemptResult, CodeProblem, SpecialistSolution

if TYPE_CHECKING:
    from src.memory.controller import MemoryController


def _load_prompt(name: str) -> dict[str, object]:
    path = Path(f"prompts/specialists/code_{name}.yaml")
    if not path.exists():
        path = Path(f"prompts/specialists/{name}.yaml")
    with open(path) as f:
        result: dict[str, object] = yaml.safe_load(f)
        return result


def _extract_code(raw: str) -> str:
    """Pull Python code out of a markdown code block, or return raw if no block."""
    match = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw.strip()


class CodeSpecialist:
    """One specialist that iteratively refines code given execution feedback."""

    def __init__(
        self,
        name: str,
        backend: LLMBackend,
        model: str = "deepseek-coder:33b",
        max_attempts: int = 10,
        memory: MemoryController | None = None,
    ) -> None:
        self.name = name
        self.backend = backend
        self.model = model
        self.max_attempts = max_attempts
        self.memory = memory
        cfg = _load_prompt(name)
        self.system_prompt: str = str(cfg["system_prompt"])
        self.temperature: float = float(cfg.get("temperature", 0.5))  # type: ignore[arg-type]
        self.max_tokens: int = int(cfg.get("max_tokens", 1024))  # type: ignore[call-overload]

    async def solve(self, problem: CodeProblem) -> SpecialistSolution:
        """Attempt to solve problem up to max_attempts times with feedback."""
        solution = SpecialistSolution(specialist_name=self.name)
        previous_attempt: AttemptResult | None = None

        # Retrieve relevant memory context before first attempt
        memory_text = ""
        if self.memory:
            ctx = await self.memory.retrieve_context(problem.prompt)
            if not ctx.is_empty():
                memory_text = ctx.to_prompt_text()

        for attempt_num in range(1, self.max_attempts + 1):
            prompt = self._build_prompt(problem, previous_attempt, memory_text)
            # Only inject memory on the first attempt — subsequent prompts
            # already have error feedback which is more relevant
            memory_text = ""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            raw = await self.backend.generate(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            code = _extract_code(raw)
            result = run_attempt(
                solution_code=code,
                assertions=problem.test_assertions,
                attempt_number=attempt_num,
            )
            solution.attempts.append(result)

            if result.passed_all:
                break
            previous_attempt = result

        # Record the outcome to memory
        if self.memory:
            best = solution.best_attempt
            if best:
                passed = best.passed_all
                importance = 0.9 if passed else (0.6 if best.pass_count > 0 else 0.3)
                content = (
                    f"[{self.name}] Problem '{problem.task_id}': "
                    f"{'solved' if passed else 'partial'} "
                    f"({best.pass_count}/{best.total_tests} tests, "
                    f"{best.attempt_number} attempt(s))"
                )
                await self.memory.record(
                    content,
                    importance=importance,
                    metadata={
                        "specialist": self.name,
                        "problem_id": problem.task_id,
                        "problem_type": problem.source,
                        "passed_all": passed,
                        "pass_count": best.pass_count,
                        "total_tests": best.total_tests,
                        "attempts": best.attempt_number,
                        "error": best.error,
                        "timed_out": best.timed_out,
                    },
                )

        return solution

    def _build_prompt(
        self,
        problem: CodeProblem,
        previous: AttemptResult | None,
        memory_text: str = "",
    ) -> str:
        lines = [f"Problem:\n{problem.prompt}"]

        if memory_text:
            lines.append(f"\n{memory_text}")

        lines.append(f"\nImplement the function named `{problem.entry_point}`.")

        if previous is not None:
            lines.append("\nYour previous attempt:")
            lines.append(f"```python\n{previous.code}\n```")
            lines.append("\nFeedback:")
            lines.append(format_feedback(previous))
            lines.append("\nFix the solution and try again.")

        return "\n".join(lines)
