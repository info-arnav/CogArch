"""Runs code in an isolated subprocess with a timeout and a memory cap."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.config import FeedbackLevel
from src.execution.feedback import describe_failure
from src.models.code import CodeProblem, ExecutionResult

_HARNESS = Path(__file__).with_name("harness.py")


def _limit_resources(memory_limit_mb: int) -> Callable[[], None] | None:
    """Address-space limit for the child (Linux only; macOS does not enforce RLIMIT_AS)."""
    if not sys.platform.startswith("linux"):
        return None

    def apply() -> None:
        import resource

        limit = memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    return apply


class Sandbox:
    """Bounded-parallelism code execution.

    `run_tests` is the environment judging a submission.
    `run_probes` is an agent trying code on inputs it wrote itself.
    """

    def __init__(
        self,
        feedback: FeedbackLevel,
        timeout_seconds: float = 10.0,
        memory_limit_mb: int = 1024,
        max_parallel: int | None = None,
    ) -> None:
        self.feedback = feedback
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self._slots = asyncio.Semaphore(max_parallel or os.cpu_count() or 4)

    async def run_tests(self, code: str, problem: CodeProblem) -> ExecutionResult:
        raw, timed_out = await self._execute(
            {
                "mode": "tests",
                "solution": code,
                "setup_code": problem.setup_code,
                "tests": problem.tests,
            }
        )
        total = len(problem.tests)
        if timed_out:
            result = ExecutionResult(
                passed=False,
                tests_passed=0,
                tests_total=total,
                error_types=["Timeout"],
                timed_out=True,
            )
        elif raw is None:
            # Harness died without reporting (memory limit, os._exit, ...).
            result = ExecutionResult(
                passed=False,
                tests_passed=0,
                tests_total=total,
                error_types=["ProcessCrashed"],
            )
        elif raw["load_error"] is not None:
            result = ExecutionResult(
                passed=False,
                tests_passed=0,
                tests_total=total,
                error_types=[raw["load_error"]],
            )
        else:
            outcomes = raw["tests"]
            passed = sum(t["passed"] for t in outcomes)
            result = ExecutionResult(
                passed=total > 0 and passed == total,
                tests_passed=passed,
                tests_total=total,
                error_types=list(
                    dict.fromkeys(t["type"] for t in outcomes if not t["passed"])
                ),
            )
        result.feedback = describe_failure(result, self.feedback)
        return result

    async def run_probes(
        self, code: str, probes: list[str], setup_code: str = ""
    ) -> list[str] | None:
        """repr() of each probe's value (or '!ExceptionType'); None if the code
        does not load or the process times out or crashes."""
        raw, timed_out = await self._execute(
            {
                "mode": "probes",
                "solution": code,
                "setup_code": setup_code,
                "probes": probes,
            }
        )
        if timed_out or raw is None or raw["load_error"] is not None:
            return None
        return list(raw["outputs"])

    async def _execute(self, payload: dict[str, Any]) -> tuple[dict | None, bool]:
        async with self._slots:
            with tempfile.TemporaryDirectory(prefix="cogarch-") as workdir:
                result_path = Path(workdir) / "result.json"
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-I",
                    str(_HARNESS),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    cwd=workdir,
                    env={"PYTHONHASHSEED": "0", "PATH": os.environ.get("PATH", "")},
                    preexec_fn=_limit_resources(self.memory_limit_mb),
                )
                body = json.dumps({**payload, "result_path": str(result_path)})
                try:
                    await asyncio.wait_for(
                        proc.communicate(body.encode()), timeout=self.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    return None, True
                if not result_path.exists():
                    return None, False
                return json.loads(result_path.read_text()), False


def behaves_alike(
    reference: list[str] | None, candidate: list[str] | None
) -> bool | None:
    """Compare two solutions' probe outputs.

    Probes on which the reference itself raises are ignored (the agent wrote a
    bad example call). Returns None when nothing can be compared.
    """
    if reference is None:
        return None
    usable = [i for i, out in enumerate(reference) if not out.startswith("!")]
    if not usable:
        return None
    if candidate is None:
        return False
    return all(reference[i] == candidate[i] for i in usable)
