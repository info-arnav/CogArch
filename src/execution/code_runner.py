"""Sandboxed Python code execution using subprocess with timeout.

Each solution runs in a separate Python process — no shared state,
no Docker required on Colab. Each assertion is tested individually
so partial credit is always available.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from src.models.code import AttemptResult

_ASSERTION_WRAPPER = """\
import sys as _sys

{solution_code}

_pass = 0
_fail = 0
_failed = []

def _test(assertion_str):
    global _pass, _fail
    try:
        exec(compile(assertion_str, "<test>", "exec"), globals())
        _pass += 1
    except AssertionError as _e:
        _fail += 1
        _failed.append(assertion_str[:120])
    except Exception as _e:
        _fail += 1
        _failed.append(f"{{type(_e).__name__}}: {{_e}} | {{assertion_str[:80]}}")

{test_calls}

print(f"COGARCH_PASS:{{_pass}}")
print(f"COGARCH_FAIL:{{_fail}}")
for _f in _failed:
    print(f"COGARCH_FAILED_TEST:{{_f}}")
"""


def run_attempt(
    solution_code: str,
    assertions: list[str],
    attempt_number: int = 1,
    timeout: float = 10.0,
) -> AttemptResult:
    """Execute solution_code against a list of assert statements.

    Returns an AttemptResult with pass/fail counts and failed assertion strings.
    The test assertions are never shown to the agent — only the failure messages
    and error output are fed back for the next attempt.
    """
    if not assertions:
        return AttemptResult(
            attempt_number=attempt_number,
            code=solution_code,
            pass_count=0,
            total_tests=0,
            error="No test assertions provided",
        )

    test_calls = "\n".join(f"_test({repr(a)})" for a in assertions)
    full_code = _ASSERTION_WRAPPER.format(
        solution_code=solution_code,
        test_calls=test_calls,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return _parse_output(
            stdout=proc.stdout,
            stderr=proc.stderr,
            solution_code=solution_code,
            attempt_number=attempt_number,
            total=len(assertions),
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        return AttemptResult(
            attempt_number=attempt_number,
            code=solution_code,
            pass_count=0,
            total_tests=len(assertions),
            error="Execution timed out",
            timed_out=True,
        )
    finally:
        os.unlink(tmp_path)


def _parse_output(
    stdout: str,
    stderr: str,
    solution_code: str,
    attempt_number: int,
    total: int,
    timed_out: bool,
) -> AttemptResult:
    pass_count = 0
    failed: list[str] = []
    error = ""

    for line in stdout.splitlines():
        if line.startswith("COGARCH_PASS:"):
            try:
                pass_count = int(line.split(":", 1)[1])
            except ValueError:
                pass
        elif line.startswith("COGARCH_FAILED_TEST:"):
            failed.append(line.split(":", 1)[1])

    if stderr.strip():
        # Trim tracebacks to last 8 lines — enough to understand the error
        tb_lines = stderr.strip().splitlines()
        error = "\n".join(tb_lines[-8:])

    # If stderr present and no pass marker, the solution itself crashed before tests
    if "COGARCH_PASS:" not in stdout and stderr.strip():
        pass_count = 0

    return AttemptResult(
        attempt_number=attempt_number,
        code=solution_code,
        pass_count=pass_count,
        total_tests=total,
        failed_assertions=failed,
        error=error,
        timed_out=timed_out,
    )


def format_feedback(result: AttemptResult) -> str:
    """Format execution result as feedback for the next attempt."""
    lines = []
    if result.timed_out:
        lines.append("Your solution timed out (> 10s). It likely has an infinite loop.")
    elif result.error and result.pass_count == 0:
        lines.append(
            f"Your solution raised an error before any tests ran:\n{result.error}"
        )
    else:
        lines.append(f"Passed {result.pass_count}/{result.total_tests} tests.")
        if result.failed_assertions:
            lines.append("Failed tests (showing up to 5):")
            for f in result.failed_assertions[:5]:
                lines.append(f"  - {f}")
        if result.error:
            lines.append(f"Error output:\n{result.error}")
    return "\n".join(lines)
