"""Tests for the sandboxed code execution engine."""

from src.execution.code_runner import format_feedback, run_attempt


class TestRunAttempt:
    def test_correct_solution_passes_all(self):
        code = "def add(a, b):\n    return a + b"
        assertions = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
        result = run_attempt(code, assertions)
        assert result.pass_count == 2
        assert result.total_tests == 2
        assert result.passed_all

    def test_wrong_solution_fails(self):
        code = "def add(a, b):\n    return a - b"
        assertions = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
        result = run_attempt(code, assertions)
        assert result.pass_count < 2
        assert not result.passed_all

    def test_partial_credit(self):
        code = "def add(a, b):\n    return a + b + 1"
        assertions = [
            "assert add(0, 0) == 0",
            "assert add(1, 1) == 2",
            "assert add(2, 2) == 4",
        ]
        result = run_attempt(code, assertions)
        # add(0,0)=1≠0, add(1,1)=3≠2, add(2,2)=5≠4 — all fail, but code runs
        assert result.total_tests == 3
        assert result.pass_count == 0

    def test_syntax_error_returns_zero_passes(self):
        code = "def add(a b):\n    return a + b"
        assertions = ["assert add(1, 2) == 3"]
        result = run_attempt(code, assertions)
        assert result.pass_count == 0
        assert result.error != ""

    def test_timeout_handled(self):
        code = "def loop(n):\n    while True: pass"
        assertions = ["assert loop(1) == 1"]
        result = run_attempt(code, assertions, timeout=1.0)
        assert result.timed_out or result.pass_count == 0

    def test_empty_assertions_returns_zero(self):
        code = "def f(): pass"
        result = run_attempt(code, [])
        assert result.total_tests == 0

    def test_pass_rate_property(self):
        code = "def add(a, b):\n    return a + b"
        assertions = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
        result = run_attempt(code, assertions)
        assert result.pass_rate == 1.0

    def test_failed_assertions_reported(self):
        code = "def add(a, b):\n    return 0"
        assertions = ["assert add(1, 2) == 3"]
        result = run_attempt(code, assertions)
        assert len(result.failed_assertions) > 0


class TestFormatFeedback:
    def test_timeout_message(self):
        from src.models.code import AttemptResult

        r = AttemptResult(
            attempt_number=1, code="x", pass_count=0, total_tests=1, timed_out=True
        )
        fb = format_feedback(r)
        assert "timed out" in fb.lower()

    def test_partial_pass_message(self):
        from src.models.code import AttemptResult

        r = AttemptResult(
            attempt_number=1,
            code="x",
            pass_count=1,
            total_tests=3,
            failed_assertions=["assert f(0)==0"],
        )
        fb = format_feedback(r)
        assert "1/3" in fb
