"""Sandboxed execution: the environment's verdicts and an agent's probes."""

from src.config import FeedbackLevel
from src.execution.sandbox import Sandbox, behaves_alike
from src.models.code import CodeProblem

PROBLEM = CodeProblem(
    task_id="t",
    prompt="add",
    entry_point="add",
    tests=["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
)


def sandbox(level: FeedbackLevel = FeedbackLevel.BINARY) -> Sandbox:
    return Sandbox(level, timeout_seconds=2)


class TestRunTests:
    async def test_correct_solution_passes(self):
        r = await sandbox().run_tests("def add(a, b):\n    return a + b", PROBLEM)
        assert r.passed and r.tests_passed == 2 and r.feedback == ""

    async def test_wrong_solution_only_told_wrong(self):
        r = await sandbox().run_tests("def add(a, b):\n    return a - b", PROBLEM)
        assert not r.passed
        assert r.tests_passed == 1  # environment bookkeeping (0+0 == 0-0)
        assert r.feedback == "Your solution is wrong."

    async def test_feedback_never_contains_tests(self):
        r = await sandbox(FeedbackLevel.ERROR_TYPE).run_tests(
            "def add(a, b):\n    return 0", PROBLEM
        )
        assert "assert" not in r.feedback and "3" not in r.feedback
        assert r.feedback == "Your solution is wrong. Errors observed: AssertionError."

    async def test_syntax_error(self):
        r = await sandbox().run_tests("def add(a b):\n    return a", PROBLEM)
        assert not r.passed and r.error_types == ["SyntaxError"]

    async def test_runtime_error_type(self):
        r = await sandbox().run_tests("def add(a, b):\n    return [][0]", PROBLEM)
        assert r.error_types == ["IndexError"]

    async def test_timeout(self):
        r = await sandbox().run_tests("def add(a, b):\n    while True: pass", PROBLEM)
        assert r.timed_out and r.error_types == ["Timeout"] and not r.passed

    async def test_solution_cannot_exit_harness(self):
        r = await sandbox().run_tests("import sys\nsys.exit(0)", PROBLEM)
        assert not r.passed and r.error_types == ["SystemExit"]

    async def test_setup_code_runs_first(self):
        problem = PROBLEM.model_copy(
            update={"setup_code": "import math", "tests": ["assert add(1, 1) == 2"]}
        )
        r = await sandbox().run_tests(
            "def add(a, b):\n    return int(math.fsum([a, b]))", problem
        )
        assert r.passed


class TestProbes:
    async def test_outputs_and_exceptions(self):
        out = await sandbox().run_probes(
            "def f(x):\n    return 10 // x", ["f(2)", "f(0)"]
        )
        assert out == ["5", "!ZeroDivisionError"]

    async def test_unloadable_code(self):
        assert await sandbox().run_probes("def f(:", ["f(1)"]) is None


class TestBehavesAlike:
    def test_same_outputs(self):
        assert behaves_alike(["1", "2"], ["1", "2"]) is True

    def test_different_outputs(self):
        assert behaves_alike(["1", "2"], ["1", "3"]) is False

    def test_ignores_probes_where_reference_raises(self):
        assert behaves_alike(["1", "!TypeError"], ["1", "anything"]) is True

    def test_unknown_when_nothing_comparable(self):
        assert behaves_alike(["!TypeError"], ["1"]) is None
        assert behaves_alike(None, ["1"]) is None

    def test_candidate_that_fails_to_load_differs(self):
        assert behaves_alike(["1"], None) is False
