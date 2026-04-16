"""Tests for code data models."""

from src.models.code import (
    AgentResult,
    AttemptResult,
    SpecialistSolution,
)


class TestAttemptResult:
    def test_pass_rate_full(self):
        r = AttemptResult(attempt_number=1, code="x", pass_count=3, total_tests=3)
        assert r.pass_rate == 1.0

    def test_pass_rate_zero(self):
        r = AttemptResult(attempt_number=1, code="x", pass_count=0, total_tests=3)
        assert r.pass_rate == 0.0

    def test_pass_rate_no_tests(self):
        r = AttemptResult(attempt_number=1, code="x", pass_count=0, total_tests=0)
        assert r.pass_rate == 0.0

    def test_passed_all(self):
        r = AttemptResult(attempt_number=1, code="x", pass_count=2, total_tests=2)
        assert r.passed_all

    def test_not_passed_all_partial(self):
        r = AttemptResult(attempt_number=1, code="x", pass_count=1, total_tests=2)
        assert not r.passed_all


class TestSpecialistSolution:
    def test_best_attempt_highest_pass_rate(self):
        sol = SpecialistSolution(specialist_name="logical")
        sol.attempts.append(
            AttemptResult(attempt_number=1, code="a", pass_count=1, total_tests=3)
        )
        sol.attempts.append(
            AttemptResult(attempt_number=2, code="b", pass_count=3, total_tests=3)
        )
        assert sol.best_attempt.pass_count == 3

    def test_best_pass_rate(self):
        sol = SpecialistSolution(specialist_name="logical")
        sol.attempts.append(
            AttemptResult(attempt_number=1, code="a", pass_count=2, total_tests=4)
        )
        assert sol.best_pass_rate == 0.5

    def test_empty_solution(self):
        sol = SpecialistSolution(specialist_name="logical")
        assert sol.best_attempt is None
        assert sol.best_pass_rate == 0.0


class TestAgentResult:
    def test_best_solution_highest_pass_rate(self):
        sol_a = SpecialistSolution(specialist_name="logical")
        sol_a.attempts.append(
            AttemptResult(attempt_number=1, code="x", pass_count=1, total_tests=3)
        )
        sol_b = SpecialistSolution(specialist_name="creative")
        sol_b.attempts.append(
            AttemptResult(attempt_number=1, code="y", pass_count=3, total_tests=3)
        )
        agent = AgentResult(
            problem_id="p1",
            solutions={"logical": sol_a, "creative": sol_b},
        )
        assert agent.best_solution.specialist_name == "creative"
        assert agent.pass_rate == 1.0

    def test_empty_agent(self):
        agent = AgentResult(problem_id="p1")
        assert agent.pass_rate == 0.0
