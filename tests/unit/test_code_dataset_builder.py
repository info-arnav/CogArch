"""Tests for code DPO dataset builder."""

from src.models.code import (
    AgentResult,
    AttemptResult,
    CodeCompetitiveResult,
    CodeProblem,
    SpecialistSolution,
)
from src.training.code_dataset_builder import build_dpo_pairs, save


def _make_problem() -> CodeProblem:
    return CodeProblem(
        task_id="test/1",
        prompt="Return sum of two numbers",
        entry_point="add",
        test_assertions=["assert add(1,2)==3"],
        source="test",
    )


def _make_attempt(code: str, passes: int, total: int, n: int = 1) -> AttemptResult:
    return AttemptResult(
        attempt_number=n,
        code=code,
        pass_count=passes,
        total_tests=total,
    )


def _make_solution(
    name: str, passes: int, total: int, code: str = "def add(a,b): return a+b"
) -> SpecialistSolution:
    sol = SpecialistSolution(specialist_name=name)
    sol.attempts.append(_make_attempt(code, passes, total))
    return sol


def _make_result(
    a_passes: int,
    b_passes: int,
    total: int = 3,
    code_a: str = "code_a",
    code_b: str = "code_b",
) -> CodeCompetitiveResult:
    from src.training.code_competitive import _determine_winner

    problem = _make_problem()
    sol_a = _make_solution("logical", a_passes, total, code_a)
    sol_b = _make_solution("logical", b_passes, total, code_b)

    agent_a = AgentResult(problem_id=problem.task_id, solutions={"logical": sol_a})
    agent_b = AgentResult(problem_id=problem.task_id, solutions={"logical": sol_b})
    winner = _determine_winner(agent_a, agent_b)

    return CodeCompetitiveResult(
        problem=problem, agent_a=agent_a, agent_b=agent_b, winner=winner
    )


class TestBuildDpoPairs:
    def test_winner_is_chosen(self):
        result = _make_result(a_passes=3, b_passes=1, code_a="good", code_b="bad")
        pairs = build_dpo_pairs([result])
        assert "logical" in pairs
        ex = pairs["logical"][0]
        assert ex.chosen_code == "good"
        assert ex.rejected_code == "bad"

    def test_both_fail_still_produces_pair(self):
        result = _make_result(a_passes=2, b_passes=0, code_a="partial", code_b="wrong")
        pairs = build_dpo_pairs([result])
        assert len(pairs.get("logical", [])) == 1
        ex = pairs["logical"][0]
        assert ex.training_signal == "partial_win"

    def test_both_zero_still_ranked_by_order(self):
        result = _make_result(
            a_passes=0, b_passes=0, code_a="attempt1", code_b="attempt2"
        )
        pairs = build_dpo_pairs([result])
        # Different code, same pass rate → still a pair (different approaches)
        assert "logical" in pairs

    def test_identical_code_skipped(self):
        result = _make_result(a_passes=0, b_passes=0, code_a="same", code_b="same")
        pairs = build_dpo_pairs([result])
        assert len(pairs.get("logical", [])) == 0

    def test_win_signal_label(self):
        result = _make_result(a_passes=3, b_passes=0, total=3)
        pairs = build_dpo_pairs([result])
        assert pairs["logical"][0].training_signal == "win"

    def test_save_writes_jsonl(self, tmp_path):
        result = _make_result(a_passes=3, b_passes=0)
        pairs = build_dpo_pairs([result])
        paths = save(pairs, tmp_path)
        assert "logical" in paths
        assert paths["logical"].exists()
        lines = paths["logical"].read_text().strip().splitlines()
        assert len(lines) == 1
