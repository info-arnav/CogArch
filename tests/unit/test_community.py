"""Community rounds, discovery sharing, credit assignment and DPO pairs."""

from src.inference.prompts import load_specialist
from src.models.code import AgentResult, Decision, ExecutionResult
from src.training.community import episode_text, play_round, share_discoveries
from src.training.preferences import COORDINATOR, build_pairs
from tests.conftest import (
    SPECIALISTS,
    ScriptedChat,
    code_reply,
    is_coordinator,
    make_agent,
)

RIGHT = "def add(a, b):\n    return a + b"
RIGHT_ALT = "def add(a, b):\n    return b + a"
WRONG = "def add(a, b):\n    return a - b"


def result(agent: str, solved_code: str | None) -> AgentResult:
    decisions = [
        Decision(
            attempt_number=1,
            proposals=[],
            coordinator_prompt=[],
            coordinator_output="",
            code=solved_code or WRONG,
            result=ExecutionResult(
                passed=solved_code is not None, tests_passed=0, tests_total=1
            ),
        )
    ]
    return AgentResult(agent=agent, problem_id="p", decisions=decisions)


class TestShareDiscoveries:
    def test_peer_solution_is_shared(self):
        known = share_discoveries(
            {"a": result("a", RIGHT), "b": result("b", None)}, True
        )
        assert known["a"].found_by == "a" and known["b"].found_by == "a"

    def test_no_sharing_when_disabled(self):
        known = share_discoveries(
            {"a": result("a", RIGHT), "b": result("b", None)}, False
        )
        assert "b" not in known

    def test_nobody_knows_when_nobody_solved(self):
        assert (
            share_discoveries({"a": result("a", None), "b": result("b", None)}, True)
            == {}
        )


def make_team(solver: bool):
    """Agent `a` (solver=True): logical/skeptical right; the coordinator picks logical.
    Agent `b` (solver=False): every specialist wrong except skeptical (an equivalent
    variant); the coordinator always picks creative's wrong code."""

    def reply(system: str, user: str, model: str) -> str:
        if system.startswith("You write example calls"):
            return "add(1, 2)\nadd(-3, 3)\nadd(10, 5)"
        if system.startswith("You review"):
            return "- Check signs"
        if is_coordinator(system):
            pick = ("logical", RIGHT) if solver else ("creative", WRONG)
            return f"BASED ON: {pick[0]}\n{code_reply(pick[1])}"
        if "Logical" in system:
            return code_reply(RIGHT if solver else WRONG)
        if "Skeptical" in system:
            return code_reply(RIGHT_ALT)
        return code_reply(WRONG)

    return reply


async def test_round_shares_and_credits(add_problem):
    agents = {
        "a": make_agent(ScriptedChat(make_team(True)), "a"),
        "b": make_agent(ScriptedChat(make_team(False)), "b", max_attempts=2),
    }
    r = await play_round(agents, add_problem, 1, agents["a"].sandbox, share=True)

    assert r.results["a"].solved and not r.results["b"].solved
    assert r.known_solution["b"].found_by == "a"
    # b checks its own proposals against a's solution on its own example calls.
    assert r.agreement["b"]["1:skeptical"] is True
    assert r.agreement["b"]["1:creative"] is False
    assert "agent a" in episode_text(r, "b")

    personas = {n: load_specialist(n) for n in SPECIALISTS}
    pairs_b = build_pairs([r], "b", personas)
    # Wrong first-attempt specialists learn from the peer's solution.
    assert {p.role for p in pairs_b["logical"]} == {"logical"}
    assert pairs_b["skeptical"] == []  # skeptical was already right
    assert all(p.source == "a" for p in pairs_b["logical"])
    # Coordinator: picking skeptical's proposal is preferred over what it chose.
    coord = pairs_b[COORDINATOR]
    assert len(coord) == 2 and coord[0].chosen.startswith("BASED ON: skeptical")

    pairs_a = build_pairs([r], "a", personas)
    assert all(p.source == "self" for ps in pairs_a.values() for p in ps)
    assert pairs_a[COORDINATOR][0].chosen.startswith("BASED ON: logical")


async def test_solo_agent_learns_nothing_from_unsolved(add_problem):
    agents = {"b": make_agent(ScriptedChat(make_team(False)), "b", max_attempts=1)}
    r = await play_round(agents, add_problem, 1, agents["b"].sandbox, share=True)
    assert r.known_solution == {}
    pairs = build_pairs([r], "b", {n: load_specialist(n) for n in SPECIALISTS})
    assert all(p == [] for p in pairs.values())
