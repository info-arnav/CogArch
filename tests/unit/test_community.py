"""Community rounds, sharing, credit assignment and competitive training pairs."""

from src.models.core import AgentResult, Decision, Proposal, Verdict
from src.training.community import episode_text, play_round, share_discoveries
from src.training.preferences import (
    COORDINATOR,
    DomainRecords,
    build_pairs,
    discussion_stats,
    specialist_accuracy,
)
from tests.conftest import SPECIALISTS, ScriptedChat, code_reply, make_agent


def decision(answer: str, passed: bool) -> Decision:
    return Decision(
        attempt_number=1,
        round1=[],
        round2=[],
        coordinator_prompt=[],
        coordinator_output=f"\\boxed{{{answer}}}",
        answer=answer,
        result=Verdict(passed=passed),
    )


class TestShareDiscoveries:
    def test_peer_answer_is_shared(self):
        results = {
            "a": AgentResult(agent="a", task_id="t", decisions=[decision("42", True)]),
            "b": AgentResult(agent="b", task_id="t", decisions=[decision("41", False)]),
        }
        known = share_discoveries(results, share=True)
        assert known["b"].found_by == "a" and known["b"].answer == "42"
        assert "b" not in share_discoveries(results, share=False)


def solver(role: str, stage: str, user: str) -> str:
    """Agent a: skeptical right from round 1; coordinator follows it."""
    if role == "coordinator":
        return "\\boxed{42}"
    return "\\boxed{42}" if role == "skeptical" else "\\boxed{36}"


def struggler(role: str, stage: str, user: str) -> str:
    """Agent b: only empathetic gets it, and only after discussion; coordinator wrong."""
    if role == "coordinator":
        return "\\boxed{36}"
    if role == "empathetic" and stage == "round2":
        return "\\boxed{42}"
    return "\\boxed{36}"


async def test_round_credit_and_competitive_pairs(math_task):
    agents = {
        "a": make_agent(ScriptedChat(solver), "a"),
        "b": make_agent(ScriptedChat(struggler), "b", max_attempts=1),
    }
    r = await play_round(
        agents, math_task, 1, agents["a"].environment.sandbox, share=True
    )
    assert r.results["a"].solved and not r.results["b"].solved
    assert r.known["b"].found_by == "a"
    assert r.correct["b"]["1:2:empathetic"] is True
    assert r.correct["b"]["1:1:empathetic"] is False
    assert "agent a" in episode_text(r, "b")

    records_b = DomainRecords(SPECIALISTS)
    pairs_b = build_pairs([r], "b", records_b)
    # Empathetic only got it right AFTER the discussion, so it is not a winner:
    # nobody in b solved it alone, so the domain owner learns it (tie -> logical).
    # The coordinator adopted the wrong "36" that creative and skeptical argued
    # for, so they are blamed and learn the correct answer too.
    roles = {
        role: p[0].responsibility
        for role, p in pairs_b.items()
        if p and role != COORDINATOR
    }
    assert roles == {"logical": "owner", "creative": "blamed", "skeptical": "blamed"}
    pair = pairs_b["logical"][0]
    assert pair.source == "a" and "42" in pair.chosen and "36" in pair.rejected
    assert records_b.get("empathetic", "math") < 0.5
    # The coordinator should still have sided with empathetic's revised position.
    assert pairs_b[COORDINATOR][0].chosen.startswith("empathetic's reasoning")

    records_a = DomainRecords(SPECIALISTS)
    pairs_a = build_pairs([r], "a", records_a)
    (pair_a,) = pairs_a["skeptical"]
    assert (
        "36" in pair_a.rejected
    )  # a teammate's wrong reply, since skeptical was never wrong

    assert specialist_accuracy([r], "b", SPECIALISTS)["empathetic"] == {"math": 0.0}
    changed, fixed = discussion_stats([r], "b")
    assert changed == 0.25 and fixed == 1.0


async def test_owner_learns_when_nobody_was_right(math_task):
    def everyone_wrong(role: str, stage: str, user: str) -> str:
        return "\\boxed{36}"

    agents = {
        "a": make_agent(ScriptedChat(solver), "a"),
        "b": make_agent(ScriptedChat(everyone_wrong), "b", max_attempts=1),
    }
    r = await play_round(
        agents, math_task, 1, agents["a"].environment.sandbox, share=True
    )
    records = DomainRecords(SPECIALISTS)
    records.values["creative"]["math"] = 0.9
    pairs = build_pairs([r], "b", records, blame_adopted=False)
    owners = [role for role, p in pairs.items() if p and role != COORDINATOR]
    assert owners == ["creative"]
    assert pairs["creative"][0].responsibility == "owner"
    assert "42" in pairs["creative"][0].chosen


async def test_code_credit_uses_the_agents_own_probes(add_task):
    right = "def add(a, b):\n    return a + b"
    alt = "def add(a, b):\n    return b + a"

    def team(role: str, stage: str, user: str) -> str:
        if role == "probes":
            return "add(1, 2)\nadd(-3, 3)"
        if role == "coordinator":
            return code_reply(right)
        return code_reply(
            alt if role == "skeptical" else "def add(a, b):\n    return 0"
        )

    agents = {"a": make_agent(ScriptedChat(team), "a")}
    r = await play_round(
        agents, add_task, 1, agents["a"].environment.sandbox, share=True
    )
    assert r.probes["a"] == ["add(1, 2)", "add(-3, 3)"]
    assert r.correct["a"]["1:1:skeptical"] is True
    assert r.correct["a"]["1:1:logical"] is False


def test_records_owner_breaks_ties_by_order():
    records = DomainRecords(SPECIALISTS)
    assert records.owner("math") == "logical"
    records.update("empathetic", "social", 1.0)
    assert records.owner("social") == "empathetic"


def test_proposal_answers_are_extracted():
    p = Proposal(specialist="logical", round=1, response="x", answer="1")
    assert p.answer == "1"


def several_solve_alone(role: str, stage: str, user: str) -> str:
    """Logical, skeptical and empathetic all solve it alone; creative does not."""
    return "\\boxed{36}" if role == "creative" else "\\boxed{42}"


async def _round(math_task, reply):
    agents = {"a": make_agent(ScriptedChat(reply), "a")}
    return await play_round(
        agents, math_task, 1, agents["a"].environment.sandbox, share=True
    )


async def test_winner_take_all_only_best_solo_winner_learns(math_task):
    r = await _round(math_task, several_solve_alone)
    records = DomainRecords(SPECIALISTS)
    records.values["skeptical"]["math"] = 0.8
    pairs = build_pairs([r], "a", records)
    assert [role for role, p in pairs.items() if p and role != COORDINATOR] == [
        "skeptical"
    ]

    everyone = build_pairs([r], "a", DomainRecords(SPECIALISTS), winner_take_all=False)
    assert {role for role, p in everyone.items() if p and role != COORDINATOR} == {
        "logical",
        "skeptical",
        "empathetic",
    }


def test_homeostasis_lets_others_win_after_one_dominates():
    records = DomainRecords(SPECIALISTS, conscience=0.5)
    assert records.owner("math") == "logical"
    for _ in range(30):
        records.record_learners(["logical"])
    assert records.owner("math") != "logical"
    assert records.win_rate["logical"] > 0.7


async def test_surprising_pairs_count_twice(math_task):
    r = await _round(math_task, several_solve_alone)
    records = DomainRecords(SPECIALISTS)
    records.values["empathetic"]["math"] = 0.05  # empathetic rarely succeeds at maths
    for s in ("logical", "skeptical"):
        records.values[s]["math"] = 0.0
    records.win_rate.update({"logical": 0.5, "skeptical": 0.5, "empathetic": 0.0})
    pairs = build_pairs([r], "a", records)
    assert len(pairs["empathetic"]) == 2  # unexpected success: |1 - 0.05| > 0.5
    plain = build_pairs([r], "a", DomainRecords(SPECIALISTS), surprise_threshold=None)
    assert sum(len(p) for role, p in plain.items() if role != COORDINATOR) == 1
