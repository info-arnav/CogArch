"""Agent: specialists propose, coordinator decides, only the decision is judged."""

from tests.conftest import ScriptedChat, code_reply, is_coordinator, make_agent

RIGHT = "def add(a, b):\n    return a + b"
WRONG = "def add(a, b):\n    return a - b"


def team(system: str, user: str, model: str) -> str:
    if is_coordinator(system):
        if "Your team's previous answer" in user:
            return f"BASED ON: logical\n{code_reply(RIGHT)}"
        return f"BASED ON: creative\n{code_reply(WRONG)}"
    if "Logical" in system:
        return code_reply(RIGHT)
    return code_reply(WRONG)


async def test_retries_until_accepted(add_problem):
    agent = make_agent(ScriptedChat(team))
    result = await agent.solve(add_problem)

    assert [d.result.passed for d in result.decisions] == [False, True]
    assert result.solved and result.solution == RIGHT
    first = result.decisions[0]
    assert first.based_on == ["creative"]
    assert first.result.feedback == "Your solution is wrong."
    assert {p.specialist for p in first.proposals} == {
        "logical",
        "creative",
        "skeptical",
        "empathetic",
    }


async def test_only_coordinator_decision_is_judged(add_problem):
    # Logical proposes the right code, but the coordinator always picks wrong code.
    def stubborn(system: str, user: str, model: str) -> str:
        if is_coordinator(system):
            return f"BASED ON: creative\n{code_reply(WRONG)}"
        return team(system, user, model)

    result = await make_agent(ScriptedChat(stubborn), max_attempts=2).solve(add_problem)
    assert not result.solved and len(result.decisions) == 2


async def test_training_prompt_has_no_memory(add_problem):
    class FakeMemory:
        async def recall(self, query):
            class Ctx:
                def to_prompt_text(self):
                    return "## Lessons you have learned\n- something"

            return Ctx()

    chat = ScriptedChat(team)
    result = await make_agent(chat, memory=FakeMemory()).solve(add_problem)
    assert "Lessons" in chat.calls[0]["messages"][-1].content
    assert "Lessons" not in result.decisions[0].coordinator_prompt[-1].content


async def test_answer_is_single_attempt(add_problem):
    chat = ScriptedChat(team)
    code = await make_agent(chat).answer(add_problem)
    assert code == WRONG  # first decision, no retry
    assert len(chat.calls) == 5  # four specialists + coordinator


async def test_seeds_differ_between_agents(add_problem):
    chat_a, chat_b = ScriptedChat(team), ScriptedChat(team)
    agent_a, agent_b = make_agent(chat_a, "a"), make_agent(chat_b, "b")
    agent_b.seed = 2
    await agent_a.answer(add_problem)
    await agent_b.answer(add_problem)
    assert [c["seed"] for c in chat_a.calls] != [c["seed"] for c in chat_b.calls]
