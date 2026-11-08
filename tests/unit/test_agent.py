"""Agent: round 1, discussion, decision; only the decision is judged."""

from tests.conftest import ScriptedChat, code_reply, make_agent

RIGHT = "def add(a, b):\n    return a + b"
WRONG = "def add(a, b):\n    return a - b"


def math_team(role: str, stage: str, user: str) -> str:
    """Round 1: only skeptical is right. After discussion, logical joins it.
    The coordinator sides with whoever argues 42."""
    if role == "coordinator":
        return "Skeptical's reasoning holds up. \\boxed{42}"
    if role == "skeptical" or (role == "logical" and stage == "round2"):
        return "6*7 = \\boxed{42}"
    return "6*7 = \\boxed{36}"


async def test_three_stages_and_verdict(math_task):
    chat = ScriptedChat(math_team)
    result = await make_agent(chat).solve(math_task)
    d = result.decisions[0]
    assert result.solved and d.answer == "42"
    assert [c.stage for c in d.calls] == ["round1"] * 4 + ["round2"] * 4 + ["decide"]
    assert {p.specialist: p.answer for p in d.round1}["logical"] == "36"
    assert {p.specialist: p.answer for p in d.round2}["logical"] == "42"


async def test_discussion_shows_teammates_but_not_self(math_task):
    chat = ScriptedChat(math_team)
    await make_agent(chat).solve(math_task)
    logical_r2 = next(
        c for c in chat.calls if c["role"] == "logical" and c["stage"] == "round2"
    )
    text = logical_r2["messages"][-1].content
    assert (
        "### creative" in text and "### skeptical" in text and "### logical" not in text
    )


async def test_without_discussion_round2_copies_round1(math_task):
    chat = ScriptedChat(math_team)
    result = await make_agent(chat, discussion=False).solve(math_task)
    d = result.decisions[0]
    assert [c.stage for c in d.calls] == ["round1"] * 4 + ["decide"]
    assert [p.answer for p in d.round2] == [p.answer for p in d.round1]


async def test_code_retries_after_wrong(add_task):
    def team(role: str, stage: str, user: str) -> str:
        if role == "coordinator":
            return code_reply(RIGHT if "previous answer" in user else WRONG)
        return code_reply(WRONG)

    result = await make_agent(ScriptedChat(team)).solve(add_task)
    assert [d.result.passed for d in result.decisions] == [False, True]
    assert result.decisions[0].result.feedback == "Your solution is wrong."


async def test_multiple_choice_gets_one_attempt(choice_task):
    team = lambda role, stage, user: "FINAL ANSWER: A"  # noqa: E731
    result = await make_agent(ScriptedChat(team)).solve(choice_task)
    assert len(result.decisions) == 1 and not result.solved


async def test_training_prompts_have_no_memory(math_task):
    class FakeMemory:
        async def recall(self, query):
            class Ctx:
                def to_prompt_text(self):
                    return "## Lessons you have learned\n- something"

            return Ctx()

    chat = ScriptedChat(math_team)
    d = (await make_agent(chat, memory=FakeMemory()).solve(math_task)).decisions[0]
    assert "Lessons" in chat.calls[0]["messages"][-1].content
    assert "Lessons" not in d.coordinator_prompt[-1].content
    assert all("Lessons" not in m[-1].content for m in d.specialist_prompts.values())


async def test_positions_shown_to_coordinator_are_shuffled(math_task):
    orders = set()
    for i in range(6):
        task = math_task.model_copy(update={"task_id": f"t{i}"})
        chat = ScriptedChat(math_team)
        d = (await make_agent(chat).solve(task)).decisions[0]
        text = d.calls[-1].messages[-1].content
        orders.add(
            tuple(line[4:] for line in text.splitlines() if line.startswith("### "))
        )
    assert len(orders) > 1


async def test_solo_answer_uses_only_that_specialist(math_task):
    chat = ScriptedChat(math_team)
    agent = make_agent(chat)
    assert await agent.solo_answer("skeptical", math_task) == "42"
    assert await agent.solo_answer("creative", math_task) == "36"
    assert {c["role"] for c in chat.calls} == {"skeptical", "creative"}
