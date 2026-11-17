"""Agent: answer alone, evidence, discuss only on disagreement, the coordinator
chooses a position; only the submitted reply is judged."""

from src.memory.models import EpisodicEntry, MemoryContext
from tests.conftest import ScriptedChat, code_reply, make_agent

RIGHT = "def add(a, b):\n    return a + b"
WRONG = "def add(a, b):\n    return a - b"


def math_team(role: str, stage: str, user: str) -> str:
    """Round 1: only skeptical is right. After discussion, logical joins it.
    The coordinator chooses skeptical."""
    if role == "coordinator":
        return "Skeptical's reasoning holds up.\nFINAL CHOICE: Y"
    if role == "skeptical" or (role == "logical" and stage == "round2"):
        return "6*7 = \\boxed{42}"
    return "6*7 = \\boxed{36}"


async def test_disagreement_is_discussed_and_a_position_chosen(math_task):
    chat = ScriptedChat(math_team)
    result = await make_agent(chat, discussion="disagreement").solve(math_task)
    d = result.decisions[0]
    assert result.solved and d.answer == "42"
    assert d.chosen == "skeptical" and d.chosen_by == "coordinator" and d.discussed
    assert d.submitted == "6*7 = \\boxed{42}"
    assert [c.stage for c in d.calls] == ["round1"] * 4 + ["round2"] * 4 + ["decide"]
    assert {p.specialist: p.answer for p in d.round2}["logical"] == "42"
    assert sorted(map(sorted, d.evidence.groups)) == [
        ["creative", "empathetic"],
        ["logical", "skeptical"],
    ]


async def test_coordinator_decides_even_when_all_agree(math_task):
    def team(role: str, stage: str, user: str) -> str:
        return "FINAL CHOICE: X" if role == "coordinator" else "\\boxed{42}"

    d = (
        await make_agent(ScriptedChat(team), discussion="disagreement").solve(math_task)
    ).decisions[0]
    assert [c.stage for c in d.calls] == ["round1"] * 4 + ["decide"]
    assert d.chosen == "creative" and d.chosen_by == "coordinator" and not d.discussed


async def test_coordinator_is_never_told_the_subject_or_a_track_record(math_task):
    d = (await make_agent(ScriptedChat(math_team)).solve(math_task)).decisions[0]
    text = d.coordinator_prompt[-1].content
    assert "math" not in text.lower() and "track record" not in text.lower()
    assert "### W" in text and "logical" not in text.lower()
    assert d.labels == {
        "logical": "W",
        "creative": "X",
        "skeptical": "Y",
        "empathetic": "Z",
    }


async def test_discussion_shows_teammates_and_evidence(math_task):
    chat = ScriptedChat(math_team)
    await make_agent(chat).solve(math_task)
    logical_r2 = next(
        c for c in chat.calls if c["role"] == "logical" and c["stage"] == "round2"
    )
    text = logical_r2["messages"][-1].content
    assert "### creative" in text and "### skeptical" in text
    assert "### logical" not in text and "Checked results" in text
    assert "AGREE <name>" in logical_r2["messages"][0].content


async def test_unreadable_choice_falls_back_to_largest_group(math_task):
    def team(role: str, stage: str, user: str) -> str:
        if role == "coordinator":
            return "Hard to say."
        return (
            "\\boxed{42}"
            if role in ("logical", "creative", "skeptical")
            else "\\boxed{1}"
        )

    d = (
        await make_agent(ScriptedChat(team), discussion="never").solve(math_task)
    ).decisions[0]
    assert d.chosen_by == "fallback" and d.answer == "42"


async def test_code_evidence_and_retry_after_wrong(add_task):
    def team(role: str, stage: str, user: str) -> str:
        if role == "probes":
            return "add(1, 2)\nadd(0, 0)"
        if role == "coordinator":
            return "FINAL CHOICE: W" if "previous answer" in user else "FINAL CHOICE: X"
        if role == "logical" and "previous answer" in user:
            return code_reply(RIGHT + "\nprint(add(1, 2))")
        return (
            code_reply(WRONG)
            if role == "creative"
            else code_reply(RIGHT if "previous answer" in user else WRONG + "  ")
        )

    result = await make_agent(ScriptedChat(team), discussion="never").solve(add_task)
    first, second = result.decisions[0], result.decisions[1]
    assert (
        not first.result.passed and first.result.feedback == "Your solution is wrong."
    )
    assert second.result.passed and second.answer == RIGHT
    assert first.probes == ["add(1, 2)", "add(0, 0)"]


async def test_multiple_choice_gets_one_attempt(choice_task):
    team = lambda role, stage, user: "FINAL ANSWER: A"  # noqa: E731
    result = await make_agent(ScriptedChat(team)).solve(choice_task)
    assert len(result.decisions) == 1 and not result.solved


async def test_training_prompts_have_no_memory(math_task):
    class FakeMemory:
        async def recall(self, query, domain=None, touch=True):
            return MemoryContext(episodic=[EpisodicEntry(content="something")])

    chat = ScriptedChat(math_team)
    d = (await make_agent(chat, memory=FakeMemory()).solve(math_task)).decisions[0]
    assert "Similar problems" in chat.calls[0]["messages"][-1].content
    assert "Similar problems" not in d.coordinator_prompt[-1].content
    assert all(
        "Similar problems" not in m[-1].content for m in d.specialist_prompts.values()
    )


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


async def test_solo_answer_returns_that_specialists_reply(math_task):
    chat = ScriptedChat(math_team)
    agent = make_agent(chat)
    assert await agent.solo_answer("skeptical", math_task) == "6*7 = \\boxed{42}"
    assert {c["role"] for c in chat.calls} == {"skeptical"}


async def test_coordinator_can_send_the_team_back_once(math_task):
    def team(role: str, stage: str, user: str) -> str:
        if role == "coordinator":
            if "none." in user and "try again" in user:
                return "All wrong.\nFINAL CHOICE: none"
            return "FINAL CHOICE: Y"
        if "rejected every answer" in user:
            return "\\boxed{42}" if role == "skeptical" else "\\boxed{40}"
        return "\\boxed{36}" if role != "logical" else "\\boxed{35}"

    d = (
        await make_agent(ScriptedChat(team), discussion="never").solve(math_task)
    ).decisions[0]
    assert d.rejected_all and not d.could_reject
    assert "FINAL CHOICE: none" in d.first_choice_output
    assert {p.specialist: p.answer for p in d.round1}["skeptical"] == "36"
    assert {p.specialist: p.answer for p in d.round2}["skeptical"] == "42"
    assert d.chosen == "skeptical" and d.answer == "42"
    assert ", none." not in d.coordinator_prompt[-1].content


async def test_coordinator_naming_an_answer_selects_who_gave_it(choice_task):
    def team(role: str, stage: str, user: str) -> str:
        if role == "coordinator":
            return "FINAL CHOICE: B"
        return "FINAL ANSWER: B" if role == "skeptical" else "FINAL ANSWER: A"

    d = (await make_agent(ScriptedChat(team)).solve(choice_task)).decisions[0]
    assert d.chosen == "skeptical" and d.chosen_by == "coordinator_answer"


async def test_unnamed_choice_falls_back_to_the_largest_group(choice_task):
    def team(role: str, stage: str, user: str) -> str:
        if role == "coordinator":
            return "I am not sure."
        return "FINAL ANSWER: B" if role == "skeptical" else "FINAL ANSWER: A"

    d = (await make_agent(ScriptedChat(team)).solve(choice_task)).decisions[0]
    assert d.chosen != "skeptical" and d.chosen_by == "fallback"
