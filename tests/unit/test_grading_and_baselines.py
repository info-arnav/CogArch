"""The grader (xVerify prompt, code extraction) and the baseline methods."""

from src.baselines import debate_answerer, moa_answerer
from src.grading import Grader, reference_text
from src.models.core import Message


class FakeClient:
    def __init__(self, reply) -> None:  # noqa: ANN001
        self.reply = reply
        self.seen: list[list[Message]] = []

    async def chat(
        self, messages, model, temperature, max_tokens, seed, on_delta=None
    ):  # noqa: ANN001, ANN201
        self.seen.append(messages)
        return self.reply(messages) if callable(self.reply) else self.reply


async def test_verifier_prompt_and_labels(choice_task):
    verifier = FakeClient("Correct")
    grader = Grader(verifier, "xverify", None, "")  # type: ignore[arg-type]
    assert await grader.correct(choice_task, "It's sad. FINAL ANSWER: B", "B")
    prompt = verifier.seen[0][0].content
    assert "Correct answer: B. sad" in prompt and "It's sad." in prompt
    assert await grader.correct(choice_task, "It's sad. FINAL ANSWER: B", "B")
    assert len(verifier.seen) == 1
    wrong = Grader(FakeClient("[Incorrect]"), "x", None, "")  # type: ignore[arg-type]
    assert not await wrong.correct(choice_task, "A", "B")
    assert reference_text(choice_task, "C") == "C. bored"


async def test_code_is_sanitised_then_extracted_by_model(add_task):
    grader = Grader(None, "", None, "")
    reply = "```python\ndef add(a, b):\n    return a + b\nprint(add(1, 2))\n```"
    assert await grader.code(add_task, reply) == "def add(a, b):\n    return a + b"
    extractor = FakeClient("```python\ndef add(a, b):\n    return b + a\n```")
    with_model = Grader(None, "", extractor, "base")  # type: ignore[arg-type]
    assert "return b + a" in await with_model.code(add_task, "I would add them.")
    assert with_model.counts["code_extracted"] == 1


async def test_debate_rounds_share_other_agents(math_task):
    client = FakeClient(lambda m: "\\boxed{42}" if len(m) > 1 else "\\boxed{41}")
    answer = debate_answerer(client, "m", 64, agents=3, rounds=3)  # type: ignore[arg-type]
    assert await answer(math_task, 0) == "\\boxed{42}"
    assert len(client.seen) == 9
    assert "One agent solution" in client.seen[-1][-2].content


async def test_moa_aggregates_proposals(math_task):
    client = FakeClient(lambda m: "AGG" if m[0].role == "system" else "\\boxed{7}")
    answer = moa_answerer(client, "m", 64, proposers=8)  # type: ignore[arg-type]
    assert await answer(math_task, 0) == "AGG"
    assert len(client.seen) == 9 and "1. \\boxed{7}" in client.seen[-1][0].content
