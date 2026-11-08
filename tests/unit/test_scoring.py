"""Majority vote (self-consistency) answerer."""

from src.eval.scoring import majority_answerer
from src.models.core import Message


class FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls = 0

    async def chat(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        reply = self.replies[self.calls % len(self.replies)]
        self.calls += 1
        return reply


async def test_majority_groups_equivalent_math_answers(math_task):
    client = FakeClient(
        ["\\boxed{41}", "\\boxed{42}", "\\boxed{42.0}", "\\boxed{41}", "\\boxed{42}"]
    )
    answer = majority_answerer(client, "m", 64, samples=5)  # type: ignore[arg-type]
    assert await answer(math_task, 0) == "42"
    assert client.calls == 5


async def test_majority_choice_and_empty(choice_task):
    client = FakeClient(["FINAL ANSWER: A", "FINAL ANSWER: B", "FINAL ANSWER: B"])
    assert await majority_answerer(client, "m", 64, samples=3)(choice_task, 0) == "B"  # type: ignore[arg-type]
    silent = FakeClient(["no idea"])
    assert await majority_answerer(silent, "m", 64, samples=3)(choice_task, 0) == ""  # type: ignore[arg-type]
