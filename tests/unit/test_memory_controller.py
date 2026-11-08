"""AgentMemory: recall by similarity, lessons, cycle episodes."""

from src.memory.controller import AgentMemory


class KeywordEmbedder:
    """Deterministic toy embedder: one dimension per keyword."""

    WORDS = ["sort", "string", "prime", "tree"]

    def embed(self, text: str) -> list[float]:
        return [float(w in text.lower()) for w in self.WORDS]


async def test_recall_similar_episode_and_lessons(tmp_path):
    memory = AgentMemory(tmp_path, KeywordEmbedder(), min_similarity=0.5)
    await memory.record(
        "sorted a list", key="Sort a list", importance=0.9, metadata={"cycle": 1}
    )
    await memory.record(
        "prime sieve", key="Find prime numbers", importance=0.9, metadata={"cycle": 1}
    )
    memory.apply_reflection([("add", None, "Check empty input")], [], set())
    memory.note("mbpp/3: wrong")

    ctx = await memory.recall("Sort these numbers")
    assert [e.content for e in ctx.episodic] == ["sorted a list"]
    text = ctx.to_prompt_text()
    assert "Check empty input" in text and "mbpp/3: wrong" in text


async def test_episodes_from_cycle_and_session(tmp_path):
    memory = AgentMemory(tmp_path, KeywordEmbedder())
    await memory.record("one", key="sort", importance=0.5, metadata={"cycle": 1})
    await memory.record("two", key="tree", importance=0.5, metadata={"cycle": 2})
    assert [e.content for e in memory.episodes_from_cycle(2)] == ["two"]
    memory.note("x")
    memory.end_session()
    assert memory.stats()["working"] == 0


def test_reflection_operations_and_banned_terms(tmp_path):
    memory = AgentMemory(tmp_path, KeywordEmbedder())
    memory.apply_reflection(
        [
            ("add", None, "Guard against empty lists"),
            ("add", None, "remove_Occ needs a loop"),
        ],
        [],
        banned_terms={"remove_Occ"},
    )
    numbered = memory.lessons()
    assert [e.concept for e in numbered] == ["Guard against empty lists"]

    memory.apply_reflection([("agree", 1, "")], numbered, set())
    assert memory.lessons()[0].confidence > numbered[0].confidence

    numbered = memory.lessons()
    memory.apply_reflection(
        [("edit", 1, "Guard against empty inputs")], numbered, set()
    )
    assert memory.lessons()[0].concept == "Guard against empty inputs"

    for _ in range(4):
        memory.apply_reflection([("disagree", 1, "")], memory.lessons(), set())
    assert memory.lessons() == []
