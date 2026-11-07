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
    memory.add_lessons(["Check empty input"])
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
