"""AgentMemory: recall by similarity across subjects, cycle episodes."""

from src.memory.controller import AgentMemory


class KeywordEmbedder:
    """Deterministic toy embedder: one dimension per keyword."""

    WORDS = ["sort", "string", "prime", "tree"]

    def embed(self, text: str) -> list[float]:
        return [float(w in text.lower()) for w in self.WORDS]


async def test_recall_searches_every_subject_and_shows_only_episodes(tmp_path):
    memory = AgentMemory(tmp_path, KeywordEmbedder(), min_similarity={"code": 0.5})
    await memory.record(
        "prime sieve",
        key="Find prime numbers",
        importance=0.9,
        metadata={"cycle": 1, "domain": "code"},
    )
    await memory.record(
        "sorted cards",
        key="Sort cards",
        importance=0.9,
        metadata={"cycle": 1, "domain": "math"},
    )

    ctx = await memory.recall("Sort these numbers", "code")
    assert [e.content for e in ctx.episodic] == ["sorted cards"]
    assert [(c.kept, c.domain) for c in ctx.candidates] == [
        (True, "math"),
        (False, "code"),
    ]
    assert ctx.episodic[0].recall_count == 1
    quiet = await memory.recall("Sort these numbers", "code", touch=False)
    assert quiet.episodic[0].recall_count == 1
    assert ctx.to_prompt_text() == "## Similar problems you have seen\nsorted cards"


async def test_cut_off_is_the_problems_subject(tmp_path):
    memory = AgentMemory(
        tmp_path, KeywordEmbedder(), min_similarity={"code": 0.99, "math": 0.5}
    )
    await memory.record(
        "half match", key="sort string", importance=0.5, metadata={"cycle": 1}
    )
    assert (await memory.recall("sort", "code")).episodic == []
    assert [e.content for e in (await memory.recall("sort", "math")).episodic] == [
        "half match"
    ]


async def test_episodes_from_cycle(tmp_path):
    memory = AgentMemory(tmp_path, KeywordEmbedder())
    await memory.record("one", key="sort", importance=0.5, metadata={"cycle": 1})
    await memory.record("two", key="tree", importance=0.5, metadata={"cycle": 2})
    assert [e.content for e in memory.episodes_from_cycle(2)] == ["two"]
    assert memory.stats() == {"episodic": 2}
