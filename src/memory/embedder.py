"""Embedding via Ollama API (nomic-embed-text).

Falls back to a zero vector if Ollama is unreachable — the rest of the
memory system degrades gracefully (retrieval returns recency-ranked results
instead of similarity-ranked).
"""

from __future__ import annotations

import math

import httpx

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768  # nomic-embed-text output dimension


async def embed(text: str, base_url: str = "http://localhost:11434") -> list[float]:
    """Return an embedding vector. Returns zero vector on failure."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{base_url}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
            )
            resp.raise_for_status()
            return resp.json()["embedding"]  # type: ignore[no-any-return]
    except Exception:
        return [0.0] * EMBED_DIM


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [–1, 1]. Returns 0 if either vector is zero."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def is_zero_vector(v: list[float]) -> bool:
    return all(x == 0.0 for x in v)
