"""Tests for the embedder utility functions (no Ollama required)."""

from src.memory.embedder import cosine_similarity, is_zero_vector


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == 1.0

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == -1.0

    def test_zero_vector_returns_zero(self):
        zero = [0.0, 0.0, 0.0]
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(zero, v) == 0.0
        assert cosine_similarity(v, zero) == 0.0

    def test_similar_vectors_high_score(self):
        a = [0.9, 0.1, 0.0]
        b = [0.8, 0.2, 0.0]
        assert cosine_similarity(a, b) > 0.95

    def test_symmetry(self):
        a = [0.3, 0.7, 0.1]
        b = [0.6, 0.2, 0.9]
        assert abs(cosine_similarity(a, b) - cosine_similarity(b, a)) < 1e-9


class TestIsZeroVector:
    def test_zero_vector(self):
        assert is_zero_vector([0.0, 0.0, 0.0])

    def test_nonzero_vector(self):
        assert not is_zero_vector([0.0, 0.0, 1.0])

    def test_empty_vector(self):
        assert is_zero_vector([])
