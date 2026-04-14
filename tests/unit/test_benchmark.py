"""Unit tests for JsonlBenchmark loader."""

import asyncio
import json

from src.eval.benchmarks.jsonl_benchmark import JsonlBenchmark


class TestJsonlBenchmark:
    def test_load_items(self, tmp_path: object) -> None:
        import pathlib

        path = pathlib.Path(str(tmp_path)) / "test.jsonl"
        items = [
            {"question": "What is 2+2?", "expected_answer": "4", "category": "math"},
            {"question": "Capital of France?", "expected_answer": "Paris"},
        ]
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        bench = JsonlBenchmark(path=path)
        loaded = asyncio.run(bench.load())
        assert len(loaded) == 2
        assert loaded[0].question == "What is 2+2?"
        assert loaded[1].expected_answer == "Paris"

    def test_load_missing_file(self, tmp_path: object) -> None:
        import pathlib

        path = pathlib.Path(str(tmp_path)) / "missing.jsonl"
        bench = JsonlBenchmark(path=path)
        loaded = asyncio.run(bench.load())
        assert loaded == []

    def test_score_exact_match(self) -> None:
        bench = JsonlBenchmark(path="x", metric="exact_match")
        assert bench.score("Paris", "Paris") == 1.0
        assert bench.score("London", "Paris") == 0.0

    def test_score_fuzzy_match(self) -> None:
        bench = JsonlBenchmark(path="x", metric="fuzzy_match")
        assert bench.score("Paris", "Paris") == 1.0

    def test_load_sample_benchmark(self) -> None:
        bench = JsonlBenchmark(path="data/benchmarks/sample.jsonl")
        loaded = asyncio.run(bench.load())
        assert len(loaded) == 3
        assert loaded[0].category == "geography"
