"""Download HumanEval and MBPP to local JSONL cache. Run once before experiments."""

from src.eval.benchmarks import humaneval, mbpp

print("Downloading HumanEval...")
problems = humaneval.load()
print(f"  {len(problems)} problems cached → data/benchmarks/humaneval.jsonl")

print("Downloading MBPP...")
train = mbpp.load(split="train")
test = mbpp.load(split="test")
print(
    f"  {len(train)} train + {len(test)} test problems cached → data/benchmarks/mbpp.jsonl"
)

print("Done.")
