"""Downloads every practice set and benchmark once, before several runs on
the same pod start together and race to write the same cache files."""

import sys

sys.path.insert(0, ".")
from src.tasks import BENCHMARKS, TRAIN_SOURCES

for name, load in {**TRAIN_SOURCES, **BENCHMARKS}.items():
    print(name, len(load()), flush=True)
