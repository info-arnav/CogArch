"""Per-run headline numbers for the run monitor, written to a JSON file:
validation per night, the coordinator against majority vote on problems where
the specialists disagreed, and how often memory was shown.

    python3 scripts/runpod/scoreboard.py /workspace/scoreboard.json
"""

import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
from src import calllog

out = {}
for run in sorted(glob.glob("data/runs/*-r*")):
    name = run.split("/")[-1]
    if name.startswith("check"):
        continue
    validation = []
    for line in open(run + "/events.jsonl"):
        if '"validation_done"' in line[:60]:
            whole = json.loads(line)["whole"]
            teams = [v for v in whole.values() if v]
            validation.append(
                round(
                    sum(100 * sum(v.values()) / len(v) for v in teams) / len(teams), 1
                )
            )
    split: defaultdict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    recall: defaultdict[int, list[int]] = defaultdict(lambda: [0, 0])
    for r in calllog.read(Path(run) / "calls.jsonl.gz"):
        kind = r["kind"]
        if kind == "team_decision" and r.get("phase") == "validate":
            right = r["final_correct"]
            if len(set(right.values())) < 2:
                continue
            d = r["decision"]
            largest = max(d["evidence"]["groups"], key=len)
            s = split[int(r.get("cycle") or 0)]
            s[0] += 1
            s[1] += bool(right.get(d["chosen"]))
            s[2] += bool(right.get(largest[0]))
        elif kind == "recall" and r.get("phase") == "wake":
            c = recall[int(r.get("cycle") or 0)]
            c[0] += 1
            c[1] += bool(r.get("text"))
    out[name] = {
        "validation": validation,
        "coordinator": {
            str(c): {
                "n": s[0],
                "coordinator": round(100 * s[1] / s[0], 1),
                "majority": round(100 * s[2] / s[0], 1),
            }
            for c, s in sorted(split.items())
            if s[0]
        },
        "memory": {
            str(c): round(100 * v[1] / v[0], 1)
            for c, v in sorted(recall.items())
            if v[0]
        },
    }
json.dump({"t": time.time(), "runs": out}, open(sys.argv[1], "w"))
