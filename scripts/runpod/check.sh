#!/usr/bin/env bash
# Quick end-to-end check on the last GPU before the full run: 2 short cycles (20
# problems each), sleep with training and replay, then small benchmarks.
# Prints whether each part of the design actually ran.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=/workspace/hf OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
LOGS=/workspace/logs; mkdir -p "$LOGS"
rm -rf data/runs/check-r1
G=$(( $(nvidia-smi -L | wc -l) - 1 ))
CUDA_VISIBLE_DEVICES=$G INFERENCE_BASE_URL=http://localhost:$((8200 + G))/v1 VERIFIER_URL=${GRADER_URL:-http://localhost:8300/v1} \
TRAINING_SLOTS_DIR=/workspace/slots/check \
  python3 -m cli.main run config/pilot.yaml > "$LOGS/check.log" 2>&1
python3 - <<'PY'
import json, os, sys
from collections import Counter
sys.path.insert(0, ".")
from src import calllog
calls = list(calllog.read("data/runs/check-r1/calls.jsonl.gz"))
ck = Counter(r["kind"] for r in calls)
models = [r for r in calls if r["kind"] == "model"]
decided = [r for r in calls if r["kind"] == "team_decision"]
coord = [r for r in models if r.get("role") == "coordinator"]
ev = [json.loads(l) for l in open("data/runs/check-r1/events.jsonl")]
kinds = Counter(e["type"] for e in ev)
decisions = [e for e in ev if e["type"] == "decision"]
by = Counter(d["chosen_by"] for d in decisions)
report = json.load(open("data/runs/check-r1/report.json"))
replayed = report["cycles"][-1].get("replayed", {})
checks = {
    "every answer went through the coordinator": by.get("coordinator", 0) + by.get("fallback", 0) == len(decisions) > 0,
    "coordinator replied with a valid letter every time": by.get("coordinator", 0) == len(decisions),
    "coordinator never told the subject or a track record": coord and not any("track record" in r["request"]["messages"][-1]["content"].lower() for r in coord),
    "memory stored episodes": kinds["memory_write"] > 0,
    "adapters trained and loaded": kinds["adapter"] > 0,
    "coordinator adapter trained": any(e["type"] == "adapter" and e["role"] == "coordinator" for e in ev),
    "replay used in cycle 2": any(n > 0 for roles in replayed.values() for n in roles.values()),
    "benchmarks scored": kinds["benchmark"] > 0,
    "coordinator saw letters, never specialist names": coord and not any(f"### {n}" in r["request"]["messages"][-1]["content"] or f"- {n}:" in r["request"]["messages"][-1]["content"] for r in coord for n in ("logical", "creative", "skeptical", "empathetic")) and all("### W" in r["request"]["messages"][-1]["content"] for r in coord if r.get("step") == "decide"),
    "no subject label, lesson list or event list in any prompt": models and not any(any(t in r["request"]["messages"][-1]["content"] for t in ("## Lessons", "## Recent events", "### Problem (")) for r in models),
    "memory recalls logged": ck["recall"] > 0,
    "persona-free specialist benchmarks ran": any(e["type"] == "benchmark" and e["mode"] == "specialist_neutral" for e in ev),
    "every model call logged with input and output": models and all(r["request"]["messages"] and ("output" in r or "error" in r) for r in models),
    "no model call failed": not any("error" in r for r in models),
    "model calls tagged by phase": {r.get("phase") for r in models} >= {"validate", "wake", "reflect", "benchmarks"},
    "grader verdicts logged": ck["graded"] > 0,
    "code executions logged": ck["execution"] > 0,
    "team decisions logged with every specialist graded": decided and all(len(r["first_correct"]) == 4 for r in decided),
    "coordinator replies constrained to the options, with probabilities logged": coord and all(r["request"].get("guided_choice") and r.get("logprobs") for r in coord if r.get("step") == "decide"),
    "coordinator pairs are single letters": all(json.loads(l)["chosen"][0]["content"] in {"W", "X", "Y", "Z", "none"} and json.loads(l)["rejected"][0]["content"] in {"W", "X", "Y", "Z", "none"} for a in ("a", "b") for c in (1, 2) if os.path.exists(f"data/runs/check-r1/adapters/{a}/coordinator/cycle_{c}_new.jsonl") for l in open(f"data/runs/check-r1/adapters/{a}/coordinator/cycle_{c}_new.jsonl") if l.strip()),
    "memory snapshots saved each cycle": all(os.path.isdir(f"data/runs/check-r1/memory_snapshots/cycle_{c}") for c in (1, 2)),
}
for name, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + name)
print("decisions:", dict(by), "| try-again:", sum(d["rejected_all"] for d in decisions))
size = os.path.getsize("data/runs/check-r1/calls.jsonl.gz") / 1e6
print(f"call log: {len(calls)} records, {len(models)} model calls, {size:.1f} MB")
PY
