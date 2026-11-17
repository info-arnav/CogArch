"""Checks every record a run wrote against the design: prompts, replies,
grading, decisions, credit, coordinator pairs, sharing, memory, replay and
adapters. Prints each check with its counts and every violation found.

    python3 scripts/runpod/audit_run.py data/runs/<run>
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
import yaml

from src import calllog

RUN = Path(sys.argv[1])
SPECIALISTS = ["logical", "creative", "skeptical", "empathetic"]
LETTERS = {"logical": "W", "creative": "X", "skeptical": "Y", "empathetic": "Z"}
OPTIONS = {"W", "X", "Y", "Z", "none"}
PERSONAS = {
    n: yaml.safe_load(open(f"prompts/specialists/{n}.yaml")) for n in SPECIALISTS
}
COORD = yaml.safe_load(open("prompts/coordinator.yaml"))
results: list[tuple[str, int, list[str]]] = []


def check(name: str, total: int, bad: list[str]) -> None:
    results.append((name, total, bad))


records = list(calllog.read(RUN / "calls.jsonl.gz"))
models = [r for r in records if r["kind"] == "model"]
kinds = Counter(r["kind"] for r in records)
rounds = [
    json.loads(line)
    for f in sorted((RUN / "rounds").glob("cycle_*.jsonl"))
    for line in open(f)
]
report = json.loads((RUN / "report.json").read_text())
cfg = report["config"]
agents = cfg["agents"]["names"]

bad = [
    f"id {r['id']}: {r.get('error')}"
    for r in models
    if "error" in r or "output" not in r
]
check("every model call has a reply and no error", len(models), bad)

bad = []
for r in models:
    role, req = r.get("role"), r["request"]
    if role in SPECIALISTS:
        sysmsg = req["messages"][0]["content"]
        neutral = (
            r.get("mode") == "specialist_neutral"
            or r.get("step") == "eval_no_persona"
            or cfg["agents"].get("personas") != "full"
        )
        if not neutral and sysmsg.strip() != PERSONAS[role]["system_prompt"].strip():
            bad.append(f"id {r['id']}: {role} system prompt is not its persona")
        if req["temperature"] != PERSONAS[role]["temperature"]:
            bad.append(f"id {r['id']}: {role} temperature {req['temperature']}")
        if "guided_choice" in req:
            bad.append(f"id {r['id']}: specialist reply constrained")
    elif role == "coordinator":
        if r.get("step") != "decide":
            bad.append(f"id {r['id']}: coordinator used for {r.get('step')}")
        if req["messages"][0]["content"].strip() != COORD["system_prompt"].strip():
            bad.append(f"id {r['id']}: coordinator system prompt differs")
        if req["temperature"] != COORD["temperature"]:
            bad.append(f"id {r['id']}: coordinator temperature {req['temperature']}")
        if set(req.get("guided_choice") or []) - OPTIONS or not req.get(
            "guided_choice"
        ):
            bad.append(f"id {r['id']}: options {req.get('guided_choice')}")
        if r.get("output") not in set(req.get("guided_choice") or []):
            bad.append(f"id {r['id']}: reply {r.get('output')!r} not an option")
        if not r.get("logprobs"):
            bad.append(f"id {r['id']}: no probabilities logged")
        user = req["messages"][-1]["content"]
        for n in SPECIALISTS:
            if f"### {n}" in user or n.capitalize() + " specialist" in user:
                bad.append(f"id {r['id']}: name {n} shown to coordinator")
        for leak in ("track record", "(W, X, Y, Z)", "Domain:", "Subject:"):
            if leak in user or leak in req["messages"][0]["content"]:
                bad.append(f"id {r['id']}: '{leak}' shown to coordinator")
    elif role == "helper":
        if r["request"]["model"] != cfg["model"]["served_name"]:
            bad.append(f"id {r['id']}: helper used {r['request']['model']}")
check("prompts, temperatures, constraints and models per role", len(models), bad)

bad = []
for r in models:
    for m in r["request"]["messages"]:
        for leak in ("## Lessons", "## Recent events", "### Problem ("):
            if leak in m["content"]:
                bad.append(f"id {r['id']}: '{leak}' in a prompt")
check("no lesson lists, event lists or subject headers in any prompt", len(models), bad)

bad = []
orders: Counter[tuple[str, ...]] = Counter()
for r in models:
    if r.get("role") == "coordinator":
        user = r["request"]["messages"][-1]["content"]
        order = tuple(
            line[4:5]
            for line in user.splitlines()
            if line.startswith("### ") and len(line.strip()) == 5
        )
        orders[order] += 1
        if sorted(order) != sorted(set(order)) or not set(order) <= {
            "W",
            "X",
            "Y",
            "Z",
        }:
            bad.append(f"id {r['id']}: positions {order}")
check(
    f"coordinator sees letters only, in {len(orders)} different orders",
    sum(orders.values()),
    bad,
)

graded = [r for r in records if r["kind"] == "graded"]
bad = [
    f"id {r['id']}: label {r.get('verifier_label')} but correct={r.get('correct')}"
    for r in graded
    if r.get("verifier_label") in ("correct", "incorrect")
    and (r["verifier_label"] == "correct") != bool(r["correct"])
]
check("grader verdicts agree with their labels", len(graded), bad)

bad = []
n_dec = 0
for r in rounds:
    task = r["task"]
    for agent, res in r["results"].items():
        decisions = res["decisions"]
        if len(decisions) > cfg["environment"]["max_attempts"]:
            bad.append(f"{task['task_id']}/{agent}: {len(decisions)} attempts")
        for i, d in enumerate(decisions):
            n_dec += 1
            if d["labels"] != LETTERS:
                bad.append(f"{task['task_id']}/{agent}: labels {d['labels']}")
            final = d["round2"] or d["round1"]
            chosen = next((p for p in final if p["specialist"] == d["chosen"]), None)
            if chosen is None:
                bad.append(
                    f"{task['task_id']}/{agent}: chosen {d['chosen']} not among positions"
                )
                continue
            if d["submitted"] != chosen["response"] or d["answer"] != chosen["answer"]:
                bad.append(
                    f"{task['task_id']}/{agent}: submitted is not the chosen specialist's reply"
                )
            if (
                d["chosen_by"] == "coordinator"
                and d["coordinator_output"] != LETTERS[d["chosen"]]
            ):
                bad.append(
                    f"{task['task_id']}/{agent}: coordinator said {d['coordinator_output']} but {d['chosen']} chosen"
                )
            if d["result"]["passed"] and i != len(decisions) - 1:
                bad.append(
                    f"{task['task_id']}/{agent}: kept going after a right answer"
                )
            if (
                i > 0
                and "wrong"
                not in d["specialist_prompts"]["logical"][-1]["content"].lower()
            ):
                bad.append(
                    f"{task['task_id']}/{agent}: attempt {i + 1} not told the last answer was wrong"
                )
            if d["rejected_all"] and d["first_choice_output"] != "none":
                bad.append(f"{task['task_id']}/{agent}: rejected without replying none")
check("decisions: chosen, submitted, attempts and retries", n_dec, bad)

bad = []
n_known = 0
for r in rounds:
    for agent in r["results"]:
        k = r.get("known", {}).get(agent)
        if not k:
            continue
        n_known += 1
        solved = any(d["result"]["passed"] for d in r["results"][agent]["decisions"])
        if solved and k["found_by"] != agent:
            bad.append(
                f"{r['task']['task_id']}/{agent}: solved itself but credited to {k['found_by']}"
            )
        if not solved and not cfg["agents"]["share_discoveries"]:
            bad.append(
                f"{r['task']['task_id']}/{agent}: learned a shared answer with sharing off"
            )
check("known answers: own when solved, shared only when sharing is on", n_known, bad)

bad = []
n_pairs = 0
correct_of = {}
decision_of = {}
for r in rounds:
    for agent, res in r["results"].items():
        corr = r.get("correct", {}).get(agent, {})
        for d in res["decisions"]:
            for p in d["round1"] + d["round2"]:
                correct_of[(agent, r["task"]["task_id"], p["response"])] = corr.get(
                    f"{d['attempt_number']}:{p['round']}:{p['specialist']}"
                )
            decision_of[
                (agent, r["task"]["task_id"], json.dumps(d["coordinator_prompt"]))
            ] = (d, corr)
for agent in agents:
    for role in SPECIALISTS + ["coordinator"]:
        for f in sorted((RUN / "adapters" / agent / role).glob("cycle_*_new.jsonl")):
            for line in open(f):
                row = json.loads(line)
                n_pairs += 1
                chosen, rejected = (
                    row["chosen"][0]["content"],
                    row["rejected"][0]["content"],
                )
                if chosen == rejected:
                    bad.append(f"{agent}/{role} {f.name}: chosen equals rejected")
                if role == "coordinator":
                    if chosen not in OPTIONS or rejected not in OPTIONS:
                        bad.append(
                            f"{agent}/coordinator {f.name}: pair {chosen!r} > {rejected!r}"
                        )
                else:
                    task_id = next(
                        (
                            t
                            for (a, t, resp) in correct_of
                            if a == agent and resp == rejected
                        ),
                        None,
                    )
                    if task_id and correct_of.get((agent, task_id, rejected)) is True:
                        bad.append(
                            f"{agent}/{role} {f.name}: rejected reply was graded correct"
                        )
check(
    "training pairs: never identical, coordinator pairs single letters, rejected replies wrong",
    n_pairs,
    bad,
)

bad = []
for agent in agents:
    for role in SPECIALISTS + ["coordinator"]:
        folder = RUN / "adapters" / agent / role
        for c in range(1, 20):
            new, full = (
                folder / f"cycle_{c}_new.jsonl",
                folder / f"cycle_{c}_pairs.jsonl",
            )
            if not full.exists():
                continue
            new_rows = {line for line in open(new)} if new.exists() else set()
            earlier = set()
            for e in range(1, c):
                p = folder / f"cycle_{e}_new.jsonl"
                if p.exists():
                    earlier |= {line for line in open(p)}
            replayed = [line for line in open(full) if line not in new_rows]
            stray = [line for line in replayed if line not in earlier]
            if stray:
                bad.append(
                    f"{agent}/{role} cycle {c}: {len(stray)} replayed rows not from earlier cycles"
                )
            if (
                c > 1
                and earlier
                and not replayed
                and cfg["training"].get("replay_ratio", 0) > 0
            ):
                bad.append(f"{agent}/{role} cycle {c}: no replay")
check("replay comes only from earlier nights", 1, bad)

loads = [r for r in records if r["kind"] == "adapter_load"]
bad = []
for r in rounds:
    cyc = r["cycle"]
    for agent, res in r["results"].items():
        for d in res["decisions"]:
            for call in d["calls"]:
                expected_base = cyc == 1
                is_base = call["model"] == cfg["model"]["served_name"]
                if (
                    expected_base
                    and not is_base
                    and call["role"] in SPECIALISTS + ["coordinator"]
                ):
                    bad.append(f"cycle 1 {agent}/{call['role']} used {call['model']}")
                if (
                    not expected_base
                    and is_base
                    and call["role"] in SPECIALISTS + ["coordinator"]
                ):
                    trained = (
                        RUN / "adapters" / agent / call["role"] / f"cycle_{cyc - 1}"
                    ).exists()
                    if trained:
                        bad.append(
                            f"cycle {cyc} {agent}/{call['role']} still on the base model after training"
                        )
check(
    f"trained adapters used the next day ({len(loads)} loads)",
    sum(len(res["decisions"]) for r in rounds for res in r["results"].values()),
    bad,
)

bad = []
mem_rows = 0
for agent in agents:
    path = RUN / "memory" / agent / "episodic.jsonl"
    if not path.exists():
        continue
    for line in open(path):
        mem_rows += 1
        episode = json.loads(line)
        text = episode.get("content", "")
        if "### Problem (" in text or "Domain:" in text:
            bad.append(
                f"{agent}: episode {episode.get('id', '?')} carries a subject label"
            )
expected = sum(len(r["results"]) for r in rounds)
if cfg["memory"]["enabled"] and mem_rows != expected:
    bad.append(f"{mem_rows} episodes stored for {expected} problem-team pairs")
snaps = (
    sorted(p.name for p in (RUN / "memory_snapshots").glob("cycle_*"))
    if (RUN / "memory_snapshots").exists()
    else []
)
nights = {
    int(d.name.split("_")[1])
    for d in (RUN / "adapters").glob("*/*/cycle_*")
    if d.is_dir()
}
if cfg["memory"]["enabled"] and len(snaps) < len(nights) - 1:
    bad.append(f"snapshots {snaps}")
check(
    "memory: one episode per problem and team, no subject labels, a snapshot each night",
    mem_rows,
    bad,
)

recalls = [r for r in records if r["kind"] == "recall"]
bad = []
for r in recalls:
    floor = r.get("min_similarity") or 0
    above = [c for c in r.get("candidates") or [] if c.get("similarity", 0) >= floor]
    if r.get("text") and not above:
        bad.append(
            f"id {r['id']}: memory shown though no candidate reached the cut-off {floor}"
        )
    if above and not r.get("text"):
        bad.append(
            f"id {r['id']}: a candidate reached the cut-off but nothing was shown"
        )
coord_mem = [
    r
    for r in models
    if r.get("role") == "coordinator"
    and "Similar problems" in r["request"]["messages"][-1]["content"]
]
bad += [f"id {r['id']}: memory shown to the coordinator" for r in coord_mem]
hits = sum(1 for r in recalls if r.get("text"))
check(
    f"memory recall: cut-offs applied, never shown to the coordinator ({hits} of {len(recalls)} recalls showed a memory)",
    len(recalls),
    bad,
)

val = [r for r in records if r["kind"] == "validated" and r.get("phase") == "validate"]
bad = []
for r in models:
    if r.get("phase") in ("validate", "benchmarks") and r.get("role") in [
        *SPECIALISTS,
        "coordinator",
    ]:
        user = r["request"]["messages"][-1]["content"]
        if "Your team's previous answer" in user or "rejected every answer" in user:
            bad.append(
                f"id {r['id']}: told a previous answer was wrong during {r['phase']}"
            )
evals = [
    r
    for r in records
    if r["kind"] == "team_decision" and r.get("phase") in ("validate", "benchmarks")
]
writes_during_eval = [
    r
    for r in recalls
    if r.get("phase") in ("validate", "benchmarks") and r.get("touched")
]
bad += [f"id {r['id']}: memory written during {r['phase']}" for r in writes_during_eval]
check("validation and benchmarks: one attempt, no feedback", len(val), bad)

print(f"Audit of {RUN.name}: {len(records)} records ({dict(kinds)})")
failed = 0
for name, total, problems in results:
    status = "PASS" if not problems else "FAIL"
    failed += bool(problems)
    print(f"{status} {name} [{total} checked]")
    for p in problems[:8]:
        print(f"      {p}")
    if len(problems) > 8:
        print(f"      ... and {len(problems) - 8} more")
print(f"{len(results) - failed} of {len(results)} checks passed")
