"""Rounds of the community: agents solve, share discoveries, and remember.

Nobody is given a correct answer. One becomes known only when the world
accepts some agent's answer; that agent shares it with its peers. Each agent
then works out which of its own specialists were right, in each round:
by answer equality for maths and multiple choice, and for code by running
proposals against the known solution on example calls it writes itself.
"""

from __future__ import annotations

import asyncio

from src.environment import answers_match
from src.execution.sandbox import Sandbox, behaves_alike
from src.inference.agent import Agent
from src.models.core import AgentResult, Discovery, LLMCall, RoundResult, Task


def proposal_key(attempt: int, round_: int, specialist: str) -> str:
    return f"{attempt}:{round_}:{specialist}"


def share_discoveries(
    results: dict[str, AgentResult], share: bool
) -> dict[str, Discovery]:
    """Each agent knows its own accepted answer, or a peer's if sharing is on."""
    known: dict[str, Discovery] = {}
    for name, result in results.items():
        accepted = result.solution
        if accepted is not None:
            known[name] = Discovery(
                answer=accepted.answer,
                response=accepted.coordinator_output,
                found_by=name,
            )
    if share:
        for name in results:
            if name in known:
                continue
            for peer in results:
                if peer != name and peer in known and known[peer].found_by == peer:
                    known[name] = known[peer].model_copy()
                    break
    return known


async def assign_credit(
    agent: Agent,
    result: AgentResult,
    task: Task,
    known: Discovery,
    sandbox: Sandbox,
) -> tuple[dict[str, bool], list[str], LLMCall | None]:
    """Which specialists' answers, in each round and attempt, match the known one."""
    proposals = [
        (proposal_key(d.attempt_number, p.round, p.specialist), p.answer)
        for d in result.decisions
        for p in [*d.round1, *d.round2]
    ]
    if task.kind != "code":
        matches = {
            key: answers_match(task, known.answer, ans) for key, ans in proposals
        }
        return matches, [], None

    probes, call = await agent.write_probes(task)
    if not probes:
        return {}, probes, call
    reference = await sandbox.run_probes(known.answer, probes, task.setup_code)
    outputs = await asyncio.gather(
        *(sandbox.run_probes(ans, probes, task.setup_code) for _, ans in proposals)
    )
    correct: dict[str, bool] = {}
    for (key, ans), out in zip(proposals, outputs):
        verdict = True if ans == known.answer else behaves_alike(reference, out)
        if verdict is not None:
            correct[key] = verdict
    return correct, probes, call


async def play_round(
    agents: dict[str, Agent], task: Task, cycle: int, sandbox: Sandbox, share: bool
) -> RoundResult:
    names = list(agents)
    solved = await asyncio.gather(*(agents[n].solve(task) for n in names))
    results = dict(zip(names, solved))
    known = share_discoveries(results, share)

    with_known = [n for n in names if n in known]
    credited = await asyncio.gather(
        *(
            assign_credit(agents[n], results[n], task, known[n], sandbox)
            for n in with_known
        )
    )
    return RoundResult(
        cycle=cycle,
        task=task,
        results=results,
        known=known,
        correct={n: c for n, (c, _, _) in zip(with_known, credited)},
        probes={n: p for n, (_, p, _) in zip(with_known, credited) if p},
        probe_calls={n: call for n, (_, _, call) in zip(with_known, credited) if call},
    )


def episode_text(r: RoundResult, agent: str) -> str:
    """The episode as the agent experienced it: only what it was told or shown."""
    result = r.results[agent]
    known = r.known.get(agent)
    attempts = len(result.decisions)
    outcome = (
        f"Your team's answer was accepted on attempt {attempts}."
        if result.solved
        else f"Your team was told its answer was wrong {attempts} time(s)."
    )
    lines = [f"### Problem ({r.task.domain})\n{r.task.prompt}", outcome]
    if known is not None:
        source = "your team" if known.found_by == agent else f"agent {known.found_by}"
        lines.append(f"A correct answer found by {source}:")
        lines.append(
            f"```python\n{known.answer}\n```" if r.task.kind == "code" else known.answer
        )
    return "\n".join(lines)


async def remember_round(agent: Agent, r: RoundResult) -> None:
    if agent.memory is None:
        return
    result = r.results[agent.name]
    known = r.known.get(agent.name)
    await agent.memory.record(
        episode_text(r, agent.name),
        key=r.task.prompt,
        importance=0.9 if known is not None else 0.4,
        metadata={
            "cycle": r.cycle,
            "task_id": r.task.task_id,
            "domain": r.task.domain,
            "solved": result.solved,
            "learned_from": known.found_by if known is not None else None,
        },
    )
    if result.solved:
        status = "accepted"
    elif known is not None:
        status = f"wrong; learned the answer from agent {known.found_by}"
    else:
        status = "wrong"
    agent.memory.note(
        f"{r.task.domain} problem {r.task.task_id}: {status}",
        importance=0.8 if known else 0.4,
    )
