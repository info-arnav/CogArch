"""Rounds of the community: agents solve, share discoveries, and remember.

Nobody is given a correct answer. One becomes known only when the world
accepts some agent's answer; that agent shares it with its peers. Each agent
then works out which of its own specialists were right, in each round:
by answer equality for maths and multiple choice, and for code by running
proposals against the known solution on example calls it writes itself.
"""

from __future__ import annotations

import asyncio

from src import calllog
from src.events import NULL_LOG, EventLog
from src.execution.sandbox import Sandbox, behaves_alike
from src.inference.agent import Agent
from src.models.core import AgentResult, Discovery, RoundResult, Task

EPISODE_REASONING_CHARS = 700


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
                response=accepted.submitted,
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
) -> tuple[dict[str, bool], list[str]]:
    """Which specialists' answers, in each round and attempt, match the known one.

    Maths and choice: the grader judges each reply against the accepted answer
    (the task's reference, which the accepted reply was judged to match). Code:
    each proposal is run on the team's own test inputs and compared with the
    accepted code's behaviour; no hidden test is used."""
    proposals = [
        (proposal_key(d.attempt_number, p.round, p.specialist), p)
        for d in result.decisions
        for p in [*d.round1, *d.round2]
    ]
    grader = agent.environment.grader
    if task.kind != "code":
        verdicts = await asyncio.gather(
            *(grader.correct(task, p.response, task.answer) for _, p in proposals)
        )
        return {key: v for (key, _), v in zip(proposals, verdicts)}, []

    probes = result.decisions[0].probes
    if not probes:
        return {}, probes
    reference = await sandbox.run_probes(known.answer, probes, task.setup_code)
    outputs = await asyncio.gather(
        *(sandbox.run_probes(p.answer, probes, task.setup_code) for _, p in proposals)
    )
    correct: dict[str, bool] = {}
    for (key, p), out in zip(proposals, outputs):
        verdict = True if p.answer == known.answer else behaves_alike(reference, out)
        if verdict is not None:
            correct[key] = verdict
    return correct, probes


async def play_round(
    agents: dict[str, Agent],
    task: Task,
    cycle: int,
    sandbox: Sandbox,
    share: bool,
    events: EventLog = NULL_LOG,
) -> RoundResult:
    names = list(agents)
    calllog.update(cycle=cycle, task_id=task.task_id)
    calllog.record("task", task=task.model_dump())
    events.emit("task", cycle=cycle, task=task.model_dump(exclude={"tests"}))
    solved = await asyncio.gather(*(agents[n].solve(task, cycle) for n in names))
    results = dict(zip(names, solved))
    known = share_discoveries(results, share)
    for name in names:
        events.emit(
            "known",
            cycle=cycle,
            task_id=task.task_id,
            agent=name,
            found_by=known[name].found_by if name in known else None,
            answer=known[name].answer if name in known else None,
        )

    with_known = [n for n in names if n in known]
    credited = await asyncio.gather(
        *(
            assign_credit(agents[n], results[n], task, known[n], sandbox)
            for n in with_known
        )
    )
    for n, (correct, probes) in zip(with_known, credited):
        events.emit(
            "credit",
            cycle=cycle,
            task_id=task.task_id,
            agent=n,
            correct=correct,
            probes=probes,
        )
    return RoundResult(
        cycle=cycle,
        task=task,
        results=results,
        known=known,
        correct={n: c for n, (c, _) in zip(with_known, credited)},
        probes={n: p for n, (_, p) in zip(with_known, credited) if p},
    )


def _excerpt(text: str) -> str:
    text = text.strip()
    if len(text) <= EPISODE_REASONING_CHARS:
        return text
    return "… " + text[-EPISODE_REASONING_CHARS:]


def _shown_answer(task: Task, answer: str) -> str:
    return f"```python\n{answer}\n```" if task.kind == "code" else answer


def episode_text(r: RoundResult, agent: str) -> str:
    """The episode as the agent experienced it: only what it was told or shown,
    with the reasoning that failed and the reasoning that worked."""
    result = r.results[agent]
    known = r.known.get(agent)
    attempts = len(result.decisions)
    lines = [f"### Problem\n{r.task.prompt}"]
    wrong = [d for d in result.decisions if not d.result.passed]
    if wrong:
        last = wrong[-1]
        lines.append(
            f"Your team was told it was wrong {len(wrong)} time(s). Its last wrong "
            f"answer: {_shown_answer(r.task, last.answer) or '(no answer given)'}\n"
            f"Reasoning behind it:\n{_excerpt(last.submitted)}"
        )
    if result.solved:
        lines.append(f"Your team's answer was accepted on attempt {attempts}.")
    if known is not None:
        source = "your team" if known.found_by == agent else f"agent {known.found_by}"
        lines.append(
            f"Correct answer (found by {source}): {_shown_answer(r.task, known.answer)}\n"
            f"Reasoning that worked:\n{_excerpt(known.response)}"
        )
    return "\n".join(lines)


async def remember_round(agent: Agent, r: RoundResult) -> None:
    if agent.memory is None:
        return
    result = r.results[agent.name]
    known = r.known.get(agent.name)
    entry = await agent.memory.record(
        episode_text(r, agent.name),
        key=r.task.prompt,
        importance=0.9 if known is not None else 0.4,
        metadata={
            "cycle": r.cycle,
            "task_id": r.task.task_id,
            "domain": r.task.domain,
            "solved": result.solved,
            "attempts": len(result.decisions),
            "learned_from": known.found_by if known is not None else None,
        },
    )
    agent.events.emit(
        "memory_write",
        agent=agent.name,
        cycle=r.cycle,
        task_id=r.task.task_id,
        entry=entry.model_dump(exclude={"embedding"}),
    )
