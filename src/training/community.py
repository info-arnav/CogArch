"""Rounds of the community: agents solve, share discoveries, and remember.

Nobody is given the right answer. A solution becomes known only when the
world accepts some agent's answer; that agent then shares it with its peers.
Each agent works out which of its own specialists' proposals were right by
running them against the known solution on example calls it writes itself.
"""

from __future__ import annotations

import asyncio

from src.execution.sandbox import Sandbox, behaves_alike
from src.inference.agent import Agent
from src.models.code import AgentResult, CodeProblem, Discovery, RoundResult


def proposal_key(attempt_number: int, specialist: str) -> str:
    return f"{attempt_number}:{specialist}"


def share_discoveries(
    results: dict[str, AgentResult], share: bool
) -> dict[str, Discovery]:
    """Each agent knows its own accepted solution, or a peer's if sharing is on."""
    known: dict[str, Discovery] = {}
    for name, result in results.items():
        if result.solution is not None:
            known[name] = Discovery(code=result.solution, found_by=name)
    if share:
        for name in results:
            if name in known:
                continue
            for peer, result in results.items():
                if peer != name and result.solution is not None:
                    known[name] = Discovery(code=result.solution, found_by=peer)
                    break
    return known


async def assign_credit(
    agent: Agent,
    result: AgentResult,
    problem: CodeProblem,
    known: Discovery,
    sandbox: Sandbox,
) -> tuple[list[str], dict[str, bool]]:
    """Which of the agent's proposals behave like the known solution."""
    probes = await agent.write_probes(problem)
    if not probes:
        return probes, {}
    reference = await sandbox.run_probes(known.code, probes, problem.setup_code)

    keyed = [
        (proposal_key(d.attempt_number, p.specialist), p.code)
        for d in result.decisions
        for p in d.proposals
    ]
    outputs = await asyncio.gather(
        *(sandbox.run_probes(code, probes, problem.setup_code) for _, code in keyed)
    )
    agreement: dict[str, bool] = {}
    for (key, code), out in zip(keyed, outputs):
        verdict = True if code == known.code else behaves_alike(reference, out)
        if verdict is not None:
            agreement[key] = verdict
    return probes, agreement


def episode_text(round_result: RoundResult, agent: str) -> str:
    """The episode as the agent experienced it — only what it was told or shown."""
    result = round_result.results[agent]
    known = round_result.known_solution.get(agent)
    attempts = len(result.decisions)
    if result.solved:
        outcome = f"Your team's answer was accepted on attempt {attempts}."
    else:
        outcome = f"Your team was told its answer was wrong {attempts} time(s)."
    lines = [f"### Problem\n{round_result.problem.prompt}", outcome]
    if known is not None:
        source = "your team" if known.found_by == agent else f"agent {known.found_by}"
        lines.append(f"A solution found by {source} that was accepted:")
        lines.append(f"```python\n{known.code}\n```")
    return "\n".join(lines)


async def play_round(
    agents: dict[str, Agent],
    problem: CodeProblem,
    cycle: int,
    sandbox: Sandbox,
    share: bool,
) -> RoundResult:
    names = list(agents)
    solved = await asyncio.gather(*(agents[n].solve(problem) for n in names))
    results = dict(zip(names, solved))
    known = share_discoveries(results, share)

    credited = await asyncio.gather(
        *(
            assign_credit(agents[n], results[n], problem, known[n], sandbox)
            for n in names
            if n in known
        )
    )
    with_known = [n for n in names if n in known]
    return RoundResult(
        cycle=cycle,
        problem=problem,
        results=results,
        known_solution=known,
        probes={n: probes for n, (probes, _) in zip(with_known, credited)},
        agreement={n: agreement for n, (_, agreement) in zip(with_known, credited)},
    )


async def remember_round(agent: Agent, round_result: RoundResult) -> None:
    if agent.memory is None:
        return
    result = round_result.results[agent.name]
    known = round_result.known_solution.get(agent.name)
    await agent.memory.record(
        episode_text(round_result, agent.name),
        key=round_result.problem.prompt,
        importance=0.9 if known is not None else 0.4,
        metadata={
            "cycle": round_result.cycle,
            "problem_id": round_result.problem.task_id,
            "solved": result.solved,
            "learned_from": known.found_by if known is not None else None,
        },
    )
    if result.solved:
        status = "accepted"
    elif known is not None:
        status = f"wrong; learned a solution from agent {known.found_by}"
    else:
        status = "wrong"
    agent.memory.note(
        f"{round_result.problem.task_id}: {status}", importance=0.8 if known else 0.4
    )
