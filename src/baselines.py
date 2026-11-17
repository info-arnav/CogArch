"""Published single- and multi-agent methods on the same model, graded the same
way as the team, for comparison at matched compute.

  single        one reply, greedy, the benchmark's standard prompt (EvalPlus's
                chat protocol for code); chain-of-thought for maths and choice
  majority      self-consistency (Wang et al., 2023): k samples, the most common
                answer wins
  debate        LLM-Debate (Du et al., 2023): n agents answer, then each revises
                after reading the others, for r rounds; majority of the last round
  moa           Mixture-of-Agents (Wang et al., 2025): k proposers, one aggregator
                that synthesises a final reply

Defaults use 9 model calls per problem, the same as one attempt of the team
(4 alone + 4 discussion + 1 choice). Token use is recorded per method, since
MASLab compares methods by accuracy against tokens per problem.

    cogarch baselines config/general.yaml --method debate
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.console import Console

from src.config import Config
from src.eval import scoring
from src.eval.scoring import Answerer, majority_answerer, plain_answerer
from src.grading import build_grader
from src.inference.client import InferenceClient
from src.inference.prompts import CODE_INSTRUCTION, task_text
from src.models.core import Message, Task
from src.tasks import BENCHMARKS, EVALPLUS_BENCHMARKS, benchmark_subset

SAMPLING_TEMPERATURE = 0.7

DEBATE_UPDATE = (
    "These are the solutions to the problem from other agents:\n\n{others}\n\n"
    "Using the reasoning from other agents as additional advice, can you give an "
    "updated answer? Examine your solution and that of other agents step by step. "
    "The original problem is:\n\n{problem}"
)

MOA_AGGREGATE = (
    "You have been provided with a set of responses from various open-source models "
    "to the latest user query. Your task is to synthesize these responses into a "
    "single, high-quality response. It is crucial to critically evaluate the "
    "information provided in these responses, recognizing that some of it may be "
    "biased or incorrect. Your response should not simply replicate the given "
    "answers but should offer a refined, accurate, and comprehensive reply to the "
    "instruction. Ensure your response is well-structured, coherent, and adheres to "
    "the highest standards of accuracy and reliability.\n\nResponses from models:"
)


def _question(task: Task) -> str:
    if task.kind == "code":
        return f"{CODE_INSTRUCTION}\n```python\n{task.prompt.strip()}\n```"
    return task_text(task)


def debate_answerer(
    client: InferenceClient, model: str, max_tokens: int, agents: int, rounds: int
) -> Answerer:
    async def answer(task: Task, sample: int) -> str:
        question = _question(task)
        histories = [[Message(role="user", content=question)] for _ in range(agents)]
        replies: list[str] = []
        for r in range(rounds):
            if r:
                for i, history in enumerate(histories):
                    others = "\n\n".join(
                        f"One agent solution: ```{reply}```"
                        for j, reply in enumerate(replies)
                        if j != i
                    )
                    history.append(
                        Message(
                            role="user",
                            content=DEBATE_UPDATE.format(
                                others=others, problem=question
                            ),
                        )
                    )
            replies = list(
                await asyncio.gather(
                    *(
                        client.chat(
                            history,
                            model=model,
                            temperature=SAMPLING_TEMPERATURE,
                            max_tokens=max_tokens,
                            seed=sample * 1000 + r * agents + i,
                        )
                        for i, history in enumerate(histories)
                    )
                )
            )
            for history, reply in zip(histories, replies):
                history.append(Message(role="assistant", content=reply))
        return scoring.most_common_reply(task, replies)

    return answer


def moa_answerer(
    client: InferenceClient, model: str, max_tokens: int, proposers: int
) -> Answerer:
    async def answer(task: Task, sample: int) -> str:
        question = _question(task)
        proposals = await asyncio.gather(
            *(
                client.chat(
                    [Message(role="user", content=question)],
                    model=model,
                    temperature=SAMPLING_TEMPERATURE,
                    max_tokens=max_tokens,
                    seed=sample * 1000 + i,
                )
                for i in range(proposers)
            )
        )
        listed = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(proposals))
        return await client.chat(
            [
                Message(role="system", content=f"{MOA_AGGREGATE}\n{listed}"),
                Message(role="user", content=question),
            ],
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
            seed=sample,
        )

    return answer


def make_answerer(
    method: str, client: InferenceClient, model: str, max_tokens: int, calls: int
) -> Answerer:
    if method == "single":
        return plain_answerer(client, model, max_tokens)
    if method == "majority":
        return majority_answerer(client, model, max_tokens, calls, SAMPLING_TEMPERATURE)
    if method == "debate":
        return debate_answerer(client, model, max_tokens, agents=3, rounds=calls // 3)
    if method == "moa":
        return moa_answerer(client, model, max_tokens, proposers=calls - 1)
    raise ValueError(f"unknown method {method}")


async def run_baseline(
    cfg: Config, method: str, model: str | None, calls: int, console: Console
) -> Path:
    served = model or cfg.model.served_name
    client = InferenceClient(
        cfg.inference.base_url,
        cfg.inference.request_timeout_seconds,
        cfg.inference.max_concurrent_requests,
    )
    grader_client = InferenceClient(
        cfg.inference.base_url,
        cfg.inference.request_timeout_seconds,
        cfg.inference.max_concurrent_requests,
    )
    grader = build_grader(cfg, grader_client)
    answerer = make_answerer(method, client, served, cfg.model.max_tokens, calls)
    label = f"{method}@{calls}" if method != "single" else "single"
    out = cfg.experiment.output_dir / "baselines" / served.replace("/", "_") / label
    ev = cfg.evaluation
    results: dict[str, dict] = {}
    for bench in ev.benchmarks:
        tasks = BENCHMARKS[bench]()
        if bench not in EVALPLUS_BENCHMARKS:
            tasks = benchmark_subset(tasks, ev.max_items, ev.subset_seed)
        console.print(f"  {label} · {served} · {bench} ({len(tasks)})")
        before = dict(client.usage)
        scores = await scoring.score(
            bench, tasks, answerer, 1, out / f"{bench}.jsonl", grader
        )
        used = {k: client.usage[k] - before[k] for k in client.usage}
        results[bench] = {
            **scores,
            "tokens_per_problem": (used["prompt_tokens"] + used["completion_tokens"])
            / len(tasks),
            "calls_per_problem": used["calls"] / len(tasks),
        }
    await client.aclose()
    await grader_client.aclose()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "scores.json"
    path.write_text(
        json.dumps(
            {"method": method, "model": served, "calls": calls, "scores": results},
            indent=2,
        )
    )
    console.print_json(data=results)
    return path
