"""Grading replies against a reference answer, the way MASLab evaluates
multi-agent systems (Ye et al., 2025).

  maths, choice   xVerify (Chen et al., 2025), a model trained only to judge
                  whether a reply's final answer matches a reference. MASLab
                  measured 98% agreement with human graders for it, against
                  65% for the best rule-based extractor. Without a verifier
                  configured, the rule-based extractor is used instead.
  code            EvalPlus's sanitiser pulls the solution out of the reply; if
                  it finds no definition of the entry point, the base model is
                  asked to extract the code (MASLab's step). The code is then run.

The same grader judges benchmarks, accepted answers during training, and
which specialists were right, so scores and credit never disagree.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import TYPE_CHECKING

from src import calllog
from src.environment import answers_match, extract_answer
from src.inference.client import InferenceClient
from src.inference.prompts import task_text
from src.models.core import Message, Task

if TYPE_CHECKING:
    from src.config import Config

LETTERS = "ABCDEFGHIJ"

XVERIFY_PROMPT = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
-
Special considerations:

1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
-

Question: """{question}"""

Output sentence: """{output}"""

Correct answer: {answer}

Judgement:
'''

EXTRACT_CODE_PROMPT = (
    "Below is a reply to a programming task. Copy out the complete final Python "
    "solution that defines `{entry}`, with its imports. Output only one ```python "
    "code block and nothing else. If the reply contains no solution, output an "
    "empty code block.\n\nTask:\n{task}\n\nReply:\n{reply}"
)


def reference_text(task: Task, answer: str) -> str:
    if task.kind == "choice" and answer in LETTERS[: len(task.choices)]:
        return f"{answer}. {task.choices[LETTERS.index(answer)]}"
    return answer


def _defines(code: str, entry: str) -> bool:
    return bool(re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", code, re.MULTILINE))


class Grader:
    def __init__(
        self,
        verifier: InferenceClient | None,
        verifier_model: str,
        extractor: InferenceClient | None,
        extractor_model: str,
    ) -> None:
        self.verifier = verifier
        self.verifier_model = verifier_model
        self.extractor = extractor
        self.extractor_model = extractor_model
        self._cache: dict[str, bool | str] = {}
        self.counts = {"verified": 0, "unclear": 0, "rule": 0, "code_extracted": 0}

    def _key(self, *parts: str) -> str:
        return hashlib.sha256("\x00".join(parts).encode()).hexdigest()

    async def code(self, task: Task, reply: str) -> str:
        """The solution in a reply, as EvalPlus would submit it."""
        from evalplus.sanitize import sanitize

        key = self._key("code", task.task_id, reply)
        if key in self._cache:
            return str(self._cache[key])
        code: str = sanitize(reply, task.entry_point) if task.entry_point else ""
        by_model = False
        if not _defines(code, task.entry_point) and self.extractor is not None:
            by_model = True
            with calllog.tagged(grading="extract_code", task_id=task.task_id):
                extracted = await self.extractor.chat(
                    [
                        Message(
                            role="user",
                            content=EXTRACT_CODE_PROMPT.format(
                                entry=task.entry_point, task=task.prompt, reply=reply
                            ),
                        )
                    ],
                    model=self.extractor_model,
                    temperature=0.0,
                    max_tokens=1024,
                    seed=0,
                )
            code = sanitize(extracted, task.entry_point)
            self.counts["code_extracted"] += 1
        calllog.record(
            "code_extracted",
            task_id=task.task_id,
            reply=reply,
            code=code,
            by_model=by_model,
        )
        self._cache[key] = code
        return code

    async def correct(self, task: Task, reply: str, reference: str) -> bool:
        """Does the reply's final answer match the reference (maths, choice)?"""
        key = self._key("judge", task.task_id, reply, reference)
        if key in self._cache:
            return bool(self._cache[key])
        label = None
        if self.verifier is None:
            verdict = answers_match(task, reference, extract_answer(task, reply))
            self.counts["rule"] += 1
        else:
            with calllog.tagged(grading="verify", task_id=task.task_id):
                judgement = await self.verifier.chat(
                    [
                        Message(
                            role="user",
                            content=XVERIFY_PROMPT.format(
                                question=task_text(task),
                                output=reply,
                                answer=reference_text(task, reference),
                            ),
                        )
                    ],
                    model=self.verifier_model,
                    temperature=0.0,
                    max_tokens=8,
                    seed=0,
                )
            label = judgement.strip().strip("[]").strip().lower()
            if label not in ("correct", "incorrect"):
                self.counts["unclear"] += 1
            verdict = label == "correct"
            self.counts["verified"] += 1
        calllog.record(
            "graded",
            task_id=task.task_id,
            reply=reply,
            reference=reference,
            extracted=extract_answer(task, reply),
            verifier_label=label,
            correct=verdict,
        )
        self._cache[key] = verdict
        return verdict


RULE_GRADER = Grader(None, "", None, "")


def build_grader(cfg: Config, client: InferenceClient) -> Grader:
    g = cfg.grading
    verifier = (
        InferenceClient(
            g.verifier_url,
            cfg.inference.request_timeout_seconds,
            cfg.inference.max_concurrent_requests,
            os.environ.get("VERIFIER_API_KEY"),
        )
        if g.verifier_url
        else None
    )
    return Grader(
        verifier,
        g.verifier_model,
        client if g.extract_code_with_model else None,
        cfg.model.served_name,
    )
