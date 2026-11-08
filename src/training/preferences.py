"""DPO pairs from a cycle's rounds, per agent and per role.

Every attempt contributes, not only the first:

Specialists: for each attempt, a proposal that behaves differently from the
known solution yields a pair (known solution preferred over that proposal),
with the prompt the specialist actually saw at that attempt (minus memory),
so retries also teach how to recover after being told "wrong".

Coordinator: for each decision, adopting a proposal that behaves like the
known solution is preferred over adopting one that does not. When the
coordinator's actual reply was accepted or rejected by the world, that reply
is used directly as the chosen or rejected side.
"""

from __future__ import annotations

from src.inference.prompts import Persona, coordinator_selection, specialist_messages
from src.models.code import Decision, PreferencePair, RoundResult
from src.training.community import proposal_key

COORDINATOR = "coordinator"
MAX_COORDINATOR_PAIRS_PER_DECISION = 2


def _code_reply(code: str) -> str:
    return f"```python\n{code}\n```"


def build_pairs(
    rounds: list[RoundResult],
    agent: str,
    specialists: dict[str, Persona],
) -> dict[str, list[PreferencePair]]:
    pairs: dict[str, list[PreferencePair]] = {
        role: [] for role in [*specialists, COORDINATOR]
    }
    for r in rounds:
        known = r.known_solution.get(agent)
        if known is None:
            continue
        decisions = r.results[agent].decisions
        agreement = r.agreement.get(agent, {})
        source = "self" if known.found_by == agent else known.found_by

        def verdict(decision: Decision, specialist: str) -> bool | None:
            return agreement.get(proposal_key(decision.attempt_number, specialist))

        for index, decision in enumerate(decisions):
            previous = decisions[index - 1] if index > 0 else None
            for proposal in decision.proposals:
                if verdict(decision, proposal.specialist) is False:
                    persona = specialists[proposal.specialist]
                    pairs[proposal.specialist].append(
                        PreferencePair(
                            problem_id=r.problem.task_id,
                            agent=agent,
                            role=proposal.specialist,
                            prompt=specialist_messages(
                                persona, r.problem, previous=previous
                            ),
                            chosen=_code_reply(known.code),
                            rejected=_code_reply(proposal.code),
                            source=source,
                        )
                    )

            good = [
                p for p in decision.proposals if verdict(decision, p.specialist) is True
            ]
            bad = [
                p
                for p in decision.proposals
                if verdict(decision, p.specialist) is False
            ]
            if decision.result.passed:
                options = [
                    (decision.coordinator_output, coordinator_selection(p)) for p in bad
                ]
            elif good:
                options = [
                    (coordinator_selection(good[0]), decision.coordinator_output)
                ]
            else:
                options = []
            for chosen, rejected in options[:MAX_COORDINATOR_PAIRS_PER_DECISION]:
                pairs[COORDINATOR].append(
                    PreferencePair(
                        problem_id=r.problem.task_id,
                        agent=agent,
                        role=COORDINATOR,
                        prompt=decision.coordinator_prompt,
                        chosen=chosen,
                        rejected=rejected,
                        source=source,
                    )
                )
    return pairs
