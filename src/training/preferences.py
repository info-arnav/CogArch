"""DPO pairs from a cycle's rounds, per agent and per role.

Specialists: the known solution is preferred over the specialist's own
first-attempt proposal when that proposal behaves differently.

Coordinator: for each decision, a reply that adopted a proposal behaving like
the known solution is preferred over one that adopted a proposal that did not.
When the coordinator's actual reply was accepted or rejected by the world, that
reply is used directly.
"""

from __future__ import annotations

from src.inference.prompts import Persona, coordinator_selection, specialist_messages
from src.models.code import PreferencePair, RoundResult
from src.training.community import proposal_key

COORDINATOR = "coordinator"


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
        result = r.results[agent]
        agreement = r.agreement.get(agent, {})
        source = "self" if known.found_by == agent else known.found_by

        for proposal in result.decisions[0].proposals:
            if agreement.get(proposal_key(1, proposal.specialist)) is False:
                persona = specialists[proposal.specialist]
                pairs[proposal.specialist].append(
                    PreferencePair(
                        problem_id=r.problem.task_id,
                        agent=agent,
                        role=proposal.specialist,
                        prompt=specialist_messages(persona, r.problem),
                        chosen=_code_reply(known.code),
                        rejected=_code_reply(proposal.code),
                        source=source,
                    )
                )

        for decision in result.decisions:
            good = [
                p
                for p in decision.proposals
                if agreement.get(proposal_key(decision.attempt_number, p.specialist))
                is True
            ]
            bad = [
                p
                for p in decision.proposals
                if agreement.get(proposal_key(decision.attempt_number, p.specialist))
                is False
            ]
            if decision.result.passed and bad:
                chosen, rejected = decision.coordinator_output, coordinator_selection(
                    bad[0]
                )
            elif not decision.result.passed and good:
                chosen, rejected = (
                    coordinator_selection(good[0]),
                    decision.coordinator_output,
                )
            else:
                continue
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
