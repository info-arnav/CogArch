"""Who learns what from each task: competition between specialists.

The brain specialises because regions compete for inputs, and the region that
handles an input best is the one that learns from it (Jacobs' mixture of
experts). Here, on every task whose correct answer became known:

  winners  specialists whose answer was right (in either round, any attempt).
           Only they learn from the task: their correct reply is preferred
           over a wrong one (their own if they were wrong at first, otherwise
           a teammate's).
  owner    if nobody was right, the specialist with the best running record in
           this domain owns the task and learns the correct answer.
  others   learn nothing from this task.

The running record per (specialist, domain) is a reward-prediction-error
update, like dopamine-trained values in the basal ganglia. Small initial
differences (the personas) can grow into domain specialisation.

The coordinator learns separately: deciding for a position that was right is
preferred over deciding for one that was wrong.
"""

from __future__ import annotations

from collections import defaultdict

from src.inference.prompts import coordinator_endorsement
from src.models.core import Decision, PreferencePair, Proposal, RoundResult
from src.training.community import proposal_key

COORDINATOR = "coordinator"
MAX_COORDINATOR_PAIRS_PER_DECISION = 2


class DomainRecords:
    """Running success record per (specialist, domain): v ← v + α (outcome − v)."""

    def __init__(self, specialists: list[str], rate: float = 0.1, initial: float = 0.5):
        self.specialists = specialists
        self.rate = rate
        self.initial = initial
        self.values: dict[str, dict[str, float]] = {s: {} for s in specialists}

    def get(self, specialist: str, domain: str) -> float:
        return self.values[specialist].get(domain, self.initial)

    def update(self, specialist: str, domain: str, outcome: float) -> None:
        value = self.get(specialist, domain)
        self.values[specialist][domain] = value + self.rate * (outcome - value)

    def owner(self, domain: str) -> str:
        """Highest record wins; ties go to the earlier specialist."""
        return max(
            self.specialists,
            key=lambda s: (self.get(s, domain), -self.specialists.index(s)),
        )

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {s: dict(v) for s, v in self.values.items()}


def _correctness(r: RoundResult, agent: str) -> dict[str, bool]:
    return r.correct.get(agent, {})


def _proposals(d: Decision) -> list[Proposal]:
    return [*d.round1, *d.round2]


def build_pairs(
    rounds: list[RoundResult],
    agent: str,
    records: DomainRecords,
) -> dict[str, list[PreferencePair]]:
    """Preference pairs per role for one agent; updates its domain records."""
    pairs: dict[str, list[PreferencePair]] = {
        role: [] for role in [*records.specialists, COORDINATOR]
    }
    for r in rounds:
        known = r.known.get(agent)
        correct = _correctness(r, agent)
        if known is None or not correct:
            continue
        decisions = r.results[agent].decisions
        domain = r.task.domain
        source = "self" if known.found_by == agent else known.found_by

        def verdict(d: Decision, p: Proposal) -> bool | None:
            return correct.get(proposal_key(d.attempt_number, p.round, p.specialist))

        judged = [(d, p, verdict(d, p)) for d in decisions for p in _proposals(d)]
        winners = [
            s
            for s in records.specialists
            if any(v and p.specialist == s for _, p, v in judged)
        ]
        for s in records.specialists:
            if any(p.specialist == s and v is not None for _, p, v in judged):
                records.update(s, domain, 1.0 if s in winners else 0.0)
        responsible = winners or [records.owner(domain)]

        for s in responsible:
            own_right = next((p for _, p, v in judged if p.specialist == s and v), None)
            own_wrong = next(
                (p for _, p, v in judged if p.specialist == s and v is False), None
            )
            mate_wrong = next(
                (p for _, p, v in judged if p.specialist != s and v is False), None
            )
            rejected = own_wrong or mate_wrong
            if rejected is None:
                continue
            chosen = own_right.response if own_right else known.response
            pairs[s].append(
                PreferencePair(
                    task_id=r.task.task_id,
                    domain=domain,
                    agent=agent,
                    role=s,
                    prompt=decisions[0].specialist_prompts[s],
                    chosen=chosen,
                    rejected=rejected.response,
                    source=source,
                    responsibility="winner" if s in winners else "owner",
                )
            )

        for d in decisions:
            good = [p for p in d.round2 if verdict(d, p)]
            bad = [p for p in d.round2 if verdict(d, p) is False]
            if d.result.passed:
                options = [
                    (d.coordinator_output, coordinator_endorsement(p)) for p in bad
                ]
            elif good:
                options = [(coordinator_endorsement(good[0]), d.coordinator_output)]
            else:
                options = []
            for chosen_text, rejected_text in options[
                :MAX_COORDINATOR_PAIRS_PER_DECISION
            ]:
                pairs[COORDINATOR].append(
                    PreferencePair(
                        task_id=r.task.task_id,
                        domain=domain,
                        agent=agent,
                        role=COORDINATOR,
                        prompt=d.coordinator_prompt,
                        chosen=chosen_text,
                        rejected=rejected_text,
                        source=source,
                    )
                )
    return pairs


def specialist_accuracy(
    rounds: list[RoundResult], agent: str, specialists: list[str]
) -> dict[str, dict[str, float]]:
    """First-attempt, round-1 accuracy per specialist and domain (known truth only)."""
    hits: dict[str, dict[str, list[bool]]] = {s: defaultdict(list) for s in specialists}
    for r in rounds:
        correct = _correctness(r, agent)
        for s in specialists:
            v = correct.get(proposal_key(1, 1, s))
            if v is not None:
                hits[s][r.task.domain].append(v)
    return {s: {d: sum(v) / len(v) for d, v in by.items()} for s, by in hits.items()}


def discussion_stats(rounds: list[RoundResult], agent: str) -> tuple[float, float]:
    """(share of specialist answers changed by the discussion,
    share of changed answers with known truth that went wrong → right)."""
    changed = total = fixed = judged = 0
    for r in rounds:
        correct = _correctness(r, agent)
        for d in r.results[agent].decisions:
            for p1, p2 in zip(d.round1, d.round2):
                total += 1
                if p1.answer.strip() == p2.answer.strip():
                    continue
                changed += 1
                before = correct.get(proposal_key(d.attempt_number, 1, p1.specialist))
                after = correct.get(proposal_key(d.attempt_number, 2, p2.specialist))
                if before is not None and after is not None:
                    judged += 1
                    fixed += int(not before and after)
    return (changed / total if total else 0.0, fixed / judged if judged else 0.0)
