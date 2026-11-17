"""Who learns what from each task: learning rules taken from neuroscience.

On every task whose correct answer became known (the world accepted an answer,
or the partner agent shared one):

1. Causality (spike-timing-dependent plasticity). A connection strengthens only
   if the neuron fired *before* the outcome, i.e. helped cause it. Only a
   specialist's OWN round-1 answer counts; agreeing after the discussion does not.

2. Winner-take-all (lateral inhibition, competitive learning). Among the
   specialists that solved the task alone, only one learns from it: the one with
   the best adjusted record in this domain. If nobody solved it alone, the same
   rule picks the domain's "owner", who learns the correct answer.

3. Homeostasis (a "conscience", DeSieno 1988). Specialists that win often are
   handicapped and rarely-winning ones helped, so no single specialist takes
   over every domain:  adjusted = record + conscience × (1/N − win_rate).

4. Attention tags (attention-gated reinforcement learning, Roelfsema). If the
   coordinator adopted a specialist's answer and the world said it was wrong,
   that specialist is blamed and learns the correct answer instead.

5. Surprise (dopamine reward-prediction error). Pairs whose outcome was
   surprising given the specialist's record (|outcome − record| > threshold)
   are included twice, so unexpected results teach more.

The running record per (specialist, domain) is itself a prediction-error update:
v ← v + α (outcome − v).

The coordinator learns separately, from its own choices: each pair is two
single letters (or 'none'), its real choice against an option it could have
taken, preferring whichever was right.

Anchors keep every specialist practised in every domain: a specialist whose own
first answer was right, on a task it did not learn from, also gets that answer
preferred over a teammate's wrong one, up to `anchor_ratio` times its main pairs,
spread evenly over domains. Without them, winner-take-all trains each specialist
on one domain only and the others fade.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from src.inference.prompts import REJECT_ALL
from src.models.core import (
    Decision,
    Message,
    PreferencePair,
    Proposal,
    RoundResult,
)
from src.training.community import proposal_key

COORDINATOR = "coordinator"
MAX_COORDINATOR_PAIRS_PER_DECISION = 2


class DomainRecords:
    """Per-agent competition state: success records per (specialist, domain)
    and how often each specialist has been the one that learns."""

    def __init__(
        self,
        specialists: list[str],
        rate: float = 0.1,
        initial: float = 0.5,
        conscience: float = 0.5,
        win_rate_step: float = 0.05,
    ) -> None:
        self.specialists = specialists
        self.rate = rate
        self.initial = initial
        self.conscience = conscience
        self.win_rate_step = win_rate_step
        self.values: dict[str, dict[str, float]] = {s: {} for s in specialists}
        self.win_rate: dict[str, float] = {s: 1 / len(specialists) for s in specialists}

    def get(self, specialist: str, domain: str) -> float:
        return self.values[specialist].get(domain, self.initial)

    def update(self, specialist: str, domain: str, outcome: float) -> None:
        value = self.get(specialist, domain)
        self.values[specialist][domain] = value + self.rate * (outcome - value)

    def adjusted(self, specialist: str, domain: str) -> float:
        fair_share = 1 / len(self.specialists)
        return self.get(specialist, domain) + self.conscience * (
            fair_share - self.win_rate[specialist]
        )

    def best(self, candidates: list[str], domain: str) -> str:
        """Highest adjusted record; ties go to the earlier specialist."""
        return max(
            candidates,
            key=lambda s: (self.adjusted(s, domain), -self.specialists.index(s)),
        )

    def owner(self, domain: str) -> str:
        return self.best(self.specialists, domain)

    def record_learners(self, learners: list[str]) -> None:
        for s in self.specialists:
            won = 1.0 if s in learners else 0.0
            self.win_rate[s] += self.win_rate_step * (won - self.win_rate[s])

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {s: dict(v) for s, v in self.values.items()}

    def restore(
        self, values: dict[str, dict[str, float]], win_rate: dict[str, float]
    ) -> None:
        self.values = {s: dict(values[s]) for s in self.specialists}
        self.win_rate = {s: win_rate[s] for s in self.specialists}


def _correctness(r: RoundResult, agent: str) -> dict[str, bool]:
    return r.correct.get(agent, {})


def _adopted(decision: Decision, proposal: Proposal) -> bool:
    """Was this specialist's final position the one submitted?"""
    return proposal.round == 2 and decision.chosen == proposal.specialist


def build_pairs(
    rounds: list[RoundResult],
    agent: str,
    records: DomainRecords,
    winner_take_all: bool = True,
    blame_adopted: bool = True,
    surprise_threshold: float | None = 0.5,
    anchor_ratio: float = 0.0,
    owner_learns: bool = True,
    log: list[dict[str, Any]] | None = None,
) -> dict[str, list[PreferencePair]]:
    """Preference pairs per role for one agent; updates its competition state.
    With `log`, appends what each rule decided on each task."""
    pairs: dict[str, list[PreferencePair]] = {
        role: [] for role in [*records.specialists, COORDINATOR]
    }
    anchors: dict[str, list[PreferencePair]] = {s: [] for s in records.specialists}
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

        # 1. Causality: only independent (round-1) answers count.
        judged = [(d, p, verdict(d, p)) for d in decisions for p in d.round1]
        solved_alone = [
            s
            for s in records.specialists
            if any(v and p.specialist == s for _, p, v in judged)
        ]
        expected = {s: records.get(s, domain) for s in records.specialists}

        # 2–3. Winner-take-all with homeostasis (or every solo winner, if disabled).
        if solved_alone:
            learners = (
                [records.best(solved_alone, domain)]
                if winner_take_all
                else solved_alone
            )
        elif owner_learns:
            learners = [records.owner(domain)]
        else:
            learners = []
        roles = {s: ("winner" if s in solved_alone else "owner") for s in learners}

        # 4. Attention tags: specialists whose position the coordinator adopted
        #    on a decision the world rejected.
        if blame_adopted:
            for d in decisions:
                if d.result.passed:
                    continue
                for p in d.round2:
                    if p.specialist not in roles and _adopted(d, p):
                        roles[p.specialist] = "blamed"

        for s in records.specialists:
            if any(p.specialist == s and v is not None for _, p, v in judged):
                records.update(s, domain, 1.0 if s in solved_alone else 0.0)
        records.record_learners(list(roles))

        for s, responsibility in roles.items():
            own_right = next((p for _, p, v in judged if p.specialist == s and v), None)
            own_wrong = next(
                (p for _, p, v in judged if p.specialist == s and v is False), None
            )
            own_adopted_wrong = next(
                (
                    p
                    for d in decisions
                    if not d.result.passed
                    for p in d.round2
                    if p.specialist == s and _adopted(d, p)
                ),
                None,
            )
            mate_wrong = next(
                (p for _, p, v in judged if p.specialist != s and v is False), None
            )
            rejected = own_wrong or own_adopted_wrong or mate_wrong
            if rejected is None:
                continue
            pair = PreferencePair(
                task_id=r.task.task_id,
                domain=domain,
                agent=agent,
                role=s,
                prompt=decisions[0].specialist_prompts[s],
                chosen=own_right.response if own_right else known.response,
                rejected=rejected.response,
                source=source,
                responsibility=responsibility,
                chosen_from="own" if own_right else "known",
            )
            outcome = 1.0 if responsibility == "winner" else 0.0
            surprising = (
                surprise_threshold is not None
                and abs(outcome - expected[s]) > surprise_threshold
            )
            pairs[s].extend([pair, pair] if surprising else [pair])
            if log is not None:
                log.append(
                    {
                        "task_id": r.task.task_id,
                        "domain": domain,
                        "specialist": s,
                        "responsibility": responsibility,
                        "surprising": surprising,
                        "record": round(expected[s], 3),
                        "solved_alone": solved_alone,
                        "chosen_from": pair.chosen_from,
                    }
                )

        if anchor_ratio > 0:
            mate_wrong_any = next((p for _, p, v in judged if v is False), None)
            for s in records.specialists:
                if s in roles or mate_wrong_any is None:
                    continue
                own_right = next(
                    (p for _, p, v in judged if p.specialist == s and v), None
                )
                mate = next(
                    (p for _, p, v in judged if p.specialist != s and v is False), None
                )
                if own_right is not None and mate is not None:
                    anchors[s].append(
                        PreferencePair(
                            task_id=r.task.task_id,
                            domain=domain,
                            agent=agent,
                            role=s,
                            prompt=decisions[0].specialist_prompts[s],
                            chosen=own_right.response,
                            rejected=mate.response,
                            source=source,
                            responsibility="anchor",
                            chosen_from="anchor",
                        )
                    )

        for d in decisions:
            if not d.coordinator_prompt:
                continue
            for prompt, chosen_text, rejected_text in _rejection_pairs(d, verdict):
                pairs[COORDINATOR].append(
                    PreferencePair(
                        task_id=r.task.task_id,
                        domain=domain,
                        agent=agent,
                        role=COORDINATOR,
                        prompt=prompt,
                        chosen=chosen_text,
                        rejected=rejected_text,
                        source=source,
                    )
                )
            for chosen_text, rejected_text in _choice_pairs(d, verdict):
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
    for s, extra in anchors.items():
        pairs[s].extend(_spread(extra, math.ceil(anchor_ratio * len(pairs[s]))))
    return pairs


def _choice_pairs(d: Decision, verdict: Any) -> list[tuple[str, str]]:
    """The coordinator's final choice against the alternatives it had, each a
    single letter (or 'none'), so a pair differs only in whom it trusted. Its
    real choice is in every pair; the other side is an option it could have
    taken. Decisions where it gave no valid reply teach it nothing."""
    if d.chosen_by != "coordinator":
        return []
    good = [d.labels[p.specialist] for p in d.round2 if verdict(d, p)]
    bad = [d.labels[p.specialist] for p in d.round2 if verdict(d, p) is False]
    own = d.labels[d.chosen]
    if own in good:
        pairs = [(own, other) for other in bad]
    elif good:
        pairs = [(right, own) for right in good]
    elif bad and d.could_reject:
        pairs = [(REJECT_ALL, own)]
    else:
        pairs = []
    return pairs[:MAX_COORDINATOR_PAIRS_PER_DECISION]


def _rejection_pairs(d: Decision, verdict: Any) -> list[tuple[list[Message], str, str]]:
    """When the coordinator sent the team back: right to do so if no first
    answer was correct, wrong if one was."""
    if not d.rejected_all or not d.first_choice_prompt:
        return []
    good = [d.labels[p.specialist] for p in d.round1 if verdict(d, p)]
    bad = [d.labels[p.specialist] for p in d.round1 if verdict(d, p) is False]
    if good:
        pairs = [(right, REJECT_ALL) for right in good]
    elif bad:
        pairs = [(REJECT_ALL, wrong) for wrong in bad]
    else:
        pairs = []
    return [
        (d.first_choice_prompt, chosen, rejected)
        for chosen, rejected in pairs[:MAX_COORDINATOR_PAIRS_PER_DECISION]
    ]


def _spread(pairs: list[PreferencePair], limit: int) -> list[PreferencePair]:
    """Up to `limit` pairs, taken from each domain in turn."""
    by_domain: dict[str, list[PreferencePair]] = defaultdict(list)
    for pair in pairs:
        by_domain[pair.domain].append(pair)
    chosen: list[PreferencePair] = []
    queues = list(by_domain.values())
    while len(chosen) < limit and any(queues):
        for queue in queues:
            if queue and len(chosen) < limit:
                chosen.append(queue.pop(0))
    return chosen


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
    """Over discussed attempts: (share of specialist answers changed by the
    discussion, share of changed answers with known truth that went wrong →
    right). Code counts as changed when its judged correctness changed, since
    rewritten code often behaves the same."""
    changed = total = fixed = judged = 0
    for r in rounds:
        correct = _correctness(r, agent)
        for d in r.results[agent].decisions:
            if not d.discussed:
                continue
            for p1, p2 in zip(d.round1, d.round2):
                total += 1
                before = correct.get(proposal_key(d.attempt_number, 1, p1.specialist))
                after = correct.get(proposal_key(d.attempt_number, 2, p2.specialist))
                if r.task.kind == "code":
                    moved = before is not None and after is not None and before != after
                else:
                    moved = p1.answer.strip() != p2.answer.strip()
                if not moved:
                    continue
                changed += 1
                if before is not None and after is not None:
                    judged += 1
                    fixed += int(not before and after)
    return (changed / total if total else 0.0, fixed / judged if judged else 0.0)
