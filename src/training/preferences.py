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

The coordinator learns separately: deciding for a position that was right is
preferred over deciding for one that was wrong.
"""

from __future__ import annotations

from collections import defaultdict

from src.environment import answers_match
from src.inference.prompts import coordinator_endorsement
from src.models.core import Decision, PreferencePair, Proposal, RoundResult, Task
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


def _correctness(r: RoundResult, agent: str) -> dict[str, bool]:
    return r.correct.get(agent, {})


def _adopted(task: Task, final: str, proposal: Proposal) -> bool:
    """Did the coordinator's final answer come from this specialist's position?"""
    if task.kind == "code":
        return " ".join(final.split()) == " ".join(proposal.answer.split())
    return answers_match(task, final, proposal.answer)


def build_pairs(
    rounds: list[RoundResult],
    agent: str,
    records: DomainRecords,
    winner_take_all: bool = True,
    blame_adopted: bool = True,
    surprise_threshold: float | None = 0.5,
) -> dict[str, list[PreferencePair]]:
    """Preference pairs per role for one agent; updates its competition state."""
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
        else:
            learners = [records.owner(domain)]
        roles = {s: ("winner" if s in solved_alone else "owner") for s in learners}

        # 4. Attention tags: specialists whose position the coordinator adopted
        #    on a decision the world rejected.
        if blame_adopted:
            for d in decisions:
                if d.result.passed:
                    continue
                for p in d.round2:
                    if p.specialist not in roles and _adopted(r.task, d.answer, p):
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
                    if p.specialist == s and _adopted(r.task, d.answer, p)
                ),
                None,
            )
            mate_wrong = next(
                (p for _, p, v in judged if p.specialist != s and v is False), None
            )
            rejected = own_wrong or own_adopted_wrong or mate_wrong
            if rejected is None:
                continue
            chosen = own_right.response if own_right else known.response
            pair = PreferencePair(
                task_id=r.task.task_id,
                domain=domain,
                agent=agent,
                role=s,
                prompt=decisions[0].specialist_prompts[s],
                chosen=chosen,
                rejected=rejected.response,
                source=source,
                responsibility=responsibility,
            )
            outcome = 1.0 if responsibility == "winner" else 0.0
            surprising = (
                surprise_threshold is not None
                and abs(outcome - expected[s]) > surprise_threshold
            )
            pairs[s].extend([pair, pair] if surprising else [pair])

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
