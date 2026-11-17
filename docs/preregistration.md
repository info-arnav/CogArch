# Pre-registration: what each component must show

Written before the prompt study and the full runs. Each component of CogArch
is kept only if it shows the value stated here; otherwise it is removed before
the full runs (inference-time choices) or reported as not helping (trained
components). Thresholds are fixed now and not changed after seeing results.

All comparisons use the same base model (Qwen2.5-1.5B-Instruct), the same
grader (xVerify for maths and choice, hidden tests for code), and report
accuracy with 95% confidence intervals and tokens per problem.

## A. Decided by the prompt study (untrained, 50 practice problems per domain, never benchmark items)

| Component | Research basis | Kept if | Otherwise |
|---|---|---|---|
| Personas | Zheng et al. 2024 (personas do not help objective tasks); Subramaniam et al. 2025 (diverse agents) | The persona team is not worse than the neutral team overall (within 2 points), and no specialist loses more than 5 points alone in any domain | Neutral specialists; any persona losing over 5 points is rewritten or dropped |
| Discussion (and its style) | Du et al. 2023; Choi et al. 2025 (debate ≈ voting unless biased toward correction) | A style beats "no discussion" in team accuracy and fixes more answers than it breaks | No discussion for that kind of problem (maths/choice and code judged separately) |
| Coordinator choice | Mixture-of-Agents, Wang et al. 2025 | On problems where the team disagrees, its pick is right at least as often as the largest agreeing group | Majority vote replaces it (CodeT clustering for code) |

## B. Decided by the full runs (the results; reported whatever they show)

Three seeds per setting unless noted. Value means higher accuracy than the
ablation on the held-out benchmarks, averaged over seeds.

| Component | Research basis | Test | Seeds |
|---|---|---|---|
| The whole trained team | — | Full run vs the untrained team and vs majority vote at equal calls | 3 |
| Shared learning between two agents | Deep Mutual Learning (Zhang et al. 2018); codistillation (Anil et al. 2018); Multiagent Finetuning (Subramaniam et al. 2025) | Pair that shares accepted answers vs the same pair with sharing off (same compute; isolates sharing); solo agent for reference. Also reported: share of training pairs that came from a peer's answer | 3 + 3 + 3 |
| Memory (episodes; see revision below) | Reflexion (Shinn et al. 2023); ExpeL (Zhao et al. 2024) | Full vs memory off; recall hit rate reported | 3 |
| Brain-inspired credit rules | MALT (Motwani et al. 2025); the neuroscience sources for each rule | Full vs every specialist training on its own correct answers only | 3 |
| Personas during training | as above | Full vs neutral specialists | 3 |
| Own reasoning as training target | STaR (Zelikman et al. 2022) | Full vs the finder's reply as the target | 1 |
| Cross-domain practice pairs | replay against forgetting | Full vs no practice pairs; maths in particular | 1 |

## Reporting rules

- Every component's result is reported, positive or not.
- No prompt, threshold or setting is changed after the full runs start.
- Benchmarks are never used to choose anything.

## Prompt study results (400 practice problems, 100 per domain)

Run on 16 November 2026 with `cogarch prompt-study --per-domain 100`; raw data in
`data/runs/prompt_study/study2_100*.json`. Paired differences use the same
problems; 95% intervals by bootstrap; p-values by McNemar's exact test.

| Team setup (discussion / specialist prompt / selection) | Accuracy | Best single answer available (oracle) | Calls per problem |
|---|---|---|---|
| du / persona / coordinator | 69.5% | 78% | 6.4 |
| du / light persona / coordinator | 68.8% | 78% | 6.2 |
| du / neutral / coordinator | 69.3% | 77% | 6.1 |
| du / neutral / majority | 67.0% | 74% | 5.9 |
| du / light / majority | 67.2% | 76% | 6.0 |
| none / neutral / majority | 69.3% | 83% | 4.2 |
| none / neutral / coordinator | 68.5% | 83% | 4.5 |

| Comparison | Difference | 95% interval | p |
|---|---|---|---|
| neutral vs persona | -0.3 | -4.3 to +3.7 | 1.0 |
| light vs neutral | -0.5 | -4.5 to +3.3 | 0.90 |
| majority vs coordinator (after discussion) | -2.2 | -5.5 to +1.3 | 0.23 |
| discussion vs none (neutral, majority) | -2.2 | -5.5 to +0.7 | 0.21 |

Specialists alone: no persona costs more than 5 points in any domain against
the neutral prompt (largest: creative on social questions, -5).

### Decisions under the rules above

- Personas: kept. They neither help nor hurt the untrained team (-0.3, interval
  includes zero) and no specialist loses more than 5 points. They exist to seed
  different specialists for training; the training ablation tests that.
- Discussion: removed. No style beat answering alone; discussion made the
  specialists converge (oracle 83% → 74–78%) without raising accuracy, and cost
  about 50% more calls. Consistent with Choi et al. 2025.
- Selection: the largest agreeing group (majority vote; CodeT-style behaviour
  groups for code); the coordinator chooses only when groups tie. Without
  discussion the coordinator was right on 47% of split problems against 51% for
  the majority.

### Revision before the full runs: the decider chooses (17 November 2026)

The brain does not vote. When specialised systems propose conflicting actions,
conflict is detected (anterior cingulate; Botvinick et al. 2001) and one
system is selected (basal ganglia; Redgrave, Prescott and Gurney 1999), with
control given to whichever has recently been more reliable (Daw, Niv and Dayan
2005; Lee, Shimojo and O'Doherty 2014). CogArch follows this:

- All specialists agree: no conflict, their answer is submitted, no decider call.
- They disagree: the coordinator always chooses, shown each specialist's learned
  track record on that kind of problem (the same prediction-error records the
  credit rules use), and it is trained every sleep on whether its choices were
  right.

This replaces majority-vote selection above, as a design decision to follow
the brain rather than an outcome of the study. Majority vote is still computed
after the fact on the same answers every cycle, so the paper reports whether
the learning decider overtakes simple voting.

### Final design, approved 17 November 2026

Following the side-by-side comparison with the brain (decide, remember, learn):

- The coordinator decides on every answer, agreed or not, and sees each
  specialist's learned track record on that subject.
- It may reject every answer once per attempt ("try again"); the specialists
  then answer again and it must choose among the new answers.
- It is trained each sleep on its choices, including when rejecting was right
  (all first answers wrong) or wrong (one was right).
- Sleep replays pairs from earlier cycles alongside the new ones (half as many).
- The five credit rules are described as two principles: reward-error learning
  (own first answer, surprise, correcting the chosen specialist) and competition
  with balance (one learner, win-rate handicap). Mechanics unchanged.

Added run: no replay (1 seed). Majority vote over the same answers is computed
after the fact every cycle to test whether the learned selector beats pooling.

### Revision before the full runs: trust is learned, not shown (17 November 2026)

- The coordinator is shown the problem, each specialist's answer by name (in
  random order) and the checked results. It is never told the subject or any
  track record. Which specialist to trust on which kind of problem must be
  learned in its weights from its rewarded choices, as corticostriatal
  synapses learn which cortical channel to gate. The analysis uses the subject
  labels only afterwards, to plot whom it chooses per subject across cycles.
- Specialists are known to the coordinator only by a fixed letter (W, X, Y,
  Z), so names such as "logical" cannot suggest whom to trust. A to D were
  tried first; on multiple-choice problems the coordinator confused them with
  the answer options (in a local test, 23 of 31 named letters equalled that
  specialist's own answer option and 5 replies named an option outright), so
  letters that no option uses were chosen.

### Measurement: nothing is lost (17 November 2026)

- Every model call (prompt, settings, reply, tokens), grading verdict, code
  execution, memory recall, adapter load and team decision is logged with its
  phase, cycle, benchmark, task, agent, role and step in `calls.jsonl.gz`.
- On the practice set and every benchmark, the team's full decision is kept
  with all four specialists' answers graded, before and after any retry. This
  gives majority vote, the best answer available (oracle) and the
  coordinator's picks over the same answers.
- Benchmarks add a "specialist without persona" mode at the same weights and
  temperature, so a persona's cost or benefit is measured on the benchmarks
  themselves (full runs).
- In the full runs, each specialist is validated alone, with and without its
  persona, after every sleep.
- Memory is copied after every sleep (`memory_snapshots/cycle_N`).

### Revision before the full runs: memory as the brain keeps it (17 November 2026)

Tested on the stored memories of the earlier runs (500 problems each):

- Searching all past problems without the subject filter recalled same-subject
  problems 100% of the time, so the filter is removed. The per-subject
  similarity cut-offs stay: with one shared cut-off, social problems (which
  share a template) were recalled 88% of the time and maths and reasoning
  6–7%. They calibrate the embedding model and are never shown to the agent.
  The subject word is removed from stored episodes.
- Written lessons are removed. The stored lessons were specific to one
  problem, off-topic or generic, all at the same confidence, and the same
  five were shown on every problem; irrelevant context lowers accuracy (Shi
  et al. 2023). General knowledge is learned in the weights by sleep replay,
  as complementary learning systems theory places it in the neocortex
  (Kumaran, Hassabis and McClelland 2016). Sleep reflection still adds what
  went wrong to each failed episode.
- The list of recent events is removed: it named the subject and dataset,
  was unrelated to the problem at hand, and was empty at evaluation. Working
  memory is the current problem's own attempts and feedback.

The memory ablation therefore tests episodic memory (Reflexion; ExpeL's
retrieval of similar trajectories).

### Revision before the full runs: the coordinator replies with a letter (17 November 2026)

- The coordinator reads the problem, every position and the checked results,
  and replies with one letter (W, X, Y or Z), or "none" to send the team back
  once. The reply is constrained to those options, and its probability for each
  is logged. The basal ganglia select a channel without composing reasoning of
  their own; the reasoning is the specialists'.
- Its training pairs are single letters: its real choice against an option it
  could have taken, preferring whichever was right. Before this, half of each
  pair was a written sentence the coordinator never produced, so preference
  training could reward the wording rather than whom it trusted, and 18% of its
  choices named an answer instead of a letter, leaving whom it meant to a guess.
- The coordinator's instructions no longer list its letters. With the list
  "(W, X, Y, Z)", the first letter listed was almost never chosen whichever
  specialist held it (8 to 13 of 400 choices, in three arrangements; the same
  held for K-N, P-S and scattered letters), a known recency bias in language
  models (Zhao et al. 2021). Without the list, over four arrangements of 400
  problems each, the letters were chosen 19%, 25%, 25% and 31% of the time,
  while the specialist most often right was chosen most often whichever letter
  it held. Each specialist keeps its letter for the whole run; the remaining
  letter preference is the coordinator's starting point, reported from the
  logged choices.
- The plain model is scored on the same benchmark subsets as every other
  system.

### Run speed, before the full runs (17 November 2026)

- Each team works on 20 practice problems at a time instead of 4. Learning still
  happens only at night; the one difference is that a problem cannot recall
  another problem from the same group of 20, solved moments earlier.
- Training without gradient checkpointing was tried and does not fit: two
  training jobs beside the model server ran out of memory on a 48 GB GPU in the
  quick check, so checkpointing stays on.
