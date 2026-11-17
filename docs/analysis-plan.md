# Analysis plan

The questions the final analysis answers, in order, with what the data showed
while the runs were still going (17–18 Nov 2026). Every early observation is
re-checked on the complete data before it goes in the paper.

## 1. Sanity checks

- Every run finished: report.json has five cycles and final benchmarks.
- Grading spot-checked by hand on a sample of every benchmark.
- The code-grading stand-in disagreeing with the world verdict (about 0.6% of
  decisions) and identical training pairs (about 0.17%) measured on the final
  data and reported as a limitation.
- Runs that stopped and were resumed (host restarts on 17 Nov, one training
  out-of-memory crash, the adapter-unload bug after restarts) listed with what
  was redone; each resumed from its last finished cycle.
- The grader pod was down from about 04:22 to 04:36 IST on 18 Nov. Every
  verdict logged in that window must come from a successful grader reply after
  retry, not from a fallback.
- Hugging Face refused uploads from one pod's address (429/403); every run's
  final and complete folders are on the repos before the analysis starts.

## 2. Main question

Does the full architecture beat the untrained team, a single model, majority
vote (5 and 9), Mixture-of-Agents and debate on the eight benchmarks? Three
replicates, mean and spread.

## 3. Each component

Full against each ablation: no-memory, no-sharing, neutral personas,
simple-credit, solo, no-star, no-practice, no-replay.

## 4. Questions raised while watching the runs

| Question | Early evidence | How to check |
|---|---|---|
| Does memory cause the social-reasoning drop? | Every run with memory lost 4–10 points on SIQA; both finished no-memory runs did not, and no-memory improved on all eight benchmarks. | Third no-memory replicate; SIQA accuracy on problems where a memory was shown against not shown; whether recalled answers were copied onto new problems; memory share by night (3% to 34%) against the drop. |
| Is the coordinator better than voting, and where? | On split problems the coordinator went from 61% to 70% while majority stayed near 63%. Code 84% to 91% (with test results as evidence), maths 57% to 73%, reasoning and social up a few points. | All runs, by subject and night, against a random pick and majority. |
| Why does the coordinator favour creative? | Creative gets 34–43% of picks in every subject. Untrained it was an average specialist; after training it became the best in maths (73 to 79) and social (58 to 67). | Night-by-night picks, credit and training pairs per specialist: whether being picked led to more training and then to improvement (a rich-get-richer loop). |
| Do personas help or hurt? | Each specialist alone scores 3–13 points higher without its persona, most on code. After training, creative's persona stopped costing it (maths 79 with, 77 without). | Specialists with and without persona, before and after training, on validation and benchmarks; neutral-persona runs against full at team level. |
| Do trained specialists get worse alone while the team gets better? | Trained logical alone on HumanEval/MBPP: 32/38 against 49/60 untrained, while the trained team rose on MBPP. | Every trained specialist on the benchmarks (full runs), against the untrained specialists. |
| Does one team learn as well as two? | Solo and no-sharing look alike on the benchmarks. | All replicates; validation curves. |
| What does replay do to forgetting? | Not measured yet. | No-replay against full on SIQA and the other non-trained abilities. |
| Where does the architecture beat standard methods, and where not? | Untrained, the team wins on code (MBPP 68.8 against at most 58.7) but plain majority voting beats it on maths, science, social and emotion (MATH 61.3 for majority of 9 against 50.3; ARC 76.7 against 68.0); a single model without a persona beats the untrained team on 6 of 8. The best trained version so far still trails majority voting outside code. | Full runs against every baseline per benchmark, with calls per problem, so cost is compared too; personas as a cause of the gap outside code. |
| When does the coordinator send the team back ("none")? | It started doing so after training; sometimes the retry is right. | Rate and success of retries by night and subject. |
