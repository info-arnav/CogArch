# CogArch — Code Competition Architecture

<p align="center">
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-0F766E?style=flat-square" alt="MIT license" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python 3.10+" /></a>
  <a href="https://github.com/info-arnav/CogArch/issues"><img src="https://img.shields.io/github/issues/info-arnav/CogArch?style=flat-square" alt="Open issues" /></a>
  <a href="https://github.com/info-arnav/CogArch/stargazers"><img src="https://img.shields.io/github/stars/info-arnav/CogArch?style=flat-square" alt="GitHub stars" /></a>
</p>

Can two models make each other smarter — the way two humans sharpen each other through competition?

CogArch tests this by running two independent agent instances against real coding problems. Each agent has four specialists with different coding styles. They compete, and the better solutions become DPO training data for fine-tuning the next generation. Skill is measured before and after on HumanEval — a held-out benchmark neither agent ever trains on.

---

## The Idea

- **Two agents, no shared state** — Agent A and Agent B solve the same problem independently
- **Four specialists per agent** — logical (temp 0.3), creative (temp 0.7), skeptical (temp 0.4), empathetic (temp 0.5)
- **Up to 10 attempts per problem** — each failure feeds back error output for the next try (iterative refinement)
- **Verifiable reward signal** — code either passes the test assertions or it doesn't; partial credit via `pass_count / total_tests`
- **Both-fail pairs still train** — even when both agents fail, the one with higher partial credit becomes `chosen`; no round is wasted
- **HumanEval as held-out eval** — never trained on; measures true generalization (Pass@1, single attempt, no feedback)
- **MBPP as training arena** — competition rounds draw from MBPP; model never sees HumanEval during training

---

## Architecture

```mermaid
graph TD
    P[MBPP Problem] --> A[Agent A]
    P --> B[Agent B]

    subgraph Agent A
        A --> L1[Logical 0.3]
        A --> C1[Creative 0.7]
        A --> S1[Skeptical 0.4]
        A --> E1[Empathetic 0.5]
        L1 & C1 & S1 & E1 --> Coord1[Coordinator\nbest pass rate wins]
    end

    subgraph Agent B
        B --> L2[Logical 0.3]
        B --> C2[Creative 0.7]
        B --> S2[Skeptical 0.4]
        B --> E2[Empathetic 0.5]
        L2 & C2 & S2 & E2 --> Coord2[Coordinator\nbest pass rate wins]
    end

    Coord1 --> Judge[Judge\nrank by pass rate]
    Coord2 --> Judge

    Judge --> DPO[DPO Pairs\nchosen / rejected]
    DPO --> Finetune[QLoRA Fine-tune\nunsloth + DeepSeek-Coder 33B]
    Finetune --> Eval[HumanEval\nPass@1]
```

### Memory System

Unlike a standard LLM that starts fresh every call, CogArch maintains three persistent memory tiers that mirror how the human brain stores experience:

| Tier | Human equivalent | What's stored | Persists |
|------|-----------------|---------------|----------|
| **Working memory** | Active thoughts | Current session context, recent rounds | Session only |
| **Episodic memory** | "I remember solving X like this" | Every problem attempt + outcome, indexed by embedding | Forever (JSONL) |
| **Semantic memory** | General knowledge | Patterns promoted from episodes ("IndexError usually means off-by-one") | Forever (SQLite) |
| **Procedural memory** | Skills / instincts | Fine-tuned model weights | Forever (Ollama) |

**Retrieval** — before attempt 1 on any problem, the specialist embeds the problem description and queries episodic memory for similar past experiences. The top-3 matches are injected into the prompt: *"Similar problems you've solved before..."*

**Consolidation (sleep cycle)** — at the end of each experiment cycle, recurring patterns are extracted from recent episodes and promoted to semantic memory. Low-importance old episodes are pruned. This mirrors REM sleep's role in the hippocampus → cortex memory transfer.

**Embeddings** — `nomic-embed-text` via Ollama (768-dim, ~274MB, CPU-friendly). Falls back to recency-ranked retrieval if unavailable.

```
New problem arrives
       ↓
Embed problem → query episodic + semantic
       ↓
Inject top-K memories into specialist prompt
       ↓
Specialist solves (up to 10 attempts)
       ↓
Record outcome → episodic store + working memory
       ↓
[end of cycle] Consolidate: episodes → semantic patterns + prune
```

Memory lives in `data/memory/` — mounted as a Docker volume so it persists across container restarts, Colab sessions, and fine-tuning cycles. Run with `--no-memory` to disable.

### Iterative Refinement

Each specialist gets up to 10 attempts on a problem. After each failed attempt, it receives:
- How many assertions passed (`2/5 tests`)
- Which assertions failed (truncated to 120 chars)
- Any stderr output (last 8 lines of traceback)

The test assertions themselves are never shown — only the error feedback. This mirrors real development.

### DPO Pair Generation

Per specialist, per round:
| Outcome | `training_signal` | Action |
|---------|-------------------|--------|
| A wins  | `win` | A=chosen, B=rejected |
| A wins (both partial) | `partial_win` | A=chosen, B=rejected |
| Both partial, A higher | `both_partial` | A=chosen, B=rejected |
| Both fail equally | `both_fail_ranked` | A=chosen (by order), B=rejected |
| Identical code | — | Skip |

---

## Setup

### Option 1 — Docker Compose (recommended)

> Pull both models on first run — `deepseek-coder:33b` for inference/fine-tuning and `nomic-embed-text` for memory embeddings.

Works on your local machine, any cloud VM (Lambda Labs, RunPod, Vast.ai), or a server. GPU is used automatically if available.

```bash
git clone https://github.com/info-arnav/CogArch.git
cd CogArch

# First run: pull the model (only needed once — weights are cached in a volume)
docker compose --profile setup up model-pull

# Start Ollama + CogArch
docker compose up -d

# Baseline eval
docker compose run --rm cogarch python -m cli.main code-eval --limit 50

# Full experiment loop
docker compose run --rm cogarch python -m cli.main code-experiment --cycles 3 --rounds 30
```

**No GPU?** Remove the `deploy` blocks from `docker-compose.yml` — inference still works, fine-tuning will be skipped gracefully.

### Option 2 — Colab (A100)

Paste these cells in order into a Colab notebook with an A100 runtime.

**Cell 1 — Install Ollama + pull models**
```bash
%%bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve > /tmp/ollama.log 2>&1 &
sleep 8
ollama pull deepseek-coder:33b
ollama pull nomic-embed-text
```

**Cell 2 — Install Python deps**
```bash
%%bash
pip install -q \
  "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
  trl transformers datasets pydantic rich pyyaml httpx typer python-dotenv
```

**Cell 3 — Clone + install repo**
```bash
%%bash
git clone https://github.com/info-arnav/CogArch.git
cd CogArch && pip install -q -e .
```

**Cell 4 — Baseline HumanEval**
```python
import os
os.chdir("/content/CogArch")
import subprocess
subprocess.run(["python", "-m", "cli.main", "code-eval", "--limit", "50"])
```

**Cell 5 — Full experiment**
```python
subprocess.run([
    "python", "-m", "cli.main", "code-experiment",
    "--cycles", "3",
    "--rounds", "30",
])
```

### Option 3 — Local (no Docker)

```bash
git clone https://github.com/info-arnav/CogArch.git
cd CogArch
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Fine-tuning deps (GPU required)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
  trl transformers datasets

# Start Ollama separately
ollama serve &
ollama pull deepseek-coder:33b
ollama pull nomic-embed-text
```

---

## Usage

```bash
# Evaluate current model on HumanEval (Pass@1)
python -m cli.main code-eval --limit 50

# Full self-improvement loop: compete on MBPP → DPO fine-tune → eval HumanEval
# Memory is enabled by default — episodic + semantic + working
python -m cli.main code-experiment --cycles 3 --rounds 30

# Disable memory (e.g. for ablation — compare with/without)
python -m cli.main code-experiment --cycles 3 --rounds 30 --no-memory

# All flags
python -m cli.main code-experiment \
  --cycles 10 \
  --rounds 50 \
  --config config/default.yaml \
  --output data/experiments/code
```

Results are written to `data/experiments/code/`. Each cycle produces:
- `competition_results.jsonl` — all round outcomes with pass rates
- `training/{specialist}.jsonl` — DPO pairs
- `eval_scores.json` — HumanEval Pass@1 before and after

---

## File Layout

```
src/
  models/code.py              — all data models (CodeProblem, AttemptResult, etc.)
  execution/code_runner.py    — sandboxed subprocess execution, per-assertion testing
  inference/
    code_specialist.py        — iterative solver with memory retrieval + error feedback
    code_orchestrator.py      — runs 4 specialists in parallel, coordinator picks best
  memory/
    models.py                 — EpisodicEntry, SemanticEntry, WorkingMemoryItem, MemoryContext
    embedder.py               — nomic-embed-text via Ollama, cosine similarity
    episodic.py               — JSONL store with similarity retrieval + importance pruning
    semantic.py               — SQLite fact store, strengthened by repetition
    working.py                — bounded deque, session-scoped (capacity 10)
    consolidator.py           — sleep-cycle: episodes → semantic patterns, prune old entries
    controller.py             — coordinates all three tiers; record / retrieve / consolidate
  training/
    code_competitive.py       — two-agent competition loop
    code_dataset_builder.py   — builds DPO pairs from competition results
    finetuner.py              — QLoRA via unsloth, exports GGUF, registers with Ollama
  eval/
    benchmarks/
      humaneval.py            — eval-only (HumanEval, never trained on)
      mbpp.py                 — training-only (MBPP competition arena)
    code_experiment.py        — baseline → compete → finetune → consolidate → eval loop
cli/main.py                   — code-eval and code-experiment commands
config/default.yaml           — all tunable parameters
Dockerfile                    — cogarch Python image
docker-compose.yml            — ollama + cogarch services, data/memory volume
```

---

## Key Design Decisions

**Why DeepSeek-Coder 33B?** ~18GB in 4-bit via Ollama, ~22GB for QLoRA fine-tune. Fits in an 80GB A100 with room for the training framework. Strong coding baseline.

**Why same model for both agents?** Temperature variance makes outputs diverge. `logical` at temp 0.3 and `creative` at temp 0.7 produce genuinely different code — same principle as two humans with similar backgrounds making different choices.

**Why up to 10 attempts?** Real coding rarely works first try. Iterative refinement with error feedback is how actual development works. The model learns to read tracebacks and fix them.

**Why HumanEval as held-out eval?** It's the benchmark used when announcing new models (Claude, GPT-4, Gemini). Using the same standard makes scores directly comparable to published results — and since we never train on it, improvement is genuine.

**Why subprocess, not Docker, for code execution?** Each solution already runs in a separate Python process with a timeout — isolated, no shared state. Docker-in-Docker adds daemon dependency and latency with no security benefit for this use case.

**Why three memory tiers?** Fine-tuning (procedural) is slow, expensive, and irreversible — you can't fine-tune after every round. Working memory is fast but lost on restart. Episodic + semantic give the model persistent, queryable recall between sessions without touching weights. The three tiers mirror Complementary Learning Systems theory: fast episodic binding (hippocampus) + slow semantic consolidation (cortex) + procedural motor skills (cerebellum/basal ganglia).

**Why `nomic-embed-text` for embeddings?** 274MB, CPU-friendly, already served by the same Ollama instance. No second service, no API key, no latency spike. The embedding quality is sufficient for retrieval at this scale.

---

## Requirements

- Python 3.10+
- Docker + nvidia-container-toolkit (for Docker setup), or
- Ollama with `deepseek-coder:33b` + `nomic-embed-text` for local/Colab
- GPU required for fine-tuning (A100 recommended, T4 works with smaller batch)
- No API keys — runs 100% locally

---

## License

MIT — see [LICENSE.md](LICENSE.md)

Built by [Arnav Gupta](https://github.com/info-arnav) • [Report an issue](https://github.com/info-arnav/CogArch/issues)
