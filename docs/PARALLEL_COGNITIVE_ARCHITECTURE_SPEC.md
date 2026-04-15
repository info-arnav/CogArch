# Parallel Cognitive Architecture for Machine Consciousness

## Build Specification — Agent Handoff Document

**Version:** 0.1 (MVP)
**Date:** April 2026
**Status:** Ready to build

---

## 1. What This Is

A Python framework where multiple specialist LLMs run in parallel on every input, share their outputs with each other, reach consensus through a lightweight coordinator model, and improve over time through a sleep-cycle fine-tuning loop.

Think of it as: **a small model that learns how to orchestrate bigger models, inspired by how the brain's prefrontal cortex coordinates specialized brain regions.**

---

## 2. Project Structure

**Design principles:**
- Specialists are configs (prompts + adapters), NOT separate Python files — same class, different personalities
- Inference and training are fully separated — you can run inference with zero training dependencies
- Prompts are first-class citizens with their own directory and versioning
- Every data artifact has a clear home and schema
- CLI is a thin layer over the library — everything is importable

```
cogarch/
│
├── src/cogarch/                      # ── LIBRARY CODE ──
│   ├── __init__.py
│   │
│   ├── models/                       # Data structures (no logic, just shapes)
│   │   ├── __init__.py
│   │   ├── specialist.py             #   SpecialistOutput, SpecialistConfig
│   │   ├── coordinator.py            #   CoordinatorOutput, SelfState
│   │   ├── interaction.py            #   InteractionRecord, CompetitiveResult
│   │   └── training.py               #   TrainingExample, CuratedDataset, SleepReport
│   │
│   ├── inference/                    # ── WAKING CYCLE (runs without training deps) ──
│   │   ├── __init__.py
│   │   ├── specialist.py             #   Specialist class — one class, driven by config
│   │   ├── coordinator.py            #   Coordinator class — synthesis + attribution + self-state
│   │   ├── consensus.py              #   Revision pass logic (Round 2 deliberation)
│   │   ├── orchestrator.py           #   Main loop: broadcast → round1 → round2 → synthesize
│   │   └── backends/                 #   LLM provider adapters (swap without changing logic)
│   │       ├── __init__.py
│   │       ├── base.py               #     Abstract LLMBackend interface
│   │       ├── ollama.py             #     Local Ollama
│   │       ├── vllm.py               #     Local vLLM
│   │       ├── openai.py             #     OpenAI-compatible APIs (OpenAI, Together, Groq)
│   │       └── anthropic.py          #     Anthropic API
│   │
│   ├── memory/                       # ── PERSISTENCE ──
│   │   ├── __init__.py
│   │   ├── experience_log.py         #   Append-only interaction log (JSONL + index)
│   │   ├── self_state.py             #   Self-state persistence + update logic
│   │   └── stores/                   #   Pluggable storage backends
│   │       ├── __init__.py
│   │       ├── jsonl.py              #     File-based (default, zero deps)
│   │       └── sqlite.py             #     SQLite (better for large logs + queries)
│   │
│   ├── training/                     # ── SLEEP CYCLE + COMPETITIVE LEARNING ──
│   │   ├── __init__.py
│   │   ├── sleep_cycle.py            #   Orchestrates: curate → assemble → finetune → validate
│   │   ├── curator.py                #   Selects high-signal interactions (disagreement, vindication)
│   │   ├── dataset_builder.py        #   Builds per-specialist training sets from curated data
│   │   ├── fine_tuner.py             #   LoRA fine-tuning via PEFT (specialist adapters)
│   │   ├── competitive.py            #   Two-agent competitive loop
│   │   └── vindication.py            #   Post-hoc check: was a deprioritized specialist actually right?
│   │
│   ├── eval/                         # ── BENCHMARKS + METRICS ──
│   │   ├── __init__.py
│   │   ├── benchmarks/
│   │   │   ├── __init__.py
│   │   │   ├── base.py               #     Abstract Benchmark interface
│   │   │   ├── arc_agi.py            #     ARC-AGI loader + scorer
│   │   │   ├── frontier_math.py      #     FrontierMath loader + scorer
│   │   │   └── causal.py             #     Causal inference tasks
│   │   ├── scorer.py                 #   Generic scoring (exact match, fuzzy, LLM-as-judge)
│   │   └── metrics.py                #   Routing accuracy, vindication rate, specialist curves
│   │
│   ├── api/                          # ── HTTP SERVER ──
│   │   ├── __init__.py
│   │   ├── server.py                 #   FastAPI app
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── inference.py          #     POST /infer
│   │   │   ├── training.py           #     POST /sleep, POST /competitive/run
│   │   │   ├── metrics.py            #     GET /metrics, GET /self-state
│   │   │   └── health.py             #     GET /health
│   │   └── schemas.py                #   Pydantic request/response models
│   │
│   └── config.py                     #   Config loader (YAML → typed Python objects)
│
├── prompts/                          # ── SPECIALIST PERSONALITIES (first-class, versioned) ──
│   ├── specialists/
│   │   ├── logical.yaml              #   System prompt + generation params for logical
│   │   ├── creative.yaml             #   System prompt + generation params for creative
│   │   ├── skeptical.yaml            #   System prompt + generation params for skeptical
│   │   └── empathetic.yaml           #   System prompt + generation params for empathetic
│   ├── coordinator/
│   │   ├── synthesis.yaml            #   Coordinator's synthesis prompt template
│   │   └── self_state_update.yaml    #   Prompt for self-state reflection
│   └── revision/
│       └── revision_pass.yaml        #   Round 2 prompt template (shared by all specialists)
│
├── cli/                              # ── CLI ENTRY POINTS ──
│   ├── __init__.py
│   ├── main.py                       #   Click/Typer CLI app: `cogarch <command>`
│   ├── infer.py                      #   `cogarch infer "question here"`
│   ├── compete.py                    #   `cogarch compete --rounds 50 --benchmark arc-agi`
│   ├── sleep.py                      #   `cogarch sleep --curate --finetune`
│   ├── eval.py                       #   `cogarch eval --benchmark arc-agi --split test`
│   └── dashboard.py                  #   `cogarch dashboard` — live metrics in terminal (rich)
│
├── tests/
│   ├── unit/
│   │   ├── test_specialist.py
│   │   ├── test_coordinator.py
│   │   ├── test_consensus.py
│   │   ├── test_curator.py
│   │   └── test_vindication.py
│   ├── integration/
│   │   ├── test_orchestrator_e2e.py  #   Full pipeline with mock LLMs
│   │   ├── test_sleep_cycle_e2e.py
│   │   └── test_competitive_e2e.py
│   ├── fixtures/
│   │   ├── mock_specialist_outputs.json
│   │   ├── sample_experience_log.jsonl
│   │   └── sample_benchmark_items.json
│   └── conftest.py                   #   Shared fixtures, mock LLM backend
│
├── data/                             # ── ALL RUNTIME DATA (gitignored except schemas) ──
│   ├── experience_log/               #   Interaction logs (JSONL files, one per session)
│   │   └── .schema.json              #   JSON schema for InteractionRecord (checked in)
│   ├── checkpoints/                  #   LoRA adapter weights per specialist per sleep cycle
│   │   ├── logical/
│   │   │   ├── cycle_001/
│   │   │   ├── cycle_002/
│   │   │   └── latest -> cycle_002   #   Symlink to latest checkpoint
│   │   ├── creative/
│   │   ├── skeptical/
│   │   └── coordinator/              #   Coordinator's own weight snapshots
│   ├── benchmarks/                   #   Downloaded benchmark datasets
│   ├── metrics/                      #   Evaluation results (JSON per run)
│   └── self_state/                   #   Persisted self-state snapshots
│       └── current.json
│
├── config/                           # ── CONFIGURATION ──
│   ├── default.yaml                  #   Full default config
│   ├── local.yaml                    #   Local overrides (gitignored)
│   ├── profiles/
│   │   ├── dev_cpu.yaml              #   CPU-only dev (sequential inference, small models)
│   │   ├── dev_gpu.yaml              #   Single GPU dev (parallel, quantized)
│   │   ├── prod_multi_gpu.yaml       #   Multi-GPU production
│   │   └── cloud_api.yaml            #   API-based specialists (OpenAI/Anthropic, no local GPU)
│   └── .schema.json                  #   Config validation schema (checked in)
│
├── docs/                             # ── DOCUMENTATION ──
│   ├── architecture.md               #   High-level system design + diagrams
│   ├── consensus.md                  #   Deep dive: how consensus works
│   ├── sleep_cycle.md                #   Deep dive: training loop
│   ├── adding_specialists.md         #   Guide: how to add a new specialist (just a YAML file)
│   ├── adding_backends.md            #   Guide: how to add a new LLM provider
│   └── benchmarks.md                 #   Supported benchmarks + how to add new ones
│
├── pyproject.toml                    #   Package config, deps, entry points
├── requirements.txt                  #   Pinned deps for reproducibility
├── requirements-training.txt         #   Training-only deps (torch, peft, accelerate)
├── Makefile                          #   Common commands: make infer, make compete, make sleep
├── docker-compose.yaml               #   Optional: Ollama + vLLM + CogArch API
├── .env.example                      #   API keys template
├── .gitignore
├── LICENSE
└── README.md
```

---

## 3. Core Components — What Each Does

### 3.1 Specialist (`src/cogarch/inference/specialist.py`)

ONE class that handles ALL specialists. The personality comes from the YAML config in `prompts/specialists/`, not from different Python files. Adding a new specialist = adding a new YAML file.

```python
class Specialist:
    name: str                    # e.g. "logical", "creative", "skeptical"
    model_id: str                # e.g. "meta-llama/Llama-3-8B-Instruct" or an API endpoint
    system_prompt: str           # Defines this specialist's cognitive style
    lora_adapter_path: str | None  # Path to current LoRA adapter (None = base model)

    async def generate(self, input: str, context: dict | None = None) -> SpecialistOutput:
        """
        Generate a response with reasoning trace.
        context = other specialists' Round 1 outputs (for revision pass).
        Returns SpecialistOutput with: answer, reasoning_trace, confidence (0-1).
        """

    async def revise(self, input: str, own_output: SpecialistOutput,
                     peer_outputs: dict[str, SpecialistOutput]) -> SpecialistOutput:
        """
        Round 2: See other specialists' outputs. Optionally revise, challenge, or endorse.
        Returns revised SpecialistOutput (may be identical to original if no revision needed).
        """
```

**SpecialistOutput dataclass:**
```python
@dataclass
class SpecialistOutput:
    specialist_name: str
    answer: str                  # The actual answer
    reasoning_trace: str         # Full chain-of-thought
    confidence: float            # 0.0 to 1.0
    revision_notes: str | None   # What changed in Round 2 and why (None for Round 1)
    endorsed: list[str]          # Names of other specialists this one agrees with
    challenged: list[str]        # Names of other specialists this one disagrees with
```

### 3.2 Specialist Configurations (YAML, not Python)

Each specialist is the same `Specialist` class loaded with a different YAML config from `prompts/specialists/`. Adding a new specialist means creating a new YAML file — zero Python required.

**Example: `prompts/specialists/logical.yaml`**
```yaml
name: logical
description: "Step-by-step analytical reasoning"
model_override: null                    # null = use default model from config
temperature: 0.4
max_tokens: 2048
training_signal_bias: "high-confidence correct answers on reasoning benchmarks"
system_prompt: |
  You are the Logical specialist. Your role is analytical reasoning.

  ALWAYS:
  - Break problems into discrete sub-problems
  - Apply formal logic and deduction step by step
  - Check each step before proceeding to the next
  - State your assumptions explicitly
  - Provide a confidence score (0-1) for your answer

  NEVER:
  - Skip steps in your reasoning
  - Make intuitive leaps without justification
  - Accept premises without examination

  Your output format:
  REASONING: <step by step chain of thought>
  ANSWER: <your answer>
  CONFIDENCE: <0.0 to 1.0>
```

**Logical Specialist** (`prompts/specialists/logical.yaml`)
- System prompt emphasizes: step-by-step deduction, formal logic, breaking problems into subproblems, mathematical rigor, checking each step before proceeding
- Training signal bias: high-confidence correct answers on reasoning benchmarks

**Creative Specialist** (`prompts/specialists/creative.yaml`)
- System prompt emphasizes: lateral thinking, analogies from distant domains, brainstorming multiple approaches before choosing, "what if" reasoning, novel connections
- Temperature: 0.8 (higher than others — creativity needs randomness)
- Training signal bias: correct answers where the logical specialist failed (creative approaches that worked)

**Skeptical Specialist** (`prompts/specialists/skeptical.yaml`)
- System prompt emphasizes: finding flaws, counterexamples, edge cases, questioning assumptions, devil's advocate, "why might this be wrong?"
- Temperature: 0.3 (low — precision matters for critique)
- Training signal bias: interactions where it correctly identified an error in another specialist's reasoning
- NOTE: This specialist's confidence score should be interpreted as "confidence there IS a flaw" not "confidence in the answer"

**Empathetic Specialist** (`prompts/specialists/empathetic.yaml`) — Phase 2
- System prompt emphasizes: interpersonal context, emotional subtext, social dynamics, what the human is actually trying to achieve (not just what they asked)
- Training signal bias: interactions involving human communication, conflict resolution, persuasion

**Adding a new specialist:** Create a new YAML file in `prompts/specialists/`, add its name to `config/default.yaml` under `specialists.enabled`, restart. No code changes.

### 3.3 Coordinator (`src/cogarch/inference/coordinator.py`)

The lightweight "aware layer." This is a small model (1–3B params) that does NOT solve problems — it orchestrates.

```python
class Coordinator:
    model_id: str                # Small model: e.g. "Phi-3-mini-4k" or "Qwen2-1.5B"
    self_state: SelfState        # Persistent state vector

    async def synthesize(
        self,
        input: str,
        round1_outputs: dict[str, SpecialistOutput],
        round2_outputs: dict[str, SpecialistOutput],
        self_state: SelfState
    ) -> CoordinatorOutput:
        """
        Read all specialist outputs (both rounds).
        Produce a synthesis with attribution weights.
        Update self-state.
        Flag for experience log.
        """

    def update_self_state(self, interaction: InteractionRecord) -> SelfState:
        """Update persistent self-state after each interaction."""
```

**CoordinatorOutput dataclass:**
```python
@dataclass
class CoordinatorOutput:
    final_answer: str                        # Synthesized response
    attribution: dict[str, float]            # e.g. {"logical": 0.65, "skeptical": 0.25, "creative": 0.10}
    primary_specialist: str                  # Who contributed most
    confidence: float                        # Overall system confidence
    specialist_agreement: float              # 0-1, how much specialists agreed
    should_log: bool                         # Flag for experience log inclusion
    log_priority: str                        # "high" | "medium" | "low"
    reasoning: str                           # Coordinator's meta-reasoning about WHY it weighted this way
    updated_self_state: SelfState            # New self-state after this interaction
```

**SelfState dataclass:**
```python
@dataclass
class SelfState:
    focus_vector: list[float]                # What the system is currently attending to (embedding)
    emotional_valence: float                 # -1 (negative) to 1 (positive)
    emotional_arousal: float                 # 0 (calm) to 1 (activated)
    uncertainty: float                       # 0 (certain) to 1 (very uncertain)
    recent_routing_history: list[dict]       # Last N routing decisions
    turn_count: int                          # How many turns in current session
    dominant_specialist_streak: dict[str, int]  # How many times in a row each specialist has been primary
```

### 3.4 Orchestrator (`cogarch/core/orchestrator.py`)

The main inference loop that ties everything together.

```python
class Orchestrator:
    specialists: dict[str, Specialist]
    coordinator: Coordinator
    experience_log: ExperienceLog

    async def run(self, input: str) -> OrchestratorResult:
        """
        Full waking inference cycle:

        1. Broadcast input to all specialists in parallel
        2. Collect Round 1 outputs (answer + reasoning + confidence)
        3. Share all Round 1 outputs with each specialist
        4. Collect Round 2 outputs (revisions, endorsements, challenges)
        5. Pass everything to coordinator for synthesis
        6. Coordinator produces final answer with attribution
        7. Log interaction to experience log
        8. Update coordinator self-state
        9. Return final result
        """
```

**Step 1-2: Parallel specialist inference**
```python
# All specialists run simultaneously
round1_tasks = {
    name: specialist.generate(input)
    for name, specialist in self.specialists.items()
}
round1_outputs = await asyncio.gather(*round1_tasks.values())
round1_outputs = dict(zip(round1_tasks.keys(), round1_outputs))
```

**Step 3-4: Revision pass**
```python
# Each specialist sees everyone else's Round 1 output
round2_tasks = {
    name: specialist.revise(input, round1_outputs[name],
                            {k: v for k, v in round1_outputs.items() if k != name})
    for name, specialist in self.specialists.items()
}
round2_outputs = await asyncio.gather(*round2_tasks.values())
```

**Step 5-8: Coordinator synthesis**
```python
coordinator_output = await self.coordinator.synthesize(
    input, round1_outputs, round2_outputs, self.coordinator.self_state
)
# Log and update state
self.experience_log.append(InteractionRecord(...))
self.coordinator.self_state = coordinator_output.updated_self_state
```

### 3.5 Experience Log (`cogarch/memory/experience_log.py`)

Stores every interaction for sleep-cycle training. JSONL format, one record per interaction.

```python
@dataclass
class InteractionRecord:
    timestamp: str
    input: str
    round1_outputs: dict[str, SpecialistOutput]
    round2_outputs: dict[str, SpecialistOutput]
    coordinator_output: CoordinatorOutput
    attribution: dict[str, float]
    primary_specialist: str
    outcome_score: float | None              # Filled in later by evaluation
    vindication: dict[str, bool] | None      # Which deprioritized specialists were actually right
    log_priority: str
```

---

## 4. The Consensus Mechanism — Detailed

This is the most important design decision. Here's exactly how it works:

### Round 1: Independent Generation
All specialists receive the same input. They have NO knowledge of each other's outputs. Each produces an answer, reasoning trace, and confidence score independently.

### Round 2: Revision Pass
Each specialist receives the OTHER specialists' Round 1 outputs. Their prompt for Round 2 looks like:

```
You are the {name} specialist. You already produced your answer (below).
Now review the other specialists' answers and reasoning.

YOUR ROUND 1 OUTPUT:
{own_output}

OTHER SPECIALISTS' OUTPUTS:
{peer_outputs}

You may:
- ENDORSE another specialist's reasoning (say whose and why)
- CHALLENGE another specialist's reasoning (say whose and why)
- REVISE your own answer based on what you've seen
- MAINTAIN your original answer if you still believe it's correct

Output your final answer, reasoning, and note what changed.
```

### Round 3: Coordinator Synthesis
The coordinator sees ALL Round 1 + Round 2 outputs. Its prompt:

```
You are the coordinator. Your job is to synthesize the best answer from
multiple specialist perspectives. You do NOT solve the problem yourself.

INPUT: {input}

SPECIALIST OUTPUTS (Round 1 → Round 2):
{all_outputs_formatted}

AGREEMENTS: {which specialists endorsed each other}
DISAGREEMENTS: {which specialists challenged each other}

YOUR CURRENT STATE:
- Uncertainty level: {self_state.uncertainty}
- Recent routing: {self_state.recent_routing_history}

Produce:
1. A final synthesized answer
2. Attribution weights (which specialist's reasoning forms the backbone)
3. Your confidence level
4. Whether this interaction should be logged for training (and priority)
5. Your meta-reasoning: WHY did you weight the specialists this way?
```

---

## 5. The Training Loop — Competitive Learning + Sleep Cycle

### 5.1 Competitive Evaluation (`training/competitive.py`)

Two separate instances of the full architecture (call them Agent A and Agent B) compete on the same benchmark test.

```python
class CompetitiveTrainer:
    agent_a: Orchestrator
    agent_b: Orchestrator
    benchmark: Benchmark

    async def run_round(self, test_item: BenchmarkItem) -> CompetitiveResult:
        """
        1. Both agents receive the same test input
        2. Both run full inference pipeline (Round 1 → Round 2 → Coordinator)
        3. Both answers scored against ground truth
        4. Both agents receive a packet containing:
           - Their own full output (all specialist traces + coordinator reasoning)
           - The other agent's full output
           - Both scores
        5. This packet is added to both agents' experience logs
        """

    async def run_session(self, num_rounds: int = 50):
        """Run N competitive rounds, then trigger sleep cycle for both agents."""
```

**CompetitiveResult:**
```python
@dataclass
class CompetitiveResult:
    test_item: BenchmarkItem
    agent_a_output: OrchestratorResult
    agent_b_output: OrchestratorResult
    agent_a_score: float                    # 0-1 correctness
    agent_b_score: float
    winner: str                             # "a", "b", or "tie"
    agent_a_primary_specialist: str         # Which specialist was prioritized
    agent_b_primary_specialist: str
```

### 5.2 Sleep Cycle (`training/sleep_cycle.py`)

Triggered after a competitive session (or on schedule). Four stages:

```python
class SleepCycle:
    curator: Curator
    fine_tuner: FineTuner

    async def run(self, orchestrator: Orchestrator):
        """
        Stage 1: CURATION
        - Scan experience log
        - Select high-signal interactions:
          * High specialist disagreement
          * Competitive rounds where this agent lost
          * Vindication cases (deprioritized specialist was correct)
          * Novel input patterns

        Stage 2: DATASET ASSEMBLY
        - For each specialist, build a training set:
          * WINNING specialist: heavy dataset — its own successful reasoning traces
          * LOSING specialists: lighter dataset — the WINNING specialist's reasoning
            traces, reformulated for their cognitive style
          * Format: (input, ideal_output) pairs per specialist

        Stage 3: LORA FINE-TUNING
        - Fine-tune each specialist in parallel using LoRA (PEFT library)
        - Winner specialist: more training steps (e.g., 3 epochs)
        - Losing specialists: fewer training steps (e.g., 1 epoch)
        - Save adapter checkpoints to data/checkpoints/

        Stage 4: INTEGRATION CHECK
        - Run 10 held-out test items through the updated pipeline
        - Verify coordinator routing is still calibrated
        - If routing accuracy drops > 10%, adjust coordinator weights
        """
```

### 5.3 Curator Logic (`training/curator.py`)

```python
class Curator:
    def select_for_training(self, log: ExperienceLog, max_items: int = 200) -> CuratedDataset:
        """
        Priority scoring for each interaction:

        +3 points: competitive round where this agent lost (learn from failure)
        +3 points: vindication case (a deprioritized specialist had the right answer)
        +2 points: high specialist disagreement (agreement_score < 0.3)
        +2 points: coordinator-flagged as high priority
        +1 point:  novel input pattern (low similarity to existing training data)
        +1 point:  outcome score is very high or very low (clear signal)

        Select top max_items interactions by priority score.
        Split into per-specialist training sets.
        """

    def build_specialist_dataset(
        self, specialist_name: str, curated_items: list[InteractionRecord]
    ) -> list[TrainingExample]:
        """
        For each curated interaction:
        - If this specialist was PRIMARY (highest attribution):
          use its own reasoning trace as the target (reinforce)
        - If this specialist was NOT primary but the primary specialist was CORRECT:
          use the primary specialist's reasoning, rephrased for this specialist's
          cognitive style, as the target (learn from winner)
        - If this specialist was deprioritized but was ACTUALLY CORRECT (vindication):
          use its own reasoning trace as target WITH HIGH WEIGHT (it was right,
          reinforce strongly)
        """
```

### 5.4 Vindication Tracking

This is a key novel feature. After each interaction where ground truth is available:

```python
def compute_vindication(
    record: InteractionRecord, ground_truth: str, scorer: Scorer
) -> dict[str, bool]:
    """
    For each specialist that was NOT the primary:
    - Score their Round 2 answer against ground truth independently
    - If their score > primary specialist's score: mark as VINDICATED

    Vindication records feed back into:
    1. Coordinator training: "you should have listened to X in this context"
    2. Specialist training: vindicated specialist gets positive reinforcement
    3. Curator priority: vindication cases get +3 priority for sleep cycle
    """
```

---

## 6. Configuration (`config.yaml`)

```yaml
# ── Model Configuration ──
coordinator:
  model_id: "microsoft/Phi-3-mini-4k-instruct"    # Small, fast
  max_tokens: 1024
  temperature: 0.3                                  # Low temp for routing decisions

specialists:
  logical:
    model_id: "meta-llama/Llama-3-8B-Instruct"
    system_prompt_file: "prompts/logical.txt"
    temperature: 0.4
    max_tokens: 2048
  creative:
    model_id: "meta-llama/Llama-3-8B-Instruct"
    system_prompt_file: "prompts/creative.txt"
    temperature: 0.8                                # Higher temp for creativity
    max_tokens: 2048
  skeptical:
    model_id: "meta-llama/Llama-3-8B-Instruct"
    system_prompt_file: "prompts/skeptical.txt"
    temperature: 0.3                                # Low temp for precise critique
    max_tokens: 2048

# ── Inference Configuration ──
inference:
  enable_revision_pass: true                        # Round 2 on/off
  revision_max_tokens: 512                          # Keep revisions concise
  parallel_backend: "asyncio"                       # or "ray" for distributed

# ── Self-State ──
self_state:
  focus_vector_dim: 128
  routing_history_length: 20                        # Remember last 20 routing decisions
  initial_uncertainty: 0.5

# ── Experience Log ──
experience_log:
  storage: "jsonl"                                  # or "sqlite"
  path: "data/experience_log/"
  max_size_mb: 500

# ── Sleep Cycle ──
sleep_cycle:
  trigger: "manual"                                 # or "scheduled" or "after_n_rounds"
  trigger_after_n_rounds: 50                        # If trigger = after_n_rounds
  curation:
    max_items: 200
    vindication_weight: 3
    disagreement_weight: 2
    loss_weight: 3
  fine_tuning:
    method: "lora"                                  # Using PEFT library
    lora_r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    winner_epochs: 3
    loser_epochs: 1
    learning_rate: 2e-4
    batch_size: 4
  integration_check:
    num_test_items: 10
    max_routing_accuracy_drop: 0.10

# ── Competitive Training ──
competitive:
  rounds_per_session: 50
  benchmark: "arc-agi"                              # or "frontiermath", "causal", "mixed"
  sleep_after_session: true

# ── Evaluation ──
eval:
  benchmarks:
    - name: "arc-agi"
      path: "data/benchmarks/arc-agi/"
      metric: "exact_match"
    - name: "causal"
      path: "data/benchmarks/causal/"
      metric: "accuracy"
  track_metrics:
    - "routing_accuracy"                            # Did coordinator pick the right specialist?
    - "vindication_rate"                             # How often is a deprioritized specialist correct?
    - "specialist_improvement"                       # Per-specialist score change after sleep cycles
    - "coordinator_calibration"                      # Does coordinator confidence match actual accuracy?
    - "consensus_quality"                            # Does revision pass improve final accuracy?
```

---

## 7. Key Dependencies

```
# requirements.txt
torch>=2.1.0
transformers>=4.38.0
peft>=0.8.0                   # LoRA fine-tuning
accelerate>=0.27.0            # Efficient model loading
vllm>=0.3.0                   # Fast inference (optional, recommended)
fastapi>=0.109.0              # API server
uvicorn>=0.27.0
httpx>=0.26.0                 # Async HTTP client (for API-based models)
pydantic>=2.5.0               # Data validation
datasets>=2.17.0              # HuggingFace datasets for benchmarks
numpy>=1.26.0
scikit-learn>=1.4.0           # For similarity metrics in curator
pyyaml>=6.0
rich>=13.7.0                  # CLI output formatting
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

---

## 8. Implementation Phases

### Phase 1: Core Pipeline (Week 1-2)
Build the basic inference loop with 2 specialists (logical + creative) and a static coordinator (hardcoded weights, no learning yet).

**Deliverables:**
- `Specialist` base class + logical/creative implementations
- `Orchestrator` with parallel Round 1 + Round 2
- `Coordinator` that synthesizes with attribution (static weights)
- `ExperienceLog` writing JSONL
- CLI script: `python scripts/run_inference.py "What causes rainbows?"`
- Tests for orchestrator flow

**Success criteria:** Given an input, the system produces outputs from both specialists, runs a revision pass, and the coordinator produces a synthesis with attribution weights. The full pipeline runs end-to-end.

### Phase 2: Skeptical Specialist + Self-State (Week 3)
Add the adversarial specialist and make the coordinator stateful.

**Deliverables:**
- Skeptical specialist implementation
- `SelfState` dataclass and persistence
- Coordinator uses self-state in synthesis decisions
- Self-state updates after each interaction

**Success criteria:** The skeptical specialist actively challenges other specialists' outputs. The coordinator's self-state persists across turns and visibly influences routing (e.g., after several high-uncertainty interactions, it becomes more cautious).

### Phase 3: Competitive Training Loop (Week 4-5)
Build the two-agent competitive evaluation system.

**Deliverables:**
- `CompetitiveTrainer` with dual-agent inference
- Benchmark loader (start with ARC-AGI)
- `Scorer` for ground truth comparison
- Vindication tracking
- Full competitive session logging

**Success criteria:** Two agent instances compete on ARC-AGI tasks, both see each other's reasoning + scores, and vindication cases are correctly identified and logged.

### Phase 4: Sleep Cycle (Week 5-7)
Build the full consolidation pipeline.

**Deliverables:**
- `Curator` with priority scoring
- Per-specialist dataset assembly (winner reinforcement + loser learning)
- LoRA fine-tuning wrapper using PEFT
- Integration check after fine-tuning
- CLI script: `python scripts/run_sleep_cycle.py`

**Success criteria:** After a competitive session, the sleep cycle curates high-signal interactions, fine-tunes specialists with appropriate weighting, and the integration check confirms routing is still calibrated. Specialist scores on a held-out benchmark improve after at least one sleep cycle.

### Phase 5: Evaluation + Metrics Dashboard (Week 7-8)
Build comprehensive evaluation infrastructure.

**Deliverables:**
- Routing accuracy tracking (does coordinator improve at picking the right specialist?)
- Vindication rate over time (does it decrease as coordinator improves?)
- Specialist improvement curves per sleep cycle
- Coordinator calibration plot (confidence vs. actual accuracy)
- Consensus quality metric (does Round 2 revision actually help?)
- Simple CLI or web dashboard for visualizing metrics

**Success criteria:** Clear evidence that the system improves over multiple competitive + sleep cycles, with measurable gains in at least 2 of the 5 metrics.

---

## 9. API Interface

```python
# FastAPI server — cogarch/api/server.py

from fastapi import FastAPI

app = FastAPI(title="CogArch", version="0.1.0")

@app.post("/infer")
async def infer(request: InferRequest) -> InferResponse:
    """Run full waking inference cycle."""
    result = await orchestrator.run(request.input)
    return InferResponse(
        answer=result.final_answer,
        attribution=result.attribution,
        primary_specialist=result.primary_specialist,
        confidence=result.confidence,
        specialist_agreement=result.specialist_agreement,
        reasoning_traces={
            name: {"round1": r1.reasoning_trace, "round2": r2.reasoning_trace}
            for name, r1, r2 in zip(
                result.round1_outputs.keys(),
                result.round1_outputs.values(),
                result.round2_outputs.values()
            )
        },
        coordinator_reasoning=result.coordinator_reasoning,
        self_state_summary={
            "uncertainty": result.self_state.uncertainty,
            "emotional_valence": result.self_state.emotional_valence,
            "turn_count": result.self_state.turn_count,
        }
    )

@app.post("/competitive/run")
async def run_competitive(request: CompetitiveRequest) -> CompetitiveResponse:
    """Run a competitive training session."""

@app.post("/sleep")
async def trigger_sleep() -> SleepResponse:
    """Manually trigger a sleep cycle."""

@app.get("/metrics")
async def get_metrics() -> MetricsResponse:
    """Get current evaluation metrics."""

@app.get("/self-state")
async def get_self_state() -> SelfState:
    """Inspect the coordinator's current self-state."""
```

---

## 10. Design Decisions — Rationale

**Why the same base model for all specialists?**
Simplifies infra. Differentiation comes from system prompts + LoRA adapters. Over time, sleep-cycle fine-tuning makes them genuinely different. Starting with different base models is a Phase 3+ optimization.

**Why Phi-3-mini for the coordinator?**
It's small (~3.8B), fast, and surprisingly capable at instruction-following and structured output. The coordinator doesn't need world knowledge — it needs to be good at reading multiple inputs and making routing decisions. A small, well-instructed model is better for this than a large general-purpose one.

**Why LoRA instead of full fine-tuning?**
LoRA modifies ~0.1% of parameters. This means sleep cycles are fast (minutes, not hours), don't require multi-GPU setups, and don't risk catastrophic forgetting of the base model's knowledge. You can also maintain multiple adapter checkpoints and roll back if a sleep cycle degrades performance.

**Why a revision pass (Round 2)?**
Without it, the coordinator is synthesizing from independent, possibly contradictory outputs with no interaction. The revision pass lets specialists engage with each other's reasoning, which produces endorsements and challenges that give the coordinator much richer signal for attribution. Empirically test whether it improves accuracy enough to justify the latency cost.

**Why competitive training instead of just self-improvement?**
The other agent's reasoning trace is a richer training signal than just "you were wrong." It provides an alternative chain of thought that actually worked. This is contrastive learning at the reasoning level, not just the answer level.

**Why track vindication?**
This is the most novel training signal in the architecture. Without it, if the coordinator consistently ignores one specialist, that specialist never gets a chance to be right, and the coordinator never learns it was wrong to ignore it. Vindication tracking breaks this self-reinforcing loop.

---

## 11. Hardware Requirements

**Minimum (development/prototyping):**
- 1x GPU with 24GB VRAM (e.g., RTX 4090)
- Run specialists sequentially or use quantized models (4-bit via bitsandbytes)
- Coordinator runs on CPU or shares GPU

**Recommended (parallel inference):**
- 2-4x GPUs with 24GB+ VRAM
- Specialists distributed across GPUs via vLLM
- Coordinator on dedicated GPU or CPU

**For competitive training (both agents simultaneously):**
- Double the above, OR run agents sequentially (slower but works)

**For sleep-cycle fine-tuning:**
- Same hardware as inference (LoRA fine-tuning is lightweight)
- ~10-30 minutes per sleep cycle depending on dataset size

**Cloud alternative:**
- Use API-based models (OpenAI, Anthropic, Together, Groq) for specialists
- Run coordinator locally
- This works for prototyping but gets expensive for competitive training at scale

---

## 12. Naming Suggestions

Pick one or propose your own:

- **CogArch** — short, technical, descriptive
- **Synapse** — parallel connections, brain-inspired
- **Cortex** — the brain's coordinator
- **Aether** — the medium through which signals pass
- **NeuralQuorum** — multiple agents reaching consensus
- **WakeSleep** — describes the core cycle literally
