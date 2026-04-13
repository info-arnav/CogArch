# CogArch — Parallel Cognitive Architecture

<p align="center">
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-0F766E?style=flat-square" alt="MIT license" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python 3.10+" /></a>
  <a href="https://github.com/info-arnav/CogArch/issues"><img src="https://img.shields.io/github/issues/info-arnav/CogArch?style=flat-square" alt="Open issues" /></a>
  <a href="https://github.com/info-arnav/CogArch/stargazers"><img src="https://img.shields.io/github/stars/info-arnav/CogArch?style=flat-square" alt="GitHub stars" /></a>
</p>

A Python framework for machine consciousness through parallel specialist LLMs, competitive learning, and sleep-cycle fine-tuning.

**Core concept:** A smaller coordinator model (GPT-4o-mini) learns to orchestrate multiple larger specialist models (GPT-4o), inspired by how the brain's prefrontal cortex coordinates specialized regions.

---

## What This Is

CogArch is a research framework where:

- Multiple specialist LLMs run in parallel on every input (logical, creative, skeptical, empathetic)
- Specialists share outputs and engage in a revision pass (consensus deliberation)
- A lightweight coordinator model synthesizes their perspectives with attribution weights
- Continuous improvement through competitive training and sleep-cycle fine-tuning
- Self-state tracking - the system maintains awareness of its own uncertainty, focus, and routing patterns

**Novel contributions:**
- Vindication tracking: deprioritized specialists get credit when they were actually right
- Competitive learning: two agent instances compete on benchmarks and learn from each other's reasoning traces
- Sleep-cycle consolidation: LoRA fine-tuning of specialists based on curated high-signal interactions

---

## Current Status

Pre-alpha / Active Development

**Implemented:**
- Complete architecture specification
- Project structure and configuration
- Specialist personality configs (YAML-based)
- OpenAI backend integration (GPT-4o specialists, GPT-4o-mini coordinator)

**In Progress:**
- Phase 1: Core inference pipeline (specialists, coordinator, orchestrator)
- Phase 2: Competitive training system  
- Phase 3: Sleep-cycle fine-tuning
- Phase 4: Evaluation metrics and benchmarks

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to help.

---

## Quick Setup

```bash
git clone https://github.com/info-arnav/CogArch.git
cd CogArch
python -m venv venv && source venv/bin/activate
make install-dev

cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

---

## Architecture Overview

### Inference Pipeline

```mermaid
graph TD
    Input[Input Query] --> Broadcast{Orchestrator}
    
    Broadcast -->|Round 1| L[Logical Specialist<br/>Analytical reasoning]
    Broadcast -->|Round 1| C[Creative Specialist<br/>Novel approaches]
    Broadcast -->|Round 1| S[Skeptical Specialist<br/>Critical analysis]
    Broadcast -->|Round 1| E[Empathetic Specialist<br/>Human-centered view]
    
    L --> Share[Share All Outputs]
    C --> Share
    S --> Share
    E --> Share
    
    Share -->|Round 2| L2[Logical<br/>Revise answer]
    Share -->|Round 2| C2[Creative<br/>Revise answer]
    Share -->|Round 2| S2[Skeptical<br/>Revise answer]
    Share -->|Round 2| E2[Empathetic<br/>Revise answer]
    
    L2 --> Coord[Coordinator<br/>Small Model]
    C2 --> Coord
    S2 --> Coord
    E2 --> Coord
    
    Coord --> Output[Final Answer<br/>+ Attribution<br/>+ Self-State]
    
    style Input fill:#e1f5ff
    style Output fill:#d4edda
    style Coord fill:#fff3cd
    style Share fill:#f8f9fa
```

### Competitive Learning

```mermaid
sequenceDiagram
    participant B as Benchmark
    participant A as Agent A
    participant C as Agent B
    participant J as Judge
    
    B->>A: Same question
    B->>C: Same question
    
    Note over A: Full inference<br/>(4 specialists + coordinator)
    Note over C: Full inference<br/>(4 specialists + coordinator)
    
    A->>J: Answer A + reasoning traces
    C->>J: Answer B + reasoning traces
    
    J->>J: Score both against ground truth
    
    J->>A: Your score + opponent's reasoning
    J->>C: Your score + opponent's reasoning
    
    Note over A,C: Both log interaction with cross-agent insights
    
    rect rgb(240, 248, 255)
        Note over A,C: After N rounds: Sleep Cycle<br/>Winners teach, losers learn
    end
```

### Wake/Sleep Cycle

```mermaid
stateDiagram-v2
    [*] --> Awake: System Start
    
    Awake --> Inference: User Query
    Inference --> Logging: Record interaction
    Logging --> Inference: Continue
    
    Logging --> Asleep: Trigger<br/>(N interactions)
    
    state Asleep {
        [*] --> Curate
        Curate --> Assemble: High-signal<br/>interactions
        Assemble --> FineTune: Per-specialist<br/>datasets
        FineTune --> Validate: LoRA adapters
        Validate --> [*]
    }
    
    Asleep --> Awake: Updated specialists
    
    note right of Awake
        Four specialists process in parallel
        Coordinator synthesizes results
        Experience logged to JSONL
    end note
    
    note right of Asleep
        Curator selects valuable interactions
        Specialists fine-tuned on curated data
        Vindication tracking adjusts weights
    end note
```

**Core Components:**
- **Specialists**: Same base model (GPT-4o), different personalities (YAML configs + future LoRA adapters)
- **Coordinator**: Smaller model (GPT-4o-mini) that routes and synthesizes, doesn't solve directly
- **Experience Log**: JSONL append-only record of all interactions for training
- **Sleep Cycle**: Curate → Assemble → Fine-tune → Validate loop
- **Competitive Training**: Two agents compete, cross-learn from reasoning traces

---

## Design Principles

1. Specialists are configs, not code - adding a new specialist means creating a new YAML file
2. Inference and training fully separated - run inference with zero training dependencies
3. Prompts are first-class citizens - versioned, testable, swappable
4. Every data artifact has a clear home - structured data/ directory with schemas

---

## Requirements

- Python 3.10+
- OpenAI API key (GPT-4o for specialists, GPT-4o-mini for coordinator)
- Approximately $0.01-0.05 per inference depending on input length
- No GPU required - runs entirely via OpenAI API

---

## Documentation

- [Architecture Specification](docs/PARALLEL_COGNITIVE_ARCHITECTURE_SPEC.md) - Complete build document
- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [Code of Conduct](CODE_OF_CONDUCT.md) - Community standards
- [Changelog](CHANGELOG.md) - Version history

---

## Contributing

We welcome contributions. High-impact areas:

- Core pipeline implementation (Phase 1: inference loop)
- Benchmark integrations (ARC-AGI, FrontierMath, custom tasks)
- Evaluation metrics (routing accuracy, vindication tracking)
- Competitive training loop
- Documentation and examples

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and development workflow.

---

## License

MIT License - see [LICENSE.md](LICENSE.md) for details.

---

## Acknowledgments

Inspired by:
- Global Workspace Theory (Bernard Baars)
- Attention Schema Theory (Michael Graziano)
- The Wake-Sleep algorithm (Hinton et al.)
- Model-agnostic meta-learning research
- The open source AI community

Built by [Arnav Gupta](https://github.com/info-arnav) • [Report an issue](https://github.com/info-arnav/CogArch/issues)
