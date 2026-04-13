# CogArch — Parallel Cognitive Architecture

<p align="center">
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-0F766E?style=flat-square" alt="MIT license" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python 3.10+" /></a>
  <a href="https://github.com/info-arnav/CogArch/issues"><img src="https://img.shields.io/github/issues/info-arnav/CogArch?style=flat-square" alt="Open issues" /></a>
  <a href="https://github.com/info-arnav/CogArch/stargazers"><img src="https://img.shields.io/github/stars/info-arnav/CogArch?style=flat-square" alt="GitHub stars" /></a>
</p>

A Python framework for machine consciousness through parallel specialist LLMs, competitive learning, and sleep-cycle fine-tuning.

**Core concept:** A small coordinator model learns to orchestrate multiple larger specialist models, inspired by how the brain's prefrontal cortex coordinates specialized regions.

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

```
┌─────────────────────────────────────────────────────────┐
│                      INPUT QUERY                         │
└──────────────────┬──────────────────────────────────────┘
                   │
    ┌──────────────┼──────────────┬──────────────┐
    ▼              ▼              ▼              ▼
┌─────────┐  ┌──────────┐  ┌───────────┐  ┌────────────┐
│Logical  │  │Creative  │  │Skeptical  │  │Empathetic  │
│Specialist│  │Specialist│  │Specialist │  │Specialist  │
└────┬────┘  └────┬─────┘  └─────┬─────┘  └──────┬─────┘
     │            │              │               │
     │   Round 1: Independent Reasoning          │
     │            │              │               │
     └────────────┴──────┬───────┴───────────────┘
                         │
              ┌──────────▼──────────┐
              │   Share Outputs     │
              └──────────┬──────────┘
                         │
    ┌──────────────┬─────┴─────┬──────────────┐
    ▼              ▼           ▼              ▼
┌─────────┐  ┌──────────┐  ┌───────────┐  ┌────────────┐
│Revise   │  │Revise    │  │Revise     │  │Revise      │
│Endorse  │  │Challenge │  │Maintain   │  │Endorse     │
└────┬────┘  └────┬─────┘  └─────┬─────┘  └──────┬─────┘
     │            │              │               │
     │   Round 2: Revision & Deliberation        │
     │            │              │               │
     └────────────┴──────┬───────┴───────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │   Coordinator    │
              │  (Small Model)   │
              │   Synthesizes    │
              │   + Attributes   │
              └────────┬─────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Final Answer  │
              │  + Attribution │
              │  + Self-State  │
              └────────────────┘
```

**Core Components:**
- Specialists: Same base model, different personalities (defined in YAML configs + LoRA adapters)
- Coordinator: Small model (1-3B params) that routes and synthesizes, not solves
- Experience Log: JSONL append-only record of all interactions for training
- Sleep Cycle: Curate, Assemble, Fine-tune, Validate loop
- Competitive Training: Two agents compete, learn from each other's traces

---

## Design Principles

1. Specialists are configs, not code - adding a new specialist means creating a new YAML file
2. Inference and training fully separated - run inference with zero training dependencies
3. Prompts are first-class citizens - versioned, testable, swappable
4. Every data artifact has a clear home - structured data/ directory with schemas
5. Pluggable backends - swap LLM providers (Ollama, vLLM, OpenAI, Anthropic) without changing logic

---

## Requirements

**API-based (Default):**
- OpenAI API key (GPT-4o for specialists, GPT-4o-mini for coordinator)
- Approximately $0.01-0.05 per inference depending on input length
- No GPU required
- Fast parallel execution

**Local models (Optional):**
- 1x GPU with 24GB VRAM (RTX 4090) for Ollama/vLLM
- 2-4x GPUs for parallel specialist execution
- Free inference but higher upfront cost

---

## Documentation

- [Architecture Specification](PARALLEL_COGNITIVE_ARCHITECTURE_SPEC.md) - Complete build document
- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [Code of Conduct](CODE_OF_CONDUCT.md) - Community standards
- [Changelog](CHANGELOG.md) - Version history

---

## Contributing

We welcome contributions. High-impact areas:

- Core pipeline implementation (Phase 1: inference loop)
- LLM backend adapters (support for more providers)
- Benchmark integrations (ARC-AGI, FrontierMath, custom tasks)
- Evaluation metrics (routing accuracy, vindication tracking)
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
