# Changelog

All notable changes to CogArch are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Self-improvement experiment pipeline
  - `experiment` CLI command: `cogarch experiment gsm8k --cycles 5 --fine-tune`
  - ExperimentRunner: baseline → N competitive/fine-tune cycles → final eval
  - BenchmarkSplitter: deterministic train/test split with per-cycle unique partitioning
  - No question reuse across cycles — each cycle trains on a fresh chunk
  - Model swap mechanism: specialist models auto-updated to fine-tuned versions
  - Full experiment report with per-cycle progress and improvement metrics
  - Report saved as JSON to `data/experiments/`
- HuggingFace benchmark loaders (via `datasets` library)
  - GSM8KBenchmark: grade school math with numeric answer extraction
  - MMLUBenchmark: 57-subject multiple choice with letter-based scoring
  - TruthfulQABenchmark: free-form truthfulness with fuzzy matching
  - Factory function `load_benchmark("gsm8k")` for easy instantiation
- Experiment data models (Pydantic): ExperimentConfig, CycleResult, ExperimentReport
- OpenAI fine-tuning integration
  - FineTuner: prepares chat-format JSONL, uploads to OpenAI, creates/monitors fine-tuning jobs
  - `--fine-tune` and `--wait` flags on `sleep` CLI command
  - `finetune-status` CLI command to check job status and cycle manifests
  - Minimum 10-example threshold per specialist before fine-tuning
  - Job manifest saved per cycle at `data/checkpoints/finetune_cycle_N.json`
- Benchmark evaluation suite
  - `bench` CLI command: runs pipeline against benchmark with per-item scoring table
  - Supports exact_match, fuzzy_match, and llm_judge metrics
  - Per-category breakdown in results
- SleepReport now tracks `fine_tune_jobs` (list of OpenAI job IDs)
- Sleep cycle now supports 4 stages: curate → assemble → fine-tune → validate
- Integration tests with mock LLM backend
  - MockBackend and CorrectAnswerBackend in tests/conftest.py
  - Full pipeline E2E: inference, competitive rounds, sleep cycle, benchmark evaluation
- Unit tests: fine-tuner, sleep cycle, splitter, benchmark scoring, experiment models
- Phase 2: Competitive training system
  - CompetitiveTrainer: two agent instances compete on benchmark items
  - JSONL benchmark loader with exact-match and fuzzy-match scoring
  - Vindication tracking on competitive rounds
  - Session summary with win/loss/tie stats
- Phase 3: Sleep cycle system
  - Curator: priority-scored selection of high-signal interactions
  - DatasetBuilder: per-specialist training set assembly (win / learn_from_winner / vindicated signals)
  - SleepCycle orchestrator: curate → build datasets → produce SleepReport
- Phase 4: Evaluation infrastructure
  - Scorer: exact_match, fuzzy_match, LLM-as-judge scoring methods
  - MetricsTracker: routing accuracy, vindication rate, coordinator calibration (ECE), consensus quality
  - Benchmark base class (ABC) for pluggable benchmark loaders
  - BenchmarkItem, CompetitiveResult, TrainingExample, SleepReport data models
- CLI commands: `infer`, `compete`, `sleep`, `dashboard`, `bench`, `experiment`, `finetune-status`
- Full Phase 1 inference pipeline
  - Specialist class with Round 1 (independent) and Round 2 (revision) reasoning
  - Coordinator class with synthesis prompt and self-state tracking
  - Orchestrator tying specialists and coordinator together with asyncio
  - OpenAI backend (AsyncOpenAI adapter for GPT-4o / GPT-4o-mini)
  - YAML-based specialist personality configs (logical, creative, skeptical, empathetic)
  - Coordinator synthesis prompt externalized to prompts/coordinator/synthesis.yaml
  - Experience log (JSONL append-only interaction recording)
  - Pydantic data models for specialist outputs, coordinator outputs, and interaction records
  - Typer CLI with Rich formatted output (attribution table, reasoning display)
- Configuration loader for YAML specialist and coordinator prompts
- Pre-commit hooks (Black, Ruff, MyPy)
- Sample benchmark file (data/benchmarks/sample.jsonl)

### Changed
- Backend simplified to OpenAI-only (removed Ollama, vLLM, Anthropic stubs)
- Removed FastAPI/Uvicorn API server (CLI-only for now)
- Coordinator `backend` parameter is now required (no default None)
- Ruff config updated to use `[tool.ruff.lint]` section

### Fixed
- All Black formatting issues resolved
- All Ruff lint errors resolved (unused imports, import ordering)
- All mypy type errors resolved across 21 source files

---

## [0.1.0] - 2026-04-14

### Added
- Project initialization
- Architecture specification
- Core project structure
