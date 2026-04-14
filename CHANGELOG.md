# Changelog

All notable changes to CogArch are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
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
