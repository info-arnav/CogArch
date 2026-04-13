# Contributing to CogArch

Thank you for your interest in contributing to CogArch. This project is a research framework for parallel cognitive architecture using multiple LLMs.

## Development Setup

**Prerequisites:**
- Python 3.10+
- pip 23+
- Git
- OpenAI API key (for GPT-4o/GPT-4o-mini)

**Clone and Install:**

```bash
git clone https://github.com/YOUR_USERNAME/CogArch.git
cd CogArch

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -e ".[dev]"

cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Install pre-commit hooks
pre-commit install
```

**Project Structure:**

```
CogArch/
├── src/              # Core library code
│   ├── models/      # Data structures
│   ├── inference/   # Specialists, coordinator, orchestrator
│   ├── training/    # Sleep cycle, competitive learning
│   ├── memory/      # Experience log, persistence
│   └── eval/        # Benchmarks, metrics
├── cli/             # Command-line interface
├── prompts/         # Specialist configs (YAML)
├── tests/           # Unit and integration tests
├── config/          # Configuration files
└── data/            # Runtime data (gitignored)
```

**Running Tests:**

```bash
pytest                                    # Run all tests
pytest tests/unit/test_specialist.py     # Run specific test
pytest --cov=src --cov-report=html       # With coverage
```

**Code Quality:**

```bash
black src/ tests/ cli/        # Format code
ruff check src/ tests/ cli/   # Lint
mypy src/                     # Type checking
make lint                     # All checks (CI will run this)
```

---

## Contribution Workflow

**1. Find or Create an Issue**
- Browse [open issues](https://github.com/info-arnav/CogArch/issues)
- Look for `good-first-issue` or `help-wanted` labels
- For new features, open an issue first to discuss

**2. Create a Branch**

```bash
git checkout -b feat/your-feature-name
# OR
git checkout -b fix/bug-description
```

**3. Make Changes**
- Follow existing code style (Black formatting, type hints)
- Add tests for new functionality
- Update documentation if needed
- Keep commits atomic and well-described

**4. Test Locally**

```bash
pytest                # Run tests
make lint             # Check code quality
```

**5. Submit a Pull Request**
- Push your branch to your fork
- Open a PR against `main`
- Fill out the PR template
- Link related issues (e.g., "Fixes #123")
- Wait for code review

**6. Code Review**
- Address reviewer feedback
- Keep discussions focused and constructive
- Update your PR as needed
- Once approved, a maintainer will merge

---

## Code Style Guidelines

**Python Style:**
- Black for formatting (88 char line length)
- Type hints for all function signatures
- Docstrings for public APIs (Google style)
- Pydantic for data models

**Example:**

```python
from typing import Optional
from pydantic import BaseModel, Field

class SpecialistOutput(BaseModel):
    """Output from a single specialist's reasoning process.
    
    Attributes:
        specialist_name: Identifier for this specialist
        answer: The specialist's answer to the input
        reasoning_trace: Full chain-of-thought reasoning
        confidence: 0-1 confidence score
    """
    
    specialist_name: str = Field(..., description="Specialist identifier")
    answer: str = Field(..., description="Generated answer")
    reasoning_trace: str = Field(..., description="Step-by-step reasoning")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
```

**YAML Configs:**
- Consistent indentation (2 spaces)
- Comments explaining non-obvious settings
- Follow existing specialist config patterns

**Tests:**
- One test file per source file
- Descriptive test names: `test_specialist_generates_with_high_confidence`
- Use fixtures for common setup
- Mock external dependencies (LLM calls)

---

## Reporting Bugs

Use the [Bug Report template](https://github.com/info-arnav/CogArch/issues/new?template=bug_report.md). Include Python version, OS, error traceback, and reproduction steps.

---

## Requesting Features

Use the [Feature Request template](https://github.com/info-arnav/CogArch/issues/new?template=feature_request.md). Describe the problem, proposed solution, and alternatives considered.

---

## Documentation Contributions

Documentation improvements are always welcome - README, architecture docs, API documentation, tutorials, and troubleshooting guides.

---

## Research Contributions

This is a research project. If you have ideas or want to discuss architectural improvements, open a [Discussion](https://github.com/info-arnav/CogArch/discussions) or issue.

---

## Community Guidelines

- Be respectful - see our [Code of Conduct](CODE_OF_CONDUCT.md)
- Be patient - maintainers are volunteers
- Be constructive - focus on solutions
- Be collaborative - help others, share knowledge

---

Thank you for helping build CogArch!
