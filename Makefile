.PHONY: help install install-dev install-all test lint format clean run-api infer compete sleep

help:
	@echo "CogArch Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install core dependencies"
	@echo "  make install-dev   Install with dev dependencies"
	@echo "  make install-all   Install everything (training + dev)"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run all tests"
	@echo "  make lint          Run linters (ruff + mypy)"
	@echo "  make format        Format code with black"
	@echo "  make clean         Remove build artifacts"
	@echo ""
	@echo "Usage:"
	@echo "  make infer         Run inference (pass QUERY='...')"
	@echo "  make compete       Run competitive training"
	@echo "  make sleep         Run sleep cycle"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install -r requirements-dev.txt
	pre-commit install

install-all:
	pip install -e ".[all]"
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

lint:
	ruff check src/ cli/ tests/
	mypy src/ cli/

format:
	black src/ cli/ tests/

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov/

infer:
	@if [ -z "$(QUERY)" ]; then \
		echo "Usage: make infer QUERY='Your question here'"; \
	else \
		python -m cli.main infer "$(QUERY)"; \
	fi

compete:
	python -m cli.main compete --rounds 50 --benchmark arc-agi

sleep:
	python -m cli.main sleep --curate --finetune
