.PHONY: help install-dev test lint format clean serve smoke pilot dashboard

help:
	@echo "Local:"
	@echo "  make install-dev   Install the package with dev tools"
	@echo "  make test          Run the unit tests"
	@echo "  make lint          ruff + mypy"
	@echo "  make format        black"
	@echo ""
	@echo "GPU server:"
	@echo "  scripts/setup_server.sh   One-time environment setup"
	@echo "  make serve         Start vLLM with runtime LoRA loading"
	@echo "  make smoke         Check vLLM, sandbox, DPO training, LoRA loading, throughput"
	@echo "  make pilot         Launch the pilot runs in the background"
	@echo "  make dashboard     Live dashboard on the server's localhost:8080"
	@echo "                    (view: ssh -L 8080:localhost:8080 root@<server>)"

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -q

lint:
	ruff check src/ cli/ tests/
	mypy src/ cli/

format:
	black src/ cli/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +

serve:
	scripts/serve_vllm.sh

EXP = docker exec -w $(CURDIR) cogarch-exp

smoke:
	$(EXP) python -m cli.main smoke config/smoke.yaml
	$(EXP) python -m cli.main run config/smoke.yaml

pilot:
	scripts/launch_pilot.sh

dashboard:
	docker exec -d -w $(CURDIR) cogarch-exp sh -c "python -m cli.main dashboard > /tmp/dashboard.log 2>&1"
