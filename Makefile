# One command a reviewer has to trust: make verify.
# It lints, runs the suite, re-measures every published number, and fails if a document quotes a
# figure the code no longer produces. It must pass from a fresh clone with no network.

PYTHON ?= python3
PACKAGE := replay_gap

.PHONY: help install lint format test coverage fixtures experiments receipts verify evidence clean

help:
	@echo "install      install the package and its development extras"
	@echo "lint         ruff over src, tests, tools and experiments"
	@echo "test         pytest with coverage reports written to reports/"
	@echo "fixtures     regenerate the committed interaction log from its seed"
	@echo "experiments  run E1 to E5 and write docs/experiments/*.json"
	@echo "receipts     re-measure every published number and check it against the prose"
	@echo "verify       lint + test + receipts. This is the command the README promises."
	@echo "evidence     regenerate the diagram, screenshots and demo replay (needs extras)"

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check src tests tools experiments
	ruff format --check src tests tools experiments

format:
	ruff format src tests tools experiments
	ruff check --fix src tests tools experiments

test:
	mkdir -p reports
	$(PYTHON) -m pytest --junitxml=reports/junit.xml \
		--cov=$(PACKAGE) --cov-report=json:reports/coverage.json \
		--cov-report=xml:reports/coverage.xml --cov-report=term-missing

fixtures:
	$(PYTHON) -m replay_gap.cli generate --force

experiments:
	$(PYTHON) tools/run_experiments.py

receipts:
	$(PYTHON) tools/collect_metrics.py
	$(PYTHON) tools/check_numbers.py

verify: lint test receipts

evidence:
	$(PYTHON) tools/render_diagrams.py
	$(PYTHON) tools/capture_screenshots.py
	$(PYTHON) tools/record_demo.py

clean:
	rm -rf reports .pytest_cache .ruff_cache .coverage coverage.xml
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
