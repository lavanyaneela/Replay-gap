# One command a reviewer has to trust: make verify.
# It lints, runs the suite, re-measures every published number, and fails if a document quotes a
# figure the code no longer produces. It must pass from a fresh clone with no network.

PYTHON ?= python3
PACKAGE := replay_gap

.PHONY: help install lint format test fixtures experiments receipts prove-receipts verify evidence set-handle clean

help:
	@echo "install      install the package and its development extras"
	@echo "lint         ruff over src, tests, tools and experiments"
	@echo "test         pytest with coverage reports written to reports/"
	@echo "fixtures     regenerate the committed interaction log from its seed"
	@echo "experiments  run E1 to E5 and write docs/experiments/*.json"
	@echo "receipts     re-measure every published number and check it against the prose"
	@echo "prove-receipts  corrupt the README five ways and watch the checker fail each one"
	@echo "verify       lint + test + receipts. This is the command the README promises."
	@echo "evidence     regenerate the diagram, screenshots and demo replay (needs extras)"
	@echo "set-handle   replace the testhandle placeholder: make set-handle HANDLE=you"

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

# The suite runs as its own step and the collector reads its reports, rather than the collector
# shelling out to pytest. A red test should surface as a red test, not as a CalledProcessError
# traceback from inside a metrics script, which costs an hour to attribute.
reports/coverage.json reports/junit.xml:
	$(MAKE) test

receipts: reports/coverage.json
	$(PYTHON) tools/collect_metrics.py --skip-tests
	$(PYTHON) tools/check_numbers.py --strict-reverse

prove-receipts:
	bash tools/prove_receipts.sh

verify: lint test receipts

evidence:
	$(PYTHON) tools/render_diagrams.py
	$(PYTHON) tools/capture_screenshots.py
	$(PYTHON) tools/record_demo.py

# The repository ships with testhandle as a placeholder, because the author's handle is not
# something this repository should guess. One command replaces it everywhere, including in the
# badge URLs, where a wrong name renders as "unknown" and is the first thing a reviewer notices.
set-handle:
	@test -n "$(HANDLE)" || { echo 'usage: make set-handle HANDLE=<your-github-username>'; exit 1; }
	@git grep -l testhandle | xargs sed -i 's/testhandle/$(HANDLE)/g'
	@echo "set the GitHub handle to $(HANDLE) in:"
	@git grep -l "$(HANDLE)/replay-gap" || true

clean:
	rm -rf reports .pytest_cache .ruff_cache .coverage coverage.xml
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
