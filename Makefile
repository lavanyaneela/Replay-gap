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
	@echo "set-handle   point the badge and clone URLs at a real account (HANDLE=you)"

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

# The repository ships with a placeholder account name, because the author's GitHub handle is
# not a fact this repository has, and a wrong name in a badge URL renders as "unknown", which
# is the first thing a reviewer notices.
#
# The file list is explicit rather than a repository wide grep. The first version grepped
# everything, matched this Makefile's own comment, rewrote its own placeholder on the first run,
# and then silently did nothing on every run after that while still printing success.
HANDLE_FILES := README.md
set-handle:
	@test -n "$(HANDLE)" || { echo 'usage: make set-handle HANDLE=<your-github-username>'; exit 1; }
	@grep -q GITHUB_USERNAME $(HANDLE_FILES) || { \
		echo 'no GITHUB_USERNAME placeholder left in $(HANDLE_FILES); nothing to do'; exit 1; }
	@sed -i 's|github.com/GITHUB_USERNAME/|github.com/$(HANDLE)/|g' $(HANDLE_FILES)
	@echo "pointed the badge and clone URLs at github.com/$(HANDLE)/replay-gap"
	@grep -n "github.com/$(HANDLE)/" $(HANDLE_FILES)

clean:
	rm -rf reports .pytest_cache .ruff_cache .coverage coverage.xml
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
