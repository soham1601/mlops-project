.PHONY: help setup data eda test lint format clean

PY := .venv/bin/python

help:  ## Show available commands
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ -/'

setup:  ## Create .venv and install all dependencies
	test -d .venv || uv venv --python 3.12 .venv
	uv pip install --python $(PY) -r requirements-dev.txt
	uv pip install --python $(PY) -e .

data:  ## Download the raw dataset and write the cleaned CSV
	$(PY) -m mlops_project.data.download
	$(PY) -m mlops_project.data.preprocess

eda:  ## Generate EDA figures and summary in reports/
	$(PY) -m mlops_project.data.eda

test:  ## Run the unit tests
	$(PY) -m pytest

lint:  ## Check code style (same check CI runs)
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:  ## Auto-format the code
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

clean:  ## Remove caches
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
