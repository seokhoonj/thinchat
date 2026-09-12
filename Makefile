# Run locally everything CI runs. Requires the dev extras: uv pip install -e ".[dev]"
.PHONY: check test lint types

check: test lint types

test:
	pytest -q

lint:
	ruff check src tests

types:
	mypy
