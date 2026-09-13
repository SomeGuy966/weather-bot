.PHONY: install lint test fetch build train evaluate backtest run clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests
	mypy

test:
	pytest

# Full pipeline: raw data -> features -> model -> evaluation -> backtest -> README tables/figures.
fetch:
	weatherbot fetch

build:
	weatherbot build

train:
	weatherbot train

evaluate:
	weatherbot evaluate

backtest:
	weatherbot backtest

run: fetch build train evaluate backtest

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist src/*.egg-info
