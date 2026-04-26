.PHONY: help install dev-install lint format type test test-unit test-integration test-e2e \
        api index run-naive run-self run-hybrid run-multihop run-all clean

help:
	@echo "Targets:"
	@echo "  install            Install package"
	@echo "  dev-install        Install package + dev extras + pre-commit"
	@echo "  lint               Run ruff"
	@echo "  format             Run black + ruff --fix"
	@echo "  type               Run mypy"
	@echo "  test               Run all tests with coverage"
	@echo "  test-unit          Run unit tests only"
	@echo "  test-integration   Run integration tests"
	@echo "  test-e2e           Run e2e tests (hits real Groq)"
	@echo "  api                Start FastAPI dev server"
	@echo "  index              Build FAISS + BM25 indices"
	@echo "  run-naive|self|hybrid|multihop  Run one RAG over the test set"
	@echo "  run-all            Run all four RAGs sequentially"
	@echo "  clean              Remove caches & build artifacts"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests scripts

format:
	black src tests scripts
	ruff check --fix src tests scripts

type:
	mypy src

test:
	pytest

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration

test-e2e:
	pytest -m e2e

api:
	uvicorn medqa_rag.api.main:app --reload --host 0.0.0.0 --port 8000

index:
	python scripts/build_index.py

run-naive:
	python scripts/run_one_rag.py --rag naive

run-self:
	python scripts/run_one_rag.py --rag self

run-hybrid:
	python scripts/run_one_rag.py --rag hybrid

run-multihop:
	python scripts/run_one_rag.py --rag multihop

run-all:
	python scripts/run_all_experiments.py

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
