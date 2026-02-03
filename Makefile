.PHONY: build build-cli clean test test-parity test-all lint lint-swift lint-python format format-fix format-fix-swift format-fix-python setup setup-swift setup-python setup-hooks benchmark benchmark-compare parity-report help

# Default target
.DEFAULT_GOAL := help

help:
	@echo "HTDemucs CoreML - Development Commands"
	@echo ""
	@echo "Build:"
	@echo "  make build        Build Swift package (release)"
	@echo "  make build-cli    Build CLI tool"
	@echo "  make clean        Clean build artifacts"
	@echo ""
	@echo "Test:"
	@echo "  make test         Run Swift unit tests"
	@echo "  make test-parity  Run Python parity tests"
	@echo "  make test-all     Run all tests"
	@echo ""
	@echo "Quality:"
	@echo "  make lint         Check code style (Swift + Python)"
	@echo "  make format       Check formatting (no changes)"
	@echo "  make format-fix   Apply formatting fixes"
	@echo ""
	@echo "Setup:"
	@echo "  make setup        Install all dependencies"
	@echo "  make setup-hooks  Install pre-commit hooks"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make benchmark    Run performance benchmark"
	@echo "  make benchmark-compare  Compare to baseline"
	@echo "  make parity-report      Generate HTML parity report"

# Build section
build:
	swift build -c release

build-cli:
	swift build -c release --product htdemucs-cli

clean:
	swift package clean
	rm -rf .build

# Test section - uses venv at tests/parity/venv
PARITY_VENV := tests/parity/venv
PARITY_PYTHON := $(PARITY_VENV)/bin/python
PARITY_PIP := $(PARITY_VENV)/bin/pip
PARITY_PYTEST := $(PARITY_VENV)/bin/pytest

test:
	swift test

test-parity: $(PARITY_VENV)
	$(PARITY_PYTEST) tests/parity/test_parity.py -v

test-all: test test-parity

# Quality section
lint: lint-swift lint-python

lint-swift:
	@if command -v swiftformat >/dev/null 2>&1; then \
		swiftformat --lint Sources tests --quiet; \
	else \
		echo "swiftformat not installed. Run: brew install swiftformat"; \
		exit 1; \
	fi

lint-python: $(PARITY_VENV)
	$(PARITY_VENV)/bin/ruff check .

format: format-swift format-python

format-swift:
	@if command -v swiftformat >/dev/null 2>&1; then \
		swiftformat --lint Sources tests; \
	else \
		echo "swiftformat not installed. Run: brew install swiftformat"; \
		exit 1; \
	fi

format-python: $(PARITY_VENV)
	$(PARITY_VENV)/bin/ruff format --check .
	$(PARITY_VENV)/bin/ruff check .

format-fix: format-fix-swift format-fix-python

format-fix-swift:
	@if command -v swiftformat >/dev/null 2>&1; then \
		swiftformat Sources tests; \
	else \
		echo "swiftformat not installed. Run: brew install swiftformat"; \
		exit 1; \
	fi

format-fix-python: $(PARITY_VENV)
	$(PARITY_VENV)/bin/ruff format .
	$(PARITY_VENV)/bin/ruff check --fix .

# Setup section
setup: setup-swift setup-python setup-hooks

setup-swift:
	swift package resolve

setup-python: $(PARITY_VENV)
	$(PARITY_PIP) install --upgrade pip
	$(PARITY_PIP) install -r tests/parity/requirements.txt
	$(PARITY_PIP) install ruff

$(PARITY_VENV):
	python3 -m venv $(PARITY_VENV)

setup-hooks:
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
	else \
		echo "pre-commit not installed. Run: pip install pre-commit"; \
		exit 1; \
	fi

# Benchmarks section
benchmark: $(PARITY_VENV) build-cli
	$(PARITY_PYTHON) benchmarks/run_benchmark.py

benchmark-compare: $(PARITY_VENV)
	$(PARITY_PYTHON) benchmarks/compare.py

parity-report: $(PARITY_VENV)
	$(PARITY_PYTHON) tests/parity/generate_report.py
