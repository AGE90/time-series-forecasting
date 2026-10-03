# Makefile for Time Series Forecasting

# Default Python binary
PYTHON := uv run python

# Directories to check
SRC_DIRS := src tests
NOTEBOOK_DIR := notebooks
DOCS_DIR := docs

# Help command
.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# Installation commands
.PHONY: install install-dev install-notebook install-data install-test install-minimal
install: ## Install all dependencies (base + dev + extras)
	uv sync --all-groups

install-dev: ## Install only dev dependencies
	uv sync --no-default-groups --group dev

install-notebook: ## Install notebook dependencies
	uv sync --no-default-groups --group notebook

install-data: ## Install data-science dependencies and viz dependencies
	uv sync --no-default-groups --group data-science --group viz

install-test: ## Install testing dependencies
	uv sync --no-default-groups --group test

install-minimal: ## Install base (no extras)
	uv sync --no-default-groups

# Development commands
.PHONY: run setup notebook
run: ## Run the application (example main)
	uv run my-cli

setup: ## Run full environment setup
	bash scripts/setup_env.sh
	
notebook: ## Launch Jupyter Lab
	uv run --with jupyterlab jupyter lab

# Pipeline commands (data/raw -> data/interim -> data/processed -> models)
.PHONY: data features train predict pipeline
data: ## Clean raw data into data/interim
	uv run python -m tsforecasting.data.make_dataset

features: ## Build features into data/processed
	uv run python -m tsforecasting.features.build_features

train: ## Train the model and save it to models/
	uv run python -m tsforecasting.models.train_model

predict: ## Write predictions to data/processed
	uv run python -m tsforecasting.models.predict_model

pipeline: data features train ## Run data, features and train in order

# Code quality commands
.PHONY: lint format typecheck check
lint: ## Run ruff linter on src and tests
	uv run ruff check $(SRC_DIRS)

format: ## Format code with ruff
	uv run ruff format $(SRC_DIRS)

typecheck: ## Run type checks with mypy
	uv run mypy $(SRC_DIRS)

check: format lint typecheck ## Run all code quality checks

# Testing commands
.PHONY: test test-cov test-cov-html
test: ## Run pytest on tests
	uv run pytest --cov=src

test-cov: ## Run tests with coverage report
	uv run pytest --cov=src --cov-report=term-missing

test-cov-html: ## Run tests with HTML coverage report
	uv run pytest --cov=src --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

# MLflow commands
.PHONY: mlflow-ui
mlflow-ui: ## Start MLflow tracking server
	uv run mlflow ui

# Cleanup commands
.PHONY: clean clean-pyc clean-build clean-test
clean: clean-pyc clean-build clean-test ## Remove all build artifacts

clean-pyc: ## Remove Python cache files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

clean-build: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info/

clean-test: ## Remove test artifacts
	rm -rf .coverage htmlcov/ .pytest_cache/
