.PHONY: help install train test lint clean docker-up docker-build

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies (choose extras with EXTRAS, e.g. EXTRAS=xgboost,shap)
	pip install -r requirements.txt
	@if [ -n "$(EXTRAS)" ]; then \
		echo "Installing extras: $(EXTRAS)"; \
		pip install $(EXTRAS); \
	fi
	python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

train:  ## Run the full training pipeline (data -> features -> model comparison -> best model selection)
	python -m tamasha.train_pipeline

test:  ## Run all tests with coverage (70% threshold)
	python -m pytest tests/ -v --cov=src/tamasha --cov=api --cov-config=pyproject.toml --cov-report=term-missing --cov-fail-under=70

lint:  ## Run pre-commit checks on all files
	pre-commit run --all-files

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete

docker-build:  ## Build Docker images
	docker compose build

docker-up:  ## Start all services via docker-compose
	docker compose up -d

docker-down:  ## Stop all services
	docker compose down

format:  ## Format code with black + isort
	black src/ tests/ app/ api/
	isort src/ tests/ app/ api/
