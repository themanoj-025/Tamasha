# Contributing to Tamasha

Thanks for your interest in contributing! This project is primarily a
portfolio showcase, but improvements, bug reports, and suggestions are
welcome.

## Getting Started

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dependencies: `pip install -r requirements.txt`
4. Install dev dependencies: `pip install -e .[dev]`
5. Install pre-commit hooks: `pre-commit install`
6. Make your changes.
7. Run tests: `make test`  **(119 tests)**
8. Run lint: `make lint`
9. Run coverage check: `pytest --cov=src --cov=api --cov-report=term-missing`
10. Commit and push.

## Code Style

- All functions must have type annotations.
- Google-style docstrings on all public functions/classes.
- Follow the existing project structure — notebooks call `src/`, never
  redefine logic inline.
- Configuration goes in `src/tamasha/config.py`, never hardcoded.
- API endpoints require both unit tests and contract tests.
- All new prediction features must have concurrency tests (thread safety).
- Use `structlog` for structured logging; include `request_id` for traceability.

## Pull Request Process

1. Ensure all 119 tests pass.
2. Run `pip-audit` to check for dependency vulnerabilities.
3. Update documentation (README, CHANGELOG, docstrings) if needed.
4. Keep PRs focused — one feature/fix per PR.
5. CI must pass all checks: pre-commit, pytest, coverage (≥70%), pip-audit, Docker build.
