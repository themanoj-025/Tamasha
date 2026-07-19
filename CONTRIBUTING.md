# Contributing to Tamasha

Thanks for your interest in contributing! This project is primarily a
portfolio showcase, but improvements, bug reports, and suggestions are
welcome.

## Getting Started

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dependencies: `pip install -r requirements.txt`
4. Install pre-commit hooks: `pre-commit install`
5. Make your changes.
6. Run tests: `make test`
7. Run lint: `make lint`
8. Commit and push.

## Code Style

- All functions must have type annotations.
- Google-style docstrings on all public functions/classes.
- Follow the existing project structure — notebooks call `src/`, never
  redefine logic inline.
- Configuration goes in `src/tamasha/config.py`, never hardcoded.

## Pull Request Process

1. Ensure all tests pass.
2. Update documentation (README, docstrings) if needed.
3. Keep PRs focused — one feature/fix per PR.
