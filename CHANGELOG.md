# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Phase 1 — Architecture refactor**: Thread-safe `PredictionService` class replaces module-level mutable globals.
  FastAPI Depends() injection; Streamlit `st.cache_resource` singleton.
- **Phase 2 — Test expansion**: 42 new tests (103 total). Mocked HTTP enrichment tests,
  API contract tests, Hypothesis property-based tests for clash detection.
  Removed autouse fixtures from conftest.
- **Phase 4 — ML rigor**: RandomizedSearchCV hyperparameter tuning, Wilcoxon signed-rank
  statistical significance test between top models, model versioning with metadata,
  festival multipliers moved to config with documented rationale.
- **Phase 5 — DevOps hardening**: Multi-stage Dockerfile (builder + runtime),
  CI gates (pre-commit blocking, coverage threshold, pip-audit, Docker build),
  structlog structured logging with request IDs, Prometheus metrics endpoint,
  setup.py extras (ml, dev, all), CHANGELOG.md.
- **Phase 6 — 2026 layer**: Responsible AI / Limitations section in README,
  cost/latency budget table, eval script for LLM narration feature.

### Fixed

- Scatter plot evaluation now notes that predictions come from a single train/test
  split (not cross-validation) to avoid misleading comparisons with CV-based metrics.
- `conftest.py` no longer writes dummy model files for tests that don't need them
  (autouse session-scoped fixture removed).

### Security

- API key auth, rate limiting, SHA256 model verification, and request size limits
  are noted as known gaps with implementation recommendations in the README.

## [0.1.0] — 2026-01-01

### Added

- Initial release: rating prediction, box office prediction, Bankability Score,
  Chemistry Pair analysis, plot sentiment, festival/clash analysis, poster CV.
- Streamlit dashboard with 4 pages.
- FastAPI with 5 endpoints.
- 9-model comparison pipeline with 5-fold CV.
- Docker + docker-compose setup.
- 46 initial tests.
