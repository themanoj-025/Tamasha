# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Phase 1 — Architecture refactor**: Thread-safe `PredictionService` class replaces
  module-level mutable globals. FastAPI Depends() injection; Streamlit `st.cache_resource`
  singleton. Double-checked locking for safe concurrent loading.
- **Phase 2 — Test expansion**: 57 new tests (46 → 103 total). Mocked HTTP enrichment
  tests, API contract tests, Hypothesis property-based tests for clash detection,
  concurrency tests (20 threads). Removed autouse fixtures from conftest.
- **Phase 3 — Production hardening**: X-API-Key auth middleware, slowapi rate limiting
  (60 req/min), CORS restriction to configured origins, SHA-256 model artifact
  verification, request body size limits (100KB), exponential backoff on TMDb calls
  (tenacity), async TMDb enrichment (httpx.AsyncClient, ~150x speedup).
- **Phase 4 — ML rigor**: RandomizedSearchCV hyperparameter tuning (n_iter=5→15) for 4
  models. Wilcoxon signed-rank statistical significance test. Model versioning with
  metadata.json. Scatter plots now use cross_val_predict for genuine out-of-fold
  predictions (matching CV MAE). Festival multipliers moved to config.
- **Phase 5 — DevOps hardening**: Multi-stage Dockerfile, .dockerignore, CI gates
  (pre-commit blocking, coverage ≥70%, pip-audit, Docker build, Python 3.11+3.12 matrix),
  structlog structured logging with request IDs, setup.py extras (ml, dev, all).
- **Phase 6 — 2026 layer**: Responsible AI / Limitations section in README,
  cost/latency budget table (1K DAU ~$25–40/mo).
- **Batch C — Performance optimization**: Vectorized bankability scoring (O(n) via
  pandas groupby), O(n log n) festival clash detection (sort-by-date sweep), diskcache
  response caching with model-version-aware key.
- **Batch D — Observability stack**: docker-compose.observability.yml (Prometheus +
  Grafana), auto-provisioned Grafana dashboard (4 panels: request rate, error rate,
  latency percentiles, total by endpoint), Prometheus scrape config.
- **Batch E — Gap closure**: proper httpx.AsyncClient async enrichment (removed
  ThreadPoolExecutor), clean architecture documentation, kaleido==0.2.1 pinned.
- **Batch F — Regression tests**: Bankability output-equivalence test with
  hand-computed reference, 10K-row clash detection scale test, scatter-plot CV MAE
  consistency assertion.
- **Batch G — Performance benchmarks**: Documented enrichment performance (150x async
  speedup), vectorized bankability, O(n log n) clash detection, diskcache latency.

### Fixed

- **Kaleido version conflict**: Pinned kaleido==0.2.1 (1.2.0 breaks with plotly 5.x).
- **LightGBM tuning degradation**: Constrained search space (removed max_depth=-1,
  min learning_rate=0.05) to prevent overfitting on small datasets.
- Scatter plot evaluation: replaced single train/test split with cross_val_predict
  for genuine out-of-fold predictions matching CV MAE.
- `conftest.py` no longer writes dummy model files for tests that don't need them.
- Director LabelEncoder is now persisted and loaded at inference.
- Lazy import moved from function body to module level.

### Security

- **X-API-Key authentication**: Header-based auth on all endpoints except /health and
  docs. Configurable via `API_KEY` env var.
- **Rate limiting**: slowapi with 60 req/min rate limit.
- **CORS restriction**: Explicit allowed origins instead of `["*"]`.
- **SHA-256 model verification**: Every model artifact hashed and verified before load.
- **Request body limits**: 100KB max body size with 413 response.
- **Request-ID on rejected auth**: Even 401 responses include `X-Request-ID` header.

### Changed

- API now requires `X-API-Key` header for all prediction, network, and model-info
  endpoints.
- Async enrichment uses httpx.AsyncClient instead of ThreadPoolExecutor.
- Hyperparameter tuning budget increased from n_iter=5 to n_iter=15.
- CI now tests both Python 3.11 and 3.12 with version-keyed pip cache.
- Scatter plots and reported CV MAE come from the same cross_val_predict methodology.

## [0.1.0] — 2026-01-01

### Added

- Initial release: rating prediction, box office prediction, Bankability Score,
  Chemistry Pair analysis, plot sentiment, festival/clash analysis, poster CV.
- Streamlit dashboard with 4 pages.
- FastAPI with 5 endpoints.
- 9-model comparison pipeline with 5-fold CV.
- Docker + docker-compose setup.
- 46 initial tests.
