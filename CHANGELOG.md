# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Phase 1 — Architecture refactor**: Thread-safe `PredictionService` class replaces
  module-level mutable globals. FastAPI Depends() injection; Streamlit `st.cache_resource`
  singleton. Double-checked locking for safe concurrent loading.
- **Phase 2 — Test expansion**: 57 new tests (103 → 119 total). Mocked HTTP enrichment
  tests, API contract tests, Hypothesis property-based tests for clash detection,
  concurrency tests (20 threads), auth/rate-limit tests. Removed autouse fixtures from
  conftest.
- **Phase 4 — ML rigor**: RandomizedSearchCV hyperparameter tuning (n_iter=5) for 4
  models (RandomForest, GradientBoosting, XGBoost, LightGBM). Wilcoxon signed-rank
  statistical significance test between top models using out-of-fold predictions.
  Model versioning with metadata.json. Festival multipliers moved to config with
  documented rationale.
- **Phase 5 — DevOps hardening**: Multi-stage Dockerfile (builder + runtime),
  .dockerignore, CI gates (pre-commit blocking, coverage ≥70%, pip-audit, Docker build),
  structlog structured logging with request IDs, setup.py extras (ml, dev, all),
  CHANGELOG.md.
- **Phase 6 — 2026 layer**: Responsible AI / Limitations section in README,
  cost/latency budget table (1K DAU ~$25–40/mo).
- **Auth & Rate Limiting**: X-API-Key middleware (exempts /health, /docs),
  slowapi rate limiting (60 req/min), CORS restricted to configured origins.
  Configurable via `API_KEY`, `ALLOWED_ORIGINS`, `RATE_LIMIT` env vars.
- **Training pipeline tuning**: Wired RandomizedSearchCV into train_and_compare() with
  clean model names in CSV (boolean `tuned` column). Out-of-fold statistical
  significance test via cross_val_predict. LightGBM (tuned) now the best rating model
  (MAE=0.9587, R²=0.2173). GradientBoosting (tuned) best box-office with Bankability
  (MAE=₹75.3 Cr, 10.2% improvement).

### Fixed

- Scatter plot evaluation now notes that predictions come from a single train/test
  split (not cross-validation) to avoid misleading comparisons with CV-based metrics.
- `conftest.py` no longer writes dummy model files for tests that don't need them
  (autouse session-scoped fixture removed).
- Director LabelEncoder is now persisted and loaded at inference instead of silently
  ignoring the director input.
- Lazy import (`RandomizedSearchCV`) moved from function body to module level in
  `model_selection.py`.

### Security

- **X-API-Key authentication**: Header-based auth on all endpoints except /health and
  docs. Configurable via `API_KEY` env var (default dev key: `tamasha-dev-key-2026`).
- **Rate limiting**: slowapi with 60 req/min rate limit, using API key or IP as
  identifier. Returns 429 when exceeded.
- **CORS restriction**: Explicit allowed origins (default: localhost dev ports) instead
  of `["*"]`.
- **Request-ID on rejected auth**: Even 401 responses include `X-Request-ID` header
  for traceability.

### Changed

- API now requires `X-API-Key` header for all prediction, network, and model-info
  endpoints. Update clients accordingly.
- CORS origins restricted to configured `ALLOWED_ORIGINS` (update for your deployment).

## [0.1.0] — 2026-01-01

### Added

- Initial release: rating prediction, box office prediction, Bankability Score,
  Chemistry Pair analysis, plot sentiment, festival/clash analysis, poster CV.
- Streamlit dashboard with 4 pages.
- FastAPI with 5 endpoints.
- 9-model comparison pipeline with 5-fold CV.
- Docker + docker-compose setup.
- 46 initial tests.
