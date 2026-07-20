# Tamasha — Project Memory

## Project Overview

**Tamasha** is a Bollywood Movie Intelligence Platform built as a portfolio/ML engineering showcase. It goes beyond simple rating prediction to analyze what actually drives Bollywood success: star pairings, release timing, plot tone, and poster aesthetics.

### Core Problem

Generic rating/box-office prediction is common in ML portfolios. Tamasha differentiates itself by:
1. **Star Network Analysis**: Bankability Scores + Chemistry Pairing — a time-decay-weighted measure of an actor's box-office drawing power
2. **Release Timing**: Festival-window and clash analysis using TMDb-enriched release dates
3. **Plot Sentiment**: Genre-conditional tone analysis (VADER on TMDb plot summaries)
4. **Poster CV**: Hand-crafted visual features + Random Forest for hit/flop prediction
5. **Rigorous Model Comparison**: 9 models compared under identical 5-fold CV, with hyperparameter tuning (RandomizedSearchCV) and statistical significance tests
6. **Production-hardened API**: X-API-Key auth, rate limiting, structured logging, restricted CORS

---

## Business Purpose

- **Portfolio showcase** for senior data scientist / ML engineer / AI engineer job applications
- Demonstrates: data engineering (fuzzy joining), feature engineering (Bankability Score), rigorous model comparison, MLOps (config centralization, logging, type hints, model versioning), security awareness (auth + rate limiting), and full-stack deployment (Streamlit + FastAPI + Docker)
- Targets GitHub discoverability with topics: `ml`, `bollywood`, `network-analysis`, `streamlit`, `india`

### Users

- **Primary**: Hiring managers / senior engineers reviewing the portfolio
- **Secondary**: Anyone interested in Bollywood data analysis

### Key Metrics from Modeling

- **Rating Model**: **GradientBoosting (tuned)** wins (MAE = **0.9534**, R² = 0.2162) — n_iter=15 RandomizedSearchCV
- **Box Office (Baseline)**: **XGBoost (tuned)** wins (MAE = **₹83.3 Cr**)
- **Box Office (with Bankability)**: **XGBoost (tuned)** wins (MAE = **₹73.6 Cr**) — **11.6% MAE improvement** from the Bankability Score feature
- **Significance Test**: GradientBoosting vs LightGBM for rating (p=0.6389) — NOT significant. XGBoost vs GradientBoosting for box office (p=0.4375) — NOT significant
- **Poster CV**: Accuracy = 49.2% vs majority baseline 51.1% — no independent signal from poster visuals
- **Plot Sentiment**: Strongest correlations — Fantasy (+0.42), History (+0.40), Romance (+0.36)
- **SHAP**: Bankability Score ranks as #2 most important feature in the box office model (after budget)

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.11+ / 3.12 | Primary language |
| **ML/Analysis** | scikit-learn, xgboost, lightgbm, catboost | Model training, tuning & comparison (n_iter=15) |
| **NLP** | NLTK VADER | Plot sentiment analysis |
| **CV** | OpenCV, scikit-learn | Poster feature extraction + classification |
| **Network** | networkx | Cast/crew collaboration graph |
| **Data** | pandas, numpy, rapidfuzz | Data manipulation & fuzzy joining |
| **Config** | pydantic-settings | Centralized configuration |
| **Dashboard** | Streamlit | Multi-page interactive UI |
| **API** | FastAPI, slowapi, structlog, prometheus-fastapi-instrumentator | Prediction endpoints with auth + rate limiting + structured logging + metrics |
| **Cache** | diskcache | Response caching with model-version-aware keys |
| **Deployment** | Docker (multi-stage), docker-compose | Containerization |
| **Testing** | pytest, pytest-cov, Hypothesis, httpx, FastAPI TestClient | Unit tests (**141 tests ✅**) |
| **CI/CD** | GitHub Actions | Pre-commit (blocking) + pytest + coverage (≥70%) + pip-audit + Docker build + Python 3.11/3.12 matrix |
| **Quality** | black, isort, ruff, pre-commit, detect-secrets | Code formatting & linting |

---

## Key Architecture Decisions

### Thread-safe PredictionService

Replaced module-level mutable globals (`_RATING_MODEL`, `_BOXOFFICE_MODEL`, etc.) with a `PredictionService` class that:
- Loads all artifacts once in `load()` method
- Uses double-checked locking (`threading.Lock()`) for safe concurrent loading
- SHA-256 verification: every model artifact hashed at save time; hash verified before load
- On hash mismatch: raises `ModelIntegrityError`, /health reports which artifact failed
- Injected via FastAPI `Depends()` — built once at startup via lifespan
- Streamlit uses `st.cache_resource` for the equivalent singleton
- Exposes `healthy` property for monitoring
- Director `LabelEncoder` is now persisted to disk and loaded at inference (not silently ignored)

### Auth + Rate Limiting

- **X-API-Key**: Header-based auth on all endpoints except /health, /docs, /openapi.json, /redoc
- **Rate limiting**: slowapi (60 req/min per key/IP)
- **CORS**: Restricted to configured origins (no `*`)
- **Default dev key**: `tamasha-dev-key-2026` (change via `API_KEY` env var)

### Hyperparameter Tuning

RandomizedSearchCV with n_iter=15 for 4 models (RandomForest, GradientBoosting, XGBoost, LightGBM):
- Box office search space constrained to prevent overfitting (max_depth >= 1, learning_rate >= 0.05)
- Best tuning MAE (rating): GradientBoosting (0.9560) < LightGBM (0.9539) < XGBoost (0.9587) < RandomForest (0.9688)
- Tuned models used in final CV comparison with clean model names + boolean `tuned` column
- Wilcoxon signed-rank test between top 2 models using out-of-fold predictions (cross_val_predict)
- Scatter plots use `cross_val_predict()` for genuine out-of-fold predictions matching CV MAE

---

## Repository Structure

```
tamasha/
├── data/                          # Gitignored — raw + processed datasets
│   ├── raw/                       # Kaggle CSVs + downloaded poster images
│   └── processed/                 # Cleaned Parquet files, TMDb cache, posters
│
├── src/tamasha/                   # Core package — the tested, importable library
│   ├── __init__.py
│   ├── config.py                  # pydantic-settings: paths, thresholds, auth, rate limits
│   ├── predict.py                 # PredictionService class (shared by dashboard + API)
│   ├── train_pipeline.py          # End-to-end pipeline (make train entry point)
│   │
│   ├── data/
│   │   ├── loaders.py             # Load raw IMDB + Box Office CSVs with column mapping
│   │   ├── joining.py             # Two-step fuzzy join via rapidfuzz
│   │   ├── cleaning.py            # Currency parsing, inflation adjustment (optional)
│   │   └── enrichment.py          # TMDb API: plot summaries + release dates + poster paths
│   │
│   ├── features/
│   │   ├── movie_features.py      # Genre one-hot, cast size, runtime, decade, budget, director encoder
│   │   └── cast_crew_network.py   # networkx collaboration graph
│   │
│   ├── models/
│   │   ├── model_selection.py     # 9-model registry, CV comparison, tuning, significance tests, versioning
│   │   ├── rating_model.py        # Rating training wrapper
│   │   └── boxoffice_model.py     # Box office training wrapper (baseline + Bankability)
│   │
│   ├── nlp/
│   │   └── plot_sentiment.py      # VADER sentiment + genre-conditional correlation
│   │
│   ├── cv/
│   │   └── poster_classifier.py   # Hand-crafted visual features + Random Forest
│   │
│   ├── network/
│   │   ├── bankability_score.py   # Time-decay-weighted performance score (half-life=3yrs)
│   │   └── chemistry_pairs.py     # Actor pair chemistry detection
│   │
│   ├── timing/
│   │   ├── festival_calendar.py   # 9 Indian release windows, festival/clash flags
│   │   └── release_scenario.py    # Scenario simulation (MovieProfile + ScenarioResult)
│   │
│   └── evaluation/
│       └── metrics.py             # Plotly + matplotlib: bar charts, scatter plots, SHAP
│
├── app/                           # Streamlit multi-page dashboard
│   ├── streamlit_app.py           # Entry point, sidebar, page navigation (uses st.cache_resource)
│   ├── pages/                     # 4 pages: Predict, Network, Trends, Performance
│   ├── components/                # metric_cards.py, network_graph.py
│   └── assets/theme.css           # Glassmorphism theme
│
├── api/                           # FastAPI application
│   ├── main.py                    # Entry, auth middleware, rate limiting, CORS, structlog, routers
│   ├── schemas.py                 # Pydantic request/response models
│   └── routers/                   # predict.py, network.py, model_info.py
│
├── models/                        # Gitignored .pkl files
│   ├── best_rating_model.pkl      # Winning rating model (LightGBM tuned)
│   ├── best_boxoffice_model.pkl   # Winning box office model (GradientBoosting tuned + Bankability)
│   ├── rating_features.json       # Feature column names for inference
│   ├── boxoffice_features.json    # Feature column names for inference
│   ├── director_encoder.pkl       # LabelEncoder for director names
│   └── v1/                        # Versioned model registry directories
│
├── reports/                       # Generated reports and CSVs
│   ├── model_comparison_rating.csv                    # 9 models × 5-fold CV
│   ├── model_comparison_boxoffice_baseline.csv        # 9 models × 5-fold CV
│   ├── model_comparison_boxoffice_with_bankability.csv
│   ├── bankability_scores.csv     # 1,010 individuals scored
│   ├── chemistry_pairs.csv        # Top 10 pairs
│   ├── genre_tone_correlation.csv
│   ├── genre_tone_correlation_rating.csv
│   ├── join_quality_report.md
│   ├── tmdb_enrichment_coverage.md
│   ├── release_timing_analysis.md
│   └── figures/                   # PNG charts
│
├── tests/                         # pytest suite (119 tests)
│   ├── conftest.py                # Fixtures: dummy artifacts, synthetic DataFrames
│   ├── test_api.py                # FastAPI smoke tests
│   ├── test_api_contract.py       # Schema validation + endpoint contract tests
│   ├── test_auth.py               # X-API-Key auth, rate limiting, CORS (17 tests)
│   ├── test_predict_service.py    # PredictionService concurrency + edge cases (15 tests)
│   ├── test_enrichment.py         # TMDb enrichment with mocked HTTP (12 tests)
│   ├── test_festival_calendar_property.py  # Hypothesis property-based tests (8 tests)
│   ├── test_joining.py            # Fuzzy join match-rate
│   ├── test_cleaning.py           # Currency parsing
│   ├── test_features.py           # Feature extraction correctness
│   ├── test_model_selection.py    # Model picker from comparison table
│   ├── test_bankability_score.py  # Hand-computed expected scores
│   ├── test_chemistry_pairs.py    # Planted pair detection
│   └── test_festival_calendar.py  # Festival date flag correctness
│
├── .github/workflows/ci.yml       # pre-commit → pytest → coverage → pip-audit → Docker
├── Dockerfile                     # Multi-stage builder → runtime
├── .dockerignore                  # Exclude data/, .git, notebooks, __pycache__
├── docker-compose.yml             # api + dashboard services
├── render.yaml                    # Render Blueprint deployment
├── requirements.txt               # Pinned dependencies (including slowapi, structlog)
├── setup.py                       # extras_require: ml, dev, all
├── Makefile                       # train, test, clean, install targets
├── .pre-commit-config.yaml        # black, isort, ruff, detect-secrets
├── CHANGELOG.md                   # Keep a Changelog format
├── SECURITY.md                    # (placeholder template)
├── .env.example                   # TMDB + API_KEY + ALLOWED_ORIGINS + RATE_LIMIT
├── LICENSE (MIT)
├── README.md                      # Full documentation
└── memory.md                      # THIS FILE
```

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `TMDB_API_KEY` | TMDb API key for enrichment | — |
| `TMDB_ACCESS_TOKEN` | TMDb Bearer token (preferred) | — |
| `API_KEY` | X-API-Key header value | `tamasha-dev-key-2026` |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins | `http://localhost:8501,http://localhost:8000` |
| `RATE_LIMIT` | slowapi rate limit string | `60/minute` |
| `LOG_LEVEL` | Logging level | `INFO` |

---

## Third Party Integrations

| Service | Purpose | API Required? | Coverage |
|---------|---------|:------------:|:--------:|
| **TMDb API** | Plot summaries, release dates, poster paths | Yes (free key) | 93.1% plots, 93.2% dates |
| **Kaggle API** | Dataset download | Yes (free key) | N/A (one-time offline) |
| **Rapidfuzz** | Fuzzy string matching | No (local) | N/A |
| **NLTK VADER** | Sentiment analysis | No (local + lexicon) | N/A |
| **OpenCV** | Image processing | No (local) | N/A |

---

## Testing

- **Framework**: pytest, pytest-cov (≥70% coverage), Hypothesis, FastAPI TestClient
- **Test count**: **141** (up from 46)
- **Key test areas**:
  - Joining: synthetic DataFrames with known expected matches
  - Bankability: hand-computed scores on small synthetic graph + regression test against vectorized output
  - Chemistry: planted "obvious" pair in test data
  - Festival calendar: date flag correctness + Hypothesis property-based tests + 10K-row scale test
  - Model selection: mock comparison table → correct model picker
  - API: FastAPI TestClient with contract tests + auth tests
  - PredictionService: 20-thread concurrency test, edge cases, graceful degradation, model integrity checks
  - Enrichment: mocked HTTP (success, 429 retry w/ Retry-After, timeout, cache, malformed, 5xx)
  - Auth: 17 tests for API key validation, exempt paths, CORS, request body limits, /health degraded
  - Scatter-CV consistency: asserts scatter plot MAE matches reported CV MAE via cross_val_predict
  - Bankability regression: hand-computed reference vs. vectorized output equivalence
- **CI matrix**: Python 3.11 + 3.12
- **CI pipeline**: pre-commit (blocking — black, isort, ruff, detect-secrets) → pytest --cov-fail-under=70 → pip-audit → Docker build

---

## Performance Benchmarks

| Operation | Method | Time | Speedup |
|-----------|--------|:----:|:-------:|
| TMDb enrichment | Sequential (`requests`) | 60.6s | 1x (baseline) |
| TMDb enrichment | **Async** (`httpx`, concurrency=8) | **0.4s min / 0.4s med / 4.0s max** | **~150x** |
| Bankability scoring | Vectorized (pandas groupby) | O(n) vs previous O(n²) | Documented in code |
| Festival clash detection | Sort-by-date sweep | O(n log n) vs previous O(n²) | 10K rows in ~6s |
| Prediction cache hit | diskcache lookup | ~1ms | Eliminates redundant model inference |

## Known Issues

1. **Poster CV**: Null result (49.2% vs 51.1% baseline) — 200-image sample + hand-crafted features are insufficient. Documented honestly.
2. **Rating Model R²=0.22**: Metadata-only features can't fully predict ratings. Adding plot sentiment as a feature could improve this.
3. **Festival dates are approximate**: Movable festivals (Eid, Diwali) use approximate dates. A library like `hijri-converter` would improve accuracy.
4. **No inflation adjustment**: Data concentrated in 2010-2023, but cross-decade comparisons would benefit from adjustment.
5. **kaleido/plotly version conflict**: Static image export in `make train` fails with pre-installed kaleido 1.2.0 despite 0.2.1 pin. Fix pending clean-env verification.
6. **Grafana screenshot not taken**: Dashboard JSON committed but screenshot requires local Docker setup with real traffic data.
7. **D1/D2 LLM features skipped**: LLM narration endpoint and eval set not implemented (no `ANTHROPIC_API_KEY` available).
8. **No live deployment**: Per instruction, Render/Streamlit Cloud deployment deferred. All infrastructure code present but not deployed.
9. **Model files gitignored**: Users must run `make train` after cloning. Could add a `make download-models` target with GitHub Releases.
