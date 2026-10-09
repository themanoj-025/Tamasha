# Tamasha — Package & Module Inventory

## Installed package: `tamasha` (src/tamasha)

| Module | Responsibility |
|---|---|
| `__init__.py` | Package marker + `__version__` |
| `config.py` | Environment-driven settings (paths, model names, cache TTL) — single source of truth |
| `cache.py` | SQLite-backed prediction cache with TTL |
| `exceptions.py` | Domain exception hierarchy (`TamashaError`, `ModelLoadError`, …) |
| `predict.py` | `PredictionService` facade: box-office/rating prediction, actor info, bankability, chemistry — used by both front-ends |
| `train_pipeline.py` | End-to-end training CLI entry |
| `cv/` | `poster_classifier` — genre/quality classification from poster images |
| `data/loaders.py` | Load raw CSVs (IMDb, box-office) and processed parquet |
| `data/cleaning.py` | Clean/reconcile movie + box-office records |
| `data/joining.py` | Join IMDb ↔ box-office datasets (fuzzy/identifier matching) |
| `data/enrichment.py` | TMDB enrichment: posters, actor photos, metadata |
| `evaluation/metrics.py` | Shared regression/classification metrics |
| `features/movie_features.py` | Global feature engineering for movies |
| `features/cast_crew_network.py` | Build director/actor collaboration graph |
| `models/boxoffice_model.py` | Box-office regression model wrapper |
| `models/rating_model.py` | IMDb rating regression model wrapper |
| `models/model_selection.py` | Hyperparameter/model comparison harness |
| `network/bankability_score.py` | Star bankability scoring over the cast network |
| `network/chemistry_pairs.py` | Actor-pair chemistry scores |
| `nlp/plot_sentiment.py` | Sentiment features from plot synopses |
| `timing/festival_calendar.py` | Indian festival/release-date calendar |
| `timing/release_scenario.py` | Release-timing scenario scoring |

## Application packages (not installed)

| Package | Responsibility |
|---|---|
| `api/` | FastAPI app: `main.py` (factory, lifespan DI, middleware, 3 routers), `schemas.py` (DTOs), `routers/` (predict, network, model_info) |
| `app/` | Streamlit UI: `streamlit_app.py` (multipage entry), `pages/` (4 pages), `components/` (metric cards, network graph), `assets/theme.css` |
| `tests/` | 18 test modules: API contract (`test_api`, `test_api_contract`, `test_auth`), domain regression (`test_bankability_regression`, `test_scatter_cv_consistency`), and per-module unit tests (`test_cleaning`, `test_joining`, `test_cache`, `test_features`, `test_festival_calendar*`, `test_chemistry_pairs`, `test_model_selection`, `test_predict_service`, `test_enrichment`, `test_integrity`, `test_clash_detection_scale`) + `conftest.py` |

## Non-package directories

| Path | Purpose |
|---|---|
| `data/raw/` | Source datasets (CSV) |
| `data/processed/` | Cleaned parquet, poster images, TMDB + prediction caches |
| `models/` | Trained artifacts (`*.pkl`) + feature manifests (`*_features.json`) |
| `reports/` | Analysis reports & model-comparison CSVs |
| `ops/` | Prometheus scrape config + Grafana dashboards/provisioning |
| `docs/` | Architecture, design, project, technical, reference, migration docs |
| `.github/workflows/` | CI pipeline (lint + pytest with coverage gate) |
