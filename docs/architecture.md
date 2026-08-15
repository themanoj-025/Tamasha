# Tamasha — System Architecture

Tamasha is a **Bollywood movie intelligence platform** that predicts box-office
and rating outcomes, analyzes star networks and bankability, and surfaces
release-timing insights through two front-ends (Streamlit app + FastAPI).

## High-level components

```
                ┌────────────────────────────────────────────┐
                │                Interfaces                   │
                │  Streamlit app (app/)     FastAPI (api/)    │
                │  `streamlit run app/`     `gunicorn api`    │
                └──────────────┬──────────────────┬───────────┘
                               │                  │
                               ▼                  ▼
                ┌────────────────────────────────────────────┐
                │           Core package (src/tamasha/)      │
                │  predict.py  ·  cache.py  ·  config.py     │
                │  ┌────────┐ ┌────────┐ ┌──────┐ ┌───────┐  │
                │  │ data/  │ │features│ │model │ │network│  │
                │  │ cv/    │ │ nlp/   │ │eval/ │ │timing/│  │
                │  └────────┘ └────────┘ └──────┘ └───────┘  │
                └──────────────┬──────────────────────────────┘
                               │
                ┌──────────────┴──────────────────────────────┐
                │           Data & artifact stores            │
                │  data/raw/*.csv → data/processed/*.parquet  │
                │  models/*.pkl   ·  reports/*.csv  ·  cache  │
                └─────────────────────────────────────────────┘
```

## Component responsibilities

| Component | Responsibility |
|---|---|
| `src/tamasha/data/` | Load raw CSVs (`loaders`), clean (IMDb box-office reconciliation), join IMDb↔box-office, enrich with TMDB (posters, plot sentiment) |
| `src/tamasha/features/` | Feature engineering: `movie_features` (global feature builder) and `cast_crew_network` (director/actor graph) |
| `src/tamasha/models/` | Box-office & rating models, feature manifest (`*_features.json`), `model_selection` (hyperparameter comparison) |
| `src/tamasha/network/` | Star-network analytics: `bankability_score`, `chemistry_pairs` |
| `src/tamasha/cv/` | Poster-image classification (`poster_classifier`) |
| `src/tamasha/nlp/` | Plot-synopsis sentiment (`plot_sentiment`) |
| `src/tamasha/timing/` | Release timing: `festival_calendar`, `release_scenario` |
| `src/tamasha/evaluation/` | Shared evaluation metrics (`metrics`) |
| `src/tamasha/predict.py` | The public prediction service facade used by both front-ends |
| `src/tamasha/train_pipeline.py` | End-to-end training entry (`python -m tamasha.train_pipeline`) |
| `api/` | FastAPI service: lifespan-built `PredictionService`, rate limiting (slowapi), auth guard, Prometheus metrics, 3 routers (`predict`, `network`, `model_info`) |
| `app/` | Streamlit multipage UI (4 pages) + shared components + theme CSS |
| `tests/` | 18 pytest modules incl. API contract, auth, cache, and CV-consistency regression suites |
| `ops/` | Observability: Prometheus config + Grafana dashboards |

## Key architectural decisions

- **src-layout package** (`src/tamasha`) — installed via `setup.py`/`pyproject.toml`;
  the Docker build stage installs it so runtime containers get a clean package.
- **Single prediction facade** — `tamasha.predict` is the only business-logic
  entry the Streamlit app and the API talk to, keeping logic in one place.
- **Lifespan DI (FastAPI)** — `api/main.py` builds `PredictionService` once at
  startup and injects it via `Depends()`, with rate-limit/auth middleware.
- **SQLite prediction cache** — repeated predictions are cached
  (`tamasha.cache`) with a TTL to keep inference cost low.
- **Both front-ends share one truth** — Streamlit pages and API routers both
  consume `tamasha.*` directly; no duplicated business logic.
