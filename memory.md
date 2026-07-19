# Tamasha — Project Memory

## Project Overview

**Tamasha** is a Bollywood Movie Intelligence Platform built as a portfolio/ML engineering showcase. It goes beyond simple rating prediction to analyze what actually drives Bollywood success: star pairings, release timing, plot tone, and poster aesthetics.

### Core Problem

Generic rating/box-office prediction is common in ML portfolios. Tamasha differentiates itself by:
1. **Star Network Analysis**: Bankability Scores + Chemistry Pairing — a time-decay-weighted measure of an actor's box-office drawing power
2. **Release Timing**: Festival-window and clash analysis using TMDb-enriched release dates
3. **Plot Sentiment**: Genre-conditional tone analysis (VADER on TMDb plot summaries)
4. **Poster CV**: Hand-crafted visual features + Random Forest for hit/flop prediction
5. **Rigorous Model Comparison**: 9 models compared under identical 5-fold CV for both rating and box-office tasks, with auto-selection via configurable metric

---

## Business Purpose

- **Portfolio showcase** for senior data scientist / ML engineer job applications
- Demonstrates: data engineering (fuzzy joining), feature engineering (Bankability Score), rigorous model comparison, MLOps (config centralization, logging, type hints), and full-stack deployment (Streamlit + FastAPI + Docker)
- Targets GitHub discoverability with topics: `ml`, `bollywood`, `network-analysis`, `streamlit`, `india`

### Users

- **Primary**: Hiring managers / senior engineers reviewing the portfolio
- **Secondary**: Anyone interested in Bollywood data analysis

### Key Metrics from Modeling

- **Rating Model**: **GradientBoosting** wins (MAE = 0.95, R² = 0.22) — rating prediction is inherently noisy with metadata-only features
- **Box Office (Baseline)**: **XGBoost** wins (MAE = ₹35.2 Cr)
- **Box Office (with Bankability)**: **XGBoost** wins (MAE = ₹32.1 Cr) — **8.7% MAE improvement** from the Bankability Score feature
- **Poster CV**: Accuracy = 49.2% vs majority baseline 51.1% — no independent signal from poster visuals
- **Plot Sentiment**: Strongest correlations — Fantasy (+0.42), Musical (+0.26), Biography (+0.17); Negative — Horror (-0.19)

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.11+ | Primary language |
| **ML/Analysis** | scikit-learn, xgboost, lightgbm, catboost | Model training & comparison |
| **NLP** | NLTK VADER | Plot sentiment analysis |
| **CV** | OpenCV, scikit-learn | Poster feature extraction + classification |
| **Network** | networkx | Cast/crew collaboration graph |
| **Data** | pandas, numpy, rapidfuzz | Data manipulation & fuzzy joining |
| **Config** | pydantic-settings | Centralized configuration |
| **Dashboard** | Streamlit | Multi-page interactive UI |
| **API** | FastAPI | Prediction endpoints |
| **Deployment** | Docker, docker-compose | Containerization |
| **Testing** | pytest | Unit tests (46 passing) |
| **CI/CD** | GitHub Actions | Pre-commit + pytest on every push |
| **Quality** | black, isort, ruff, pre-commit | Code formatting & linting |

---

## Repository Structure

```
tamasha/
├── data/                          # Gitignored — raw + processed datasets
│   ├── raw/                       # Kaggle CSVs + downloaded poster images
│   └── processed/                 # Cleaned Parquet files, TMDb cache, posters
│
├── notebooks/                     # (Original scaffold) Not implemented — all logic in src/
│
├── src/tamasha/                   # Core package — the tested, importable library
│   ├── __init__.py
│   ├── config.py                  # pydantic-settings: paths, thresholds, constants
│   ├── predict.py                 # Shared prediction functions (dashboard + API both call this)
│   ├── train_pipeline.py          # End-to-end pipeline (make train entry point)
│   │
│   ├── data/
│   │   ├── loaders.py             # Load raw IMDB + Box Office CSVs with column mapping
│   │   ├── joining.py             # Two-step fuzzy join via rapidfuzz
│   │   ├── cleaning.py            # Currency parsing, inflation adjustment (optional)
│   │   └── enrichment.py          # TMDb API: plot summaries + release dates + poster paths
│   │
│   ├── features/
│   │   ├── movie_features.py      # Genre one-hot, cast size, runtime, decade, budget
│   │   └── cast_crew_network.py   # networkx collaboration graph
│   │
│   ├── models/
│   │   ├── model_selection.py     # 9-model registry, CV comparison, auto-select by metric
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
│   │   ├── bankability_score.py   # Time-decay-weighted performance score (half-life=5yrs)
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
│   ├── streamlit_app.py           # Entry point, sidebar, page navigation
│   ├── pages/
│   │   ├── _1_Predict_a_Release.py      # Genre/cast/budget → rating + box office
│   │   ├── _2_Star_Network_Explorer.py   # Interactive force-directed graph + actor cards
│   │   ├── _3_Industry_Trends.py         # Genre trends, festival analysis, plot tone
│   │   └── _4_Model_Performance.py       # Model comparison charts + SHAP explainability
│   ├── components/
│   │   ├── metric_cards.py        # Glass card, badge, metric_card components
│   │   └── network_graph.py       # Force-directed graph renderer
│   └── assets/
│       └── theme.css             # Glassmorphism theme, animations, micro-interactions
│
├── api/                           # FastAPI application
│   ├── main.py                    # App entry, CORS, router includes
│   ├── schemas.py                 # Pydantic request/response models
│   └── routers/
│       ├── predict.py             # POST /predict-rating, POST /predict-boxoffice
│       ├── network.py             # GET /actor/{name}
│       └── model_info.py          # GET /model-info
│
├── models/                        # Gitignored .pkl files
│   ├── best_rating_model.pkl      # Winning rating model (GradientBoosting)
│   ├── best_boxoffice_model.pkl   # Winning box office model (XGBoost + Bankability)
│   ├── rating_features.json       # Feature column names for inference
│   └── boxoffice_features.json    # Feature column names for inference
│
├── reports/                       # Generated reports and CSVs
│   ├── model_comparison_rating.csv
│   ├── model_comparison_boxoffice_baseline.csv
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
├── tests/                         # pytest suite (46 tests)
│   ├── conftest.py                # Fixtures: synthetic DataFrames, known-match pairs
│   ├── test_joining.py            # Fuzzy join match-rate on synthetic data
│   ├── test_cleaning.py           # Currency parsing, inflation adjustment
│   ├── test_features.py           # Feature extraction correctness
│   ├── test_model_selection.py    # Model picker from mock comparison table
│   ├── test_bankability_score.py  # Hand-computed expected scores on small graph
│   ├── test_chemistry_pairs.py    # Planted "obvious" chemistry pair detection
│   ├── test_festival_calendar.py  # Festival date flag correctness
│   └── test_api.py                # FastAPI smoke tests for every route
│
├── .github/workflows/ci.yml       # pre-commit → pytest on push
├── Dockerfile                     # Streamlit container
├── docker-compose.yml             # api + dashboard services
├── render.yaml                    # Render Blueprint deployment
├── requirements.txt               # Pinned dependencies
├── Makefile                       # train, test, clean, install targets
├── .pre-commit-config.yaml        # black, isort, ruff
├── .streamlit/secrets.toml        # TMDb credentials template
├── packages.txt                   # Streamlit Cloud system deps (OpenCV)
├── setup.py                       # Package installer
├── LICENSE (MIT)
├── README.md                      # 15-section full documentation
└── memory.md                      # THIS FILE
```

---

## System Architecture

```
┌─────────────┐     ┌───────────────────────────────────────────────────┐
│  Kaggle/DVC  │────▶│  Data Loading & Processing                       │
│  Datasets    │     │  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
└─────────────┘     │  │ loaders  │─▶│  joining  │─▶│  enrichment   │  │
                    │  │ (.py)    │  │ (fuzzy)   │  │ (TMDb API)    │  │
                    │  └──────────┘  └───────────┘  └──────────────┘  │
                    └────────────────────┬──────────────────────────────┘
                                         │
                    ┌────────────────────▼──────────────────────────────┐
                    │  Feature Engineering                             │
                    │  ┌──────────────┐  ┌──────────────┐             │
                    │  │ movie_feat.  │  │ cast_crew_   │             │
                    │  │ (genre/cast/ │  │ network.py   │  ┌──────────┐│
                    │  │  budget/…)   │  │ (networkx)   │  │ bankabi- ││
                    │  └──────────────┘  └──────────────┘  │ lity.py  ││
                    │                                       └──────────┘│
                    └────────────────────┬──────────────────────────────┘
                                         │
                    ┌────────────────────▼──────────────────────────────┐
                    │  Model Training Pipeline (make train)             │
                    │                                                   │
                    │  ┌─────────────────────────────────────────────┐  │
                    │  │  Rating Model Comparison (9 models, 5-fold) │  │
                    │  │  → GradientBoosting wins (MAE=0.95)        │  │
                    │  └─────────────────────────────────────────────┘  │
                    │                                                   │
                    │  ┌─────────────────────────────────────────────┐  │
                    │  │  Box Office: Baseline (9 models, 5-fold)    │  │
                    │  │  → XGBoost wins (MAE=₹35.2 Cr)             │  │
                    │  └─────────────────────────────────────────────┘  │
                    │                                                   │
                    │  ┌─────────────────────────────────────────────┐  │
                    │  │  Box Office: With Bankability (9 models)    │  │
                    │  │  → XGBoost wins (MAE=₹32.1 Cr) — 8.7% ↑   │  │
                    │  └─────────────────────────────────────────────┘  │
                    └────────────────────┬──────────────────────────────┘
                                         │
                    ┌────────────────────▼──────────────────────────────┐
                    │  Saved Artifacts                                  │
                    │  models/              reports/                    │
                    │  ├ best_rating.pkl    ├ model_comparison_*.csv    │
                    │  └ best_boxoffice.pkl └ figures/*.png             │
                    └────────────────────┬──────────────────────────────┘
                                         │
            ┌────────────────────────────┼────────────────────────────┐
            │                            │                            │
            ▼                            ▼                            ▼
    ┌───────────────┐          ┌──────────────────┐         ┌──────────────┐
    │  Streamlit     │          │  FastAPI          │         │  API External│
    │  Dashboard     │          │  (5 endpoints)    │         │  Consumers   │
    │  (4 pages)     │          │                   │         └──────────────┘
    │  local:8501    │          │  local:8000        │
    └───────┬───────┘          └────────┬─────────┘
            │                           │
            └───────────┬───────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  docker-compose  │
              │  api + dashboard │
              └──────────────────┘
```

### Frontend-Backend Communication

- Both Streamlit and FastAPI call **the same `tamasha.predict` module** — no duplicate prediction logic
- Streamlit: synchronous Python calls direct into src/tamasha/
- FastAPI: JSON-based HTTP, wraps same predict module
- All data flows through `tamasha.config.settings` (pydantic-settings) — no hardcoded paths

---

## Routing Map

### Streamlit Pages (via sidebar navigation)

| Route | Page File | Purpose | Auth |
|-------|-----------|---------|------|
| `/` | `app/streamlit_app.py` | Redirects to Page 1 | None |
| `Predict a Release` | `app/pages/_1_Predict_a_Release.py` | Genre/cast/budget → predictions | None |
| `Star Network Explorer` | `app/pages/_2_Star_Network_Explorer.py` | Interactive actor graph | None |
| `Industry Trends` | `app/pages/_3_Industry_Trends.py` | Genre/festival/tone analysis | None |
| `Model Performance` | `app/pages/_4_Model_Performance.py` | Comparison charts + SHAP | None |

### FastAPI Routes

| Method | Route | Handler | Purpose |
|--------|-------|---------|---------|
| `GET` | `/health` | `health()` | Health check + version |
| `POST` | `/predict-rating` | `predict_rating_endpoint()` | Rating prediction |
| `POST` | `/predict-boxoffice` | `predict_boxoffice_endpoint()` | Box office prediction |
| `GET` | `/actor/{name}` | `get_actor_info_endpoint()` | Bankability + chemistry |
| `GET` | `/model-info` | `get_model_info_endpoint()` | Deployed model metadata |

---

## Data Flow — Major Features

### 1. Prediction Pipeline (Predict a Release)

```
User Input (genres, cast, budget, runtime, year)
  │
  ▼
predict_rating() / predict_boxoffice()  [tamasha.predict]
  │
  ├─ _load_rating_model() / _load_boxoffice_model()  (lazy-loaded .pkl)
  ├─ _load_feature_cols()  (JSON column names from training)
  ├─ _build_prediction_vector()  (one-hot genres, numeric features)
  │
  ├─ Model.predict(X_vec)  → predicted value
  │
  └─ Return dict → Streamlit renders (gauges, star ratings, scenario bars)
         ↓
         Also → FastAPI → JSON response
```

### 2. Training Pipeline (make train)

```
Raw CSVs (IMDb India + Bollywood Box Office + year-bridge)
  │
  ▼
load_imdb_india() + load_bollywood_boxoffice()  [data/loaders.py]
  │
  ▼
Two-Step Fuzzy Join  [data/joining.py]
  │  Box Office → year-bridge → IMDb
  │  rapidfuzz.WRatio + year_tolerance=2
  │
  ▼
Data Cleaning  [data/cleaning.py]
  │  Parse currency → numeric ₹
  │  Drop duplicates, fill NaN
  │  Decision: NO inflation adjustment (data concentrated 2010-2023)
  │
  ▼
TMDb Enrichment  [data/enrichment.py]
  │  Plot summaries (93.3% coverage)
  │  Release dates (93.5% coverage)
  │  Poster paths → download ~200 images
  │  Cache: data/processed/tmdb_cache.json
  │
  ├──▶ Rating Model Comparison (Stage 3)
  │     9 models, 5-fold CV, auto-select by MAE
  │     → GradientBoosting (MAE=0.95, R²=0.22)
  │     → Save: models/best_rating_model.pkl
  │
  ├──▶ Plot Sentiment (Stage 5)
  │     VADER on TMDb plot summaries
  │     Genre-conditional correlations
  │
  ├──▶ Bankability Scores + Chemistry Pairs (Stage 6)
  │     Exponential decay w(t) = 2^(-(current - t)/5)
  │     1,010 individuals → top chemistry pairs
  │
  ├──▶ Box Office: Baseline (Stage 4a)
  │     9 models → XGBoost (MAE=₹35.2 Cr)
  │
  ├──▶ Box Office: With Bankability (Stage 4b)
  │     9 models → XGBoost (MAE=₹32.1 Cr, 8.7% ↑)
  │     → Save: models/best_boxoffice_model.pkl
  │
  ├──▶ Release Timing (Stage 7)
  │     9 festival windows, clash detection
  │     Festival releases outperform by ~20%
  │
  └──▶ SHAP Explainability (Stage 9)
        Rating + Box Office SHAP summary plots
        Bankability = #2 feature in box office model
```

### 3. Actor Search (Star Network Explorer)

```
User searches actor name
  │
  ▼
get_actor_info(name)  [tamasha.predict]
  │
  ├─ Lookup in bankability_scores.csv (lowercase matching)
  ├─ Lookup in chemistry_pairs.csv (actor_1 or actor_2)
  │
  └─ Return dict → Streamlit renders actor card + chemistry badges
```

### 4. Model Performance (Dashboard Page)

```
Dashboard loads on page visit
  │
  ▼
get_comparison_csv(task)  [tamasha.predict]
  │
  ├─ Reads model_comparison_rating.csv from disk
  ├─ Reads model_comparison_boxoffice_baseline.csv
  └─ Reads model_comparison_boxoffice_with_bankability.csv
  │
  └─ Streamlit renders grouped bar charts + SHAP images
```

---

## Bankability Score — Decay Function

```
w(t) = 2^(-(current_year - t) / H)

where:
  t = film release year
  current_year = latest year in dataset
  H = half-life in years (default: 5, configurable via settings)

Performance signal = (rating/10) * (1 + log1p(box_office) / log1p(max_box_office))

Bankability = Σ(w * performance) / Σ(w)
```

- Half-life of 5 years means a film from 5 years ago has 50% weight
- Documented decision: exponential decay gives smooth, interpretable weights
- 1,010 individuals scored across the dataset

---

## Environment Variables

| Variable | Purpose | Source |
|----------|---------|--------|
| `TMDB_API_KEY` | TMDb API key for enrichment | `.env` file |
| `TMDB_ACCESS_TOKEN` | TMDb Bearer token (preferred) | `.env` file |
| `TAMASHA_LOG_LEVEL` | Logging level (default: INFO) | `.env` or env |
| `TAMASHA_DATA_RAW` | Override raw data path | `.env` or env |
| `TAMASHA_DATA_PROCESSED` | Override processed data path | `.env` or env |
| `TAMASHA_MODELS_DIR` | Override models directory | `.env` or env |

---

## Third Party Integrations

| Service | Purpose | API Required? | Coverage |
|---------|---------|:------------:|:--------:|
| **TMDb API** | Plot summaries, release dates, poster paths | Yes (free key) | 93.3% plots, 93.5% dates |
| **Kaggle API** | Dataset download | Yes (free key) | N/A (one-time offline) |
| **Rapidfuzz** | Fuzzy string matching | No (local) | N/A |
| **NLTK VADER** | Sentiment analysis | No (local + lexicon) | N/A |
| **OpenCV** | Image processing | No (local) | N/A |

---

## Testing

- **Framework**: pytest
- **Test count**: 46
- **Key test areas**:
  - Joining: synthetic DataFrames with known expected matches
  - Bankability: hand-computed scores on small synthetic graph
  - Chemistry: planted "obvious" pair in test data
  - Festival calendar: date flag correctness
  - Model selection: mock comparison table → correct model picker
  - API: FastAPI TestClient smoke tests for all 5 routes
- **CI**: GitHub Actions → pre-commit (black, isort, ruff) → pytest on every push

---

## Deployment

### Options

| Method | Dashboard | API | Notes |
|--------|-----------|-----|-------|
| Docker (local) | `localhost:8501` | `localhost:8000` | `docker compose up -d` |
| Streamlit Cloud | share.streamlit.io | N/A | Entry: `app/streamlit_app.py` |
| Render | N/A | render.com | Blueprint via `render.yaml` |

### Model Files

- `models/*.pkl` are gitignored (~1.4MB each)
- Must run `make train` or download from releases
- Deployed apps need either committed models or a build step

---

## Feature Inventory

| Feature | Frontend Files | Backend Files | Data | Status |
|---------|---------------|---------------|------|--------|
| Rating Prediction | `1_Predict_a_Release.py` | `predict.py`, `models/rating_model.py`, `models/model_selection.py` | `models/best_rating_model.pkl` | ✅ Done |
| Box Office Prediction | `1_Predict_a_Release.py` | `predict.py`, `models/boxoffice_model.py` | `models/best_boxoffice_model.pkl` | ✅ Done |
| Bankability Score | `2_Star_Network_Explorer.py` | `network/bankability_score.py`, `predict.py` | `reports/bankability_scores.csv` | ✅ Done |
| Chemistry Pairs | `2_Star_Network_Explorer.py` | `network/chemistry_pairs.py` | `reports/chemistry_pairs.csv` | ✅ Done |
| Genre-Tone Analysis | `3_Industry_Trends.py` | `nlp/plot_sentiment.py` | `reports/genre_tone_correlation.csv` | ✅ Done |
| Festival/Clash Analysis | `3_Industry_Trends.py` | `timing/festival_calendar.py` | `reports/release_timing_analysis.md` | ✅ Done |
| Poster CV | `3_Industry_Trends.py` | `cv/poster_classifier.py` | On-the-fly (200 images) | ✅ Done (null result) |
| SHAP Explainability | `4_Model_Performance.py` | `evaluation/metrics.py` | `reports/figures/shap_*.png` | ✅ Done |
| Model Comparison | `4_Model_Performance.py` | `evaluation/metrics.py`, `models/model_selection.py` | `reports/model_comparison_*.csv` | ✅ Done |
| Release Scenario Simulator | `1_Predict_a_Release.py` | `timing/release_scenario.py` | On-the-fly | ✅ Done |
| Actor Info API | N/A (API) | `api/routers/network.py`, `predict.py` | Same Bankability data | ✅ Done |
| Health Check | N/A (API) | `api/main.py` | None | ✅ Done |

---

## Dependency Graph (Critical Files)

```
                    ┌─────────────────┐
                    │  config.py       │◀── All modules depend on this
                    │  (pydantic-settings)│
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
  ┌─────────────┐   ┌───────────────┐   ┌──────────────────┐
  │ data/       │   │ features/     │   │ models/          │
  │ loaders.py  │   │ movie_feat.py │   │ model_selection  │
  │ joining.py  │   │ cast_crew_    │   │ rating_model.py  │
  │ cleaning.py │   │ network.py    │   │ boxoffice_model  │
  │ enrichment  │   └───────┬───────┘   └────────┬─────────┘
  └──────┬──────┘           │                    │
         │                  ▼                    ▼
         │           ┌───────────────┐   ┌──────────────────┐
         │           │ network/      │   │  predict.py      │
         │           │ bankability   │   │  (shared preds)  │
         │           │ chemistry     │   └────────┬─────────┘
         │           └───────┬───────┘            │
         │                   │              ┌─────┴──────┐
         │                   │              │            │
         ▼                   ▼              ▼            ▼
  ┌──────────────┐  ┌──────────────┐  ┌────────┐  ┌────────────┐
  │ nlp/         │  │ timing/      │  │ app/   │  │ api/       │
  │ plot_sent.   │  │ festival_    │  │ pages/ │  │ routers/   │
  │              │  │ cal + scen.  │  │        │  │            │
  └──────────────┘  └──────────────┘  └────────┘  └────────────┘

Key: Critical = high risk if changed
  - config.py      : EVERYTHING depends on it
  - predict.py     : Both dashboard and API depend on it
  - model_selection.py : Both rating + box office training depend on it
  - train_pipeline.py : Orchestrates everything — complex interdependencies
```

---

## Technical Debt / Known Issues

1. **Poster CV**: Null result (49.2% vs 51.1% baseline) — 200-image sample + hand-crafted features are insufficient. Documented honestly.
2. **Rating Model R²=0.22**: Metadata-only features can't fully predict ratings. Adding plot sentiment as a feature could improve this.
3. **Festival dates are approximate**: Movable festivals (Eid, Diwali) use approximate dates. A library like `hijri-converter` would improve accuracy.
4. **No inflation adjustment**: Data concentrated in 2010-2023, but cross-decade comparisons would benefit from adjustment.
5. **TMDb API dependency**: Enrichment requires internet + API key. Cached, but first run is slow.
6. **Model files gitignored**: Users must run `make train` after cloning. Could add a `make download-models` target with GitHub Releases.
7. **LightGBM/CatBoost warnings**: Optional heavy models may show warnings if not installed; handled gracefully in code.

---

## Development Workflow

```bash
# 1. Setup
git clone <repo>
cd tamasha
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Download data
# Download from Kaggle (see Kaggle links in README)
# Place CSVs in data/raw/

# 3. Set up TMDb (optional, for enrichment)
echo "TMDB_API_KEY=your_key" > .env
echo "TMDB_ACCESS_TOKEN=your_token" >> .env

# 4. Run full pipeline
make train    # ~15-20 minutes: loads data, joins, trains 27 model instances

# 5. Run dashboard
make run      # Streamlit on :8501

# 6. Run API
make api      # FastAPI on :8000

# 7. Tests
make test     # 46 tests

# 8. Docker
docker compose up -d
```
