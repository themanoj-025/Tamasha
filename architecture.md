# Tamasha — System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User / Browser                              │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
  ┌────────────────────┐     ┌────────────────────────┐
  │  Streamlit         │     │  FastAPI               │
  │  Dashboard         │     │  Port 8000             │
  │  Port 8501         │     │                        │
  │                    │     │  /health               │
  │  Predict Release   │     │  /predict-rating       │
  │  Star Network      │     │  /predict-boxoffice    │
  │  Industry Trends   │     │  /actor/{name}         │
  │  Model Perf.       │     │  /model-info           │
  └──────────┬─────────┘     └───────────┬────────────┘
             │                           │
             │   Both call the SAME      │
             │   prediction functions    │
             └───────┬───────────────────┘
                     │
                     ▼
      ┌─────────────────────────────────┐
      │  tamasha.predict                │
      │  (shared prediction module)     │
      └───────────────┬─────────────────┘
                      │
                      ▼
      ┌─────────────────────────────────┐
      │  Saved Artifacts                │
      │  ┌──────────────┐              │
      │  │ best_rating  │.pkl          │
      │  │ best_boxoff. │.pkl          │
      │  │ features     │.json         │
      │  │ bankability  │.csv          │
      │  │ chemistry    │.csv          │
      │  │ comparisons  │.csv          │
      │  └──────────────┘              │
      └─────────────────────────────────┘
```

## Layer Breakdown

### 1. Data Layer (`src/tamasha/data/`)

| Module | Responsibility | Input | Output |
|--------|---------------|-------|--------|
| `loaders.py` | Read raw CSVs from `data/raw/`, rename/parse columns | CSV files | Clean DataFrames with canonical names |
| `joining.py` | Two-step fuzzy join datasets via rapidfuzz | Two DataFrames + title/year columns | Joined DataFrame with `_match_score` |
| `cleaning.py` | Parse INR currency strings, handle duplicates, standardize | Raw joined DataFrame | Cleaned DataFrame |
| `enrichment.py` | Call TMDb API for plot summaries, release dates, poster paths | Movie titles + years | Enriched DataFrame + local cache (JSON) |

### 2. Feature Engineering Layer (`src/tamasha/features/`)

| Module | Responsibility | Feature Count |
|--------|---------------|:-------------:|
| `movie_features.py` | Genre one-hot, cast size, director encoding, runtime, decade dummies, budget | ~36 (rating) / ~32 (box office baseline) |

### 3. Feature: Network Analysis (`src/tamasha/network/`)

| Module | Responsibility | Key Concept |
|--------|---------------|-------------|
| `bankability_score.py` | Time-decay-weighted performance history per actor/director | Exponential decay, half-life=5yr |
| `chemistry_pairs.py` | Actor pairs with 2+ joint films: compare joint vs solo performance | Uplift score = joint_avg - max(solo_a, solo_b) |

### 4. Model Layer (`src/tamasha/models/`)

| Module | Responsibility | Models Trained |
|--------|---------------|:--------------:|
| `model_selection.py` | Model registry, 5-fold CV comparison, auto-select by metric, save/load | 9 models x 2 tasks = 18 + 1 extra run = 27 total |
| `rating_model.py` | Wrapper: rating feature matrix + comparison | 9 models, winner saved |
| `boxoffice_model.py` | Wrapper: box office (baseline + Bankability) | 9 x 2 = 18 models, winner saved |

**Model Registry** (in `model_selection.py`):

| Model | Default Params | In Registry? |
|-------|---------------|:------------:|
| LinearRegression | - | Always |
| Ridge | alpha=1.0 | Always |
| Lasso | alpha=0.01 | Always |
| DecisionTree | max_depth=10 | Always |
| RandomForest | n_estimators=200, max_depth=15 | Always |
| GradientBoosting | n_estimators=200, max_depth=5 | Always |
| XGBoost | n_estimators=200, max_depth=6 | Dynamic import |
| LightGBM | n_estimators=200, max_depth=6 | Dynamic import |
| CatBoost | iterations=200, depth=6 | Dynamic import |

### 5. Prediction Layer (`src/tamasha/predict.py`)

This is the **shared prediction module** - both Streamlit dashboard and FastAPI call into it. Key functions:

| Function | Purpose | Returns |
|----------|---------|---------|
| `predict_rating()` | Rating from metadata | `{predicted_rating, model_name, model_mae}` |
| `predict_boxoffice()` | Box office + scenario comparison | `{predicted_boxoffice_cr, model_name, scenarios, bankability_info}` |
| `get_actor_info()` | Bankability + chemistry for one actor | `{name, bankability_score, film_count, top_chemistry_pairs}` |
| `get_model_info()` | Deployed model metadata | `{rating_model: ..., boxoffice_model: ...}` |
| `get_bankability_scores()` | Full 1,010-row DataFrame | `pd.DataFrame` |
| `get_chemistry_pairs()` | Full chemistry pairs | `pd.DataFrame` |
| `get_comparison_csv()` | Specific comparison CSV as DataFrame | `pd.DataFrame` or None |

### 6. Analysis Layer

| Module | Analysis | Method | Output |
|--------|----------|--------|--------|
| `nlp/plot_sentiment.py` | Plot tone per movie + genre-conditional correlation | VADER (NLTK) | `score_plot_sentiment()` (compound/pos/neu/neg); `genre_conditional_correlation()` (per-genre correlation) |
| `cv/poster_classifier.py` | Hit/flop from poster visuals | Color histogram, brightness, edge density, face count, channel stats -> Random Forest | `train_poster_classifier()` (accuracy vs baseline) |
| `timing/festival_calendar.py` | Festival release flag + clash detection | 9 reference Indian festival dates, window_days=7 | `compute_festival_features()` (is_festival_release); `compute_clash_feature()` (has_clash) |
| `timing/release_scenario.py` | Scenario simulation | MovieProfile + scenario -> feature vector -> model.predict() | `simulate_scenarios()` (list of ScenarioResult) |

### 7. Evaluation Layer (`src/tamasha/evaluation/metrics.py`)

| Function | Purpose | Visualization Lib |
|----------|---------|:-----------------:|
| `plot_model_comparison()` | Grouped MAE/RMSE bar chart | Plotly (fallback: matplotlib) |
| `plot_predicted_vs_actual()` | Scatter plot with identity line | Plotly (fallback: matplotlib) |
| `plot_shap_summary()` | SHAP feature importance summary | SHAP + matplotlib |

### 8. Frontend Layer (`app/`)

| File | Type | Content |
|------|------|---------|
| `streamlit_app.py` | Entry point | Sidebar navigation, page switching, global styling |
| `assets/theme.css` | Stylesheet | Glassmorphism theme, keyframe animations, dark theme, micro-interactions |
| `components/metric_cards.py` | Reusable components | `glass_card()`, `badge()`, `metric_card()` with featured/gradient modes |
| `components/network_graph.py` | Graph renderer | Force-directed graph via pyvis for actor collaboration |

### 9. API Layer (`api/`)

| File | Type | Routes |
|------|------|--------|
| `main.py` | FastAPI app | `/health`, CORS, router inclusion |
| `schemas.py` | Pydantic models | PredictRatingRequest/Response, PredictBoxOfficeRequest/Response, ActorInfoResponse, ModelInfoResponse |
| `routers/predict.py` | Router | POST /predict-rating, POST /predict-boxoffice |
| `routers/network.py` | Router | GET /actor/{name} |
| `routers/model_info.py` | Router | GET /model-info |

## Configuration Centralization

All paths, thresholds, and constants are in `src/tamasha/config.py` via pydantic-settings:

```python
class Settings(BaseSettings):
    PROJECT_ROOT: Path
    DATA_RAW: Path
    DATA_PROCESSED: Path
    MODELS_DIR: Path
    REPORTS_DIR: Path
    FIGURES_DIR: Path
    MODEL_SELECTION_METRIC: str = "MAE"  # Configurable: MAE, RMSE, R2
    BANKABILITY_DECAY_HALFLIFE_YEARS: float = 5.0
    LOG_LEVEL: str = "INFO"
```

No hardcoded paths exist anywhere in the codebase - all modules import `from tamasha.config import settings`.

## Deployment Options

### Docker (Local)

```
┌─────────────────────────────────────────────────┐
│                  Docker Host                      │
│                                                   │
│  ┌──────────────────┐  ┌──────────────────────┐  │
│  │  api_container     │  │  dashboard_container  │  │
│  │  :8000             │  │  :8501               │  │
│  │                    │  │                      │  │
│  │  uvicorn           │  │  streamlit run       │  │
│  │  api.main:app      │  │  app/streamlit_app   │  │
│  └───────┬────────────┘  └──────────┬───────────┘  │
│          │                          │              │
│          └──────────┬───────────────┘              │
│                     │                              │
│               Volume mount:                        │
│               ./models:/app/models                 │
│               ./data:/app/data                     │
└────────────────────────────────────────────────────┘
```

### Render (FastAPI Only)

A `render.yaml` Blueprint is provided for one-click deployment:

- **Service**: Web Service (free tier, 512MB RAM)
- **Build**: `pip install -e .` + NLTK VADER download
- **Start**: `gunicorn -w 1 -k uvicorn.workers.UvicornWorker api.main:app --bind 0.0.0.0:$PORT --timeout 120`
- **Env vars**: `TMDB_API_KEY`, `TMDB_ACCESS_TOKEN` (set via Render dashboard, sync: false for security)
- **Python**: 3.11

### Streamlit Community Cloud (Dashboard Only)

- **Entry point**: `app/streamlit_app.py`
- **Secrets**: `.streamlit/secrets.toml` template with placeholder TMDB values
- **System deps**: `packages.txt` (libgl1-mesa-glx, libglib2.0-0 for OpenCV)
- **NLTK**: VADER lexicon downloaded at app startup

## Data Flow Diagram - Inference

```
User selects genres, enters cast, sets budget
         |
         v
Streamlit form submits to predict_rating() / predict_boxoffice()
         |
         v
tamasha.predict:
  1. Lazy-load model from .pkl (cached after first load)
  2. Lazy-load feature column names from .json
  3. Build feature vector:
     - One-hot encode genres (genre_{name} = 1.0)
     - Cast size = len(cast_list)
     - Runtime/budget as-is
     - Decade from year
     - Bankability: lookup each actor in bankability_map
       -> fallback to genre-average if actor not found
  4. Model.predict(X_vec) -> clamp rating to [0, 10]
  5. For box office: multiply by festival scenario multiplier
  6. Return result dict
         |
         v
Streamlit renders:
  - Gauge chart for rating
  - Box office in INR Crore
  - Scenario comparison bar chart
  - Bankability info + fallback warnings
```

## Data Flow Diagram - Training

```
Kaggle CSVs
    |
    v
load_imdb_india() -> df_imdb (11,500+ movies with ratings)
load_bollywood_boxoffice() -> df_box (1,000+ movies with box office)
    |
    v
fuzzy_join_datasets(box, extra, title) -> box_with_years
fuzzy_join_datasets(box_with_years, imdb, title+year) -> joined (81.2% match rate)
    |
    v
clean + TMDb enrichment (plot summaries, release dates, poster paths)
    |
    +--> Rating dataset (df_imdb): 11,500+ rows with ratings
    |     |
    |     v
    |   build_feature_matrix() -> X (36 cols), y_rating
    |   train_and_compare(9 models, 5-fold CV) -> comparison_rating.csv
    |   -> GradientBoosting wins -> save best_rating_model.pkl
    |
    +--> Box office dataset (joined): 600+ rows with box office
    |     |
    |     +--> baseline: build_feature_matrix() -> X (31 cols), y_box
    |     |   train_and_compare() -> comparison_baseline.csv
    |     |   -> XGBoost wins (MAE=35.2 Cr)
    |     |
    |     +--> with Bankability: X + avg_bankability_score (32 cols)
    |         train_and_compare() -> comparison_with_bank.csv
    |         -> XGBoost wins (MAE=32.1 Cr, 8.7% improvement)
    |
    +--> Bankability scores: compute_bankability_scores() -> 1,010 individuals
    +--> Chemistry pairs: detect_chemistry_pairs() -> top 10
    +--> Plot sentiment: score_plot_sentiment() + genre_conditional_correlation()
    +--> Release timing: compute_festival_features() + compute_clash_feature()
    +--> SHAP: shap.Explainer -> summary_plot()
```
