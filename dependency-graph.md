# Tamasha — Dependency Graph

## Critical File: `tamasha.config.settings`

**Every module** imports from `config.py`. Changes here affect the entire codebase.

```
config.py (pydantic-settings)
  │
  ├──▶ data/loaders.py        ← settings.DATA_RAW, DATA_PROCESSED
  ├──▶ data/joining.py        ← settings (via caller)
  ├──▶ data/cleaning.py       ← (uses config constants rarely)
  ├──▶ data/enrichment.py     ← settings.DATA_PROCESSED
  ├──▶ features/movie_features.py
  ├──▶ models/model_selection.py  ← settings.MODEL_SELECTION_METRIC
  ├──▶ models/rating_model.py     ← settings.MODELS_DIR, REPORTS_DIR
  ├──▶ models/boxoffice_model.py  ← settings.MODELS_DIR, REPORTS_DIR
  ├──▶ nlp/plot_sentiment.py      ← settings (indirectly)
  ├──▶ cv/poster_classifier.py    ← settings.DATA_PROCESSED
  ├──▶ network/bankability_score.py  ← settings.BANKABILITY_DECAY_HALFLIFE
  ├──▶ timing/festival_calendar.py   ← settings
  ├──▶ timing/release_scenario.py
  ├──▶ evaluation/metrics.py
  ├──▶ predict.py               ← settings.MODELS_DIR, REPORTS_DIR
  ├──▶ train_pipeline.py        ← settings everywhere
  └──▶ app/streamlit_app.py     ← settings (for theme/logo)
```

---

## Import Graph (by Module)

### `tamasha.data.loaders`

```
loaders.py ──▶ pandas
              config (for settings.DATA_RAW)
```

**No imports from other tamasha modules.**

### `tamasha.data.joining`

```
joining.py ──▶ pandas
              rapidfuzz (fuzz, process)
              config (optional)
```

**No imports from other tamasha modules.**

### `tamasha.data.cleaning`

```
cleaning.py ──▶ pandas
               numpy
               logging
               re
```

**No imports from other tamasha modules.**

### `tamasha.data.enrichment`

```
enrichment.py ──▶ pandas
                 requests
                 dotenv
                 json, os, time, pathlib
                 config
```

**No imports from other tamasha modules.**

### `tamasha.features.movie_features`

```
movie_features.py ──▶ pandas
                     numpy
                     sklearn.preprocessing (LabelEncoder, MultiLabelBinarizer)
                     config
```

**No imports from other tamasha modules.**

### `tamasha.features.cast_crew_network`

```
cast_crew_network.py ──▶ networkx
                        pandas
```

**No imports from other tamasha modules.**

### `tamasha.models.model_selection`

```
model_selection.py ──▶ sklearn.linear_model (LinearRegression, Ridge, Lasso)
                      sklearn.tree (DecisionTreeRegressor)
                      sklearn.ensemble (RandomForestRegressor, GradientBoostingRegressor)
                      sklearn.model_selection (KFold)
                      sklearn.metrics (mean_absolute_error, mean_squared_error, r2_score)
                      numpy, pandas, joblib, time, pathlib
                      config
```

**No imports from other tamasha modules.**
**Critical**: Dynamic import of xgboost, lightgbm, catboost.

### `tamasha.models.rating_model`

```
rating_model.py ──▶ tamasha.config
                   tamasha.features.movie_features (build_feature_matrix)
                   tamasha.models.model_selection (get_all_models, save_model, train_and_compare)
                   pandas
```

### `tamasha.models.boxoffice_model`

```
boxoffice_model.py ──▶ tamasha.config
                      tamasha.features.movie_features (build_feature_matrix)
                      tamasha.models.model_selection (get_all_models, save_model, train_and_compare)
                      pandas
```

### `tamasha.nlp.plot_sentiment`

```
plot_sentiment.py ──▶ nltk (VADER)
                      pandas, numpy
                      config
```

**No imports from other tamasha modules.**

### `tamasha.cv.poster_classifier`

```
poster_classifier.py ──▶ cv2 (OpenCV)
                         sklearn.ensemble (RandomForestClassifier)
                         sklearn.metrics, sklearn.model_selection
                         sklearn.preprocessing (StandardScaler)
                         numpy
                         config
```

**No imports from other tamasha modules.**

### `tamasha.network.bankability_score`

```
bankability_score.py ──▶ pandas, numpy
                         config
```

**No imports from other tamasha modules.**

### `tamasha.network.chemistry_pairs`

```
chemistry_pairs.py ──▶ tamasha.network.bankability_score (compute_bankability_scores)
                       pandas, numpy
                       scipy.stats
```

### `tamasha.timing.festival_calendar`

```
festival_calendar.py ──▶ pandas, numpy
                         datetime
                         config
```

**No imports from other tamasha modules.**

### `tamasha.timing.release_scenario`

```
release_scenario.py ──▶ pandas, numpy
                        dataclasses
```

**No imports from other tamasha modules.**

### `tamasha.evaluation.metrics`

```
metrics.py ──▶ matplotlib
               pandas, numpy
               plotly (optional)
               shap (optional)
               pathlib
```

**No imports from other tamasha modules.**

### `tamasha.predict` — **Central Hub**

```
predict.py ──▶ tamasha.config
              tamasha.features.movie_features (build_feature_matrix)
              tamasha.models.model_selection (load_model)
              pandas, numpy, json, pathlib
```

**This is the only module that both `app/` and `api/` import from `src/tamasha/`.**

### `tamasha.train_pipeline` — **Orchestrator**

```
train_pipeline.py ──▶ tamasha.config
                     tamasha.data.loaders (load_imdb_india, load_bollywood_boxoffice)
                     tamasha.data.joining (fuzzy_join_datasets, generate_join_quality_report)
                     tamasha.data.enrichment (enrich_dataset)
                     tamasha.features.movie_features (build_feature_matrix)
                     tamasha.models.model_selection (get_all_models)
                     tamasha.models.rating_model (train_rating_model)
                     tamasha.models.boxoffice_model (train_boxoffice_model, _compute_cast_avg_bankability)
                     tamasha.network.bankability_score (compute_bankability_scores)
                     tamasha.network.chemistry_pairs (detect_chemistry_pairs)
                     tamasha.nlp.plot_sentiment (score_plot_sentiment, genre_conditional_correlation)
                     tamasha.evaluation.metrics (plot_model_comparison, plot_predicted_vs_actual)
                     sklearn.model_selection (train_test_split)
                     pandas, numpy, json, sys
```

**Most complex dependency chain in the project.**

---

## `app/` Dependencies

### `app/streamlit_app.py`

```
streamlit_app.py ──▶ streamlit
                     app.pages._1_Predict_a_Release (show)
                     app.pages._2_Star_Network_Explorer (show)
                     app.pages._3_Industry_Trends (show)
                     app.pages._4_Model_Performance (show)
                     app.components.metric_cards
                     tamasha.config
                     pathlib
```

### `app/pages/_1_Predict_a_Release.py`

```
_1_Predict_a_Release.py ──▶ streamlit
                             pandas
                             plotly.graph_objects
                             tamasha.predict (predict_rating, predict_boxoffice)
```

### `app/pages/_2_Star_Network_Explorer.py`

```
_2_Star_Network_Explorer.py ──▶ streamlit
                                 pandas
                                 plotly.graph_objects
                                 networkx
                                 tamasha.predict (get_actor_info, get_bankability_scores, get_chemistry_pairs)
```

### `app/pages/_3_Industry_Trends.py`

```
_3_Industry_Trends.py ──▶ streamlit
                           pandas
                           plotly.express, plotly.graph_objects
                           tamasha.data.loaders (load_imdb_india)
                           tamasha.predict (get_comparison_csv)
                           pathlib
```

### `app/pages/_4_Model_Performance.py`

```
_4_Model_Performance.py ──▶ streamlit
                             pandas
                             plotly.graph_objects
                             tamasha.predict (get_comparison_csv, get_model_info)
                             pathlib
```

### `app/components/metric_cards.py`

```
metric_cards.py ──▶ streamlit
```

**No imports from tamasha.**

---

## `api/` Dependencies

### `api/main.py`

```
main.py ──▶ fastapi (FastAPI, CORSMiddleware)
            tamasha.config
            api.routers.predict
            api.routers.network
            api.routers.model_info
```

### `api/routers/predict.py`

```
predict.py ──▶ fastapi (APIRouter, HTTPException)
               api.schemas (PredictRatingRequest, PredictRatingResponse, etc.)
               tamasha.predict (predict_rating, predict_boxoffice)
```

### `api/routers/network.py`

```
network.py ──▶ fastapi (APIRouter, HTTPException)
               api.schemas (ActorInfoResponse)
               tamasha.predict (get_actor_info)
```

### `api/routers/model_info.py`

```
model_info.py ──▶ fastapi (APIRouter)
                  api.schemas (ModelInfoResponse)
                  tamasha.predict (get_model_info)
```

---

## `tests/` Dependencies

| Test File | Tests Module(s) |
|-----------|----------------|
| `test_joining.py` | `tamasha.data.joining` |
| `test_cleaning.py` | `tamasha.data.cleaning` |
| `test_features.py` | `tamasha.features.movie_features` |
| `test_model_selection.py` | `tamasha.models.model_selection` |
| `test_bankability_score.py` | `tamasha.network.bankability_score` |
| `test_chemistry_pairs.py` | `tamasha.network.chemistry_pairs` |
| `test_festival_calendar.py` | `tamasha.timing.festival_calendar` |
| `test_api.py` | `api.main`, `tamasha.predict` |

---

## Circular Dependency Risk

**None detected.** The dependency graph is a directed acyclic graph (DAG):

```
config
  │
  ├──▶ data/ ──▶ (leaf, no internal imports)
  ├──▶ features/ ──▶ (leaf)
  ├──▶ models/model_selection ──▶ (leaf)
  ├──▶ models/rating_model ──▶ features ──▶ model_selection
  ├──▶ models/boxoffice_model ──▶ features ──▶ model_selection
  ├──▶ network/bankability_score ──▶ (leaf)
  ├──▶ network/chemistry_pairs ──▶ bankability_score
  ├──▶ nlp/ ──▶ (leaf)
  ├──▶ cv/ ──▶ (leaf)
  ├──▶ timing/ ──▶ (leaf)
  ├──▶ evaluation/ ──▶ (leaf)
  ├──▶ predict ──▶ features, model_selection
  └──▶ train_pipeline ──▶ all of the above
```

---

## File Change Impact Analysis

| File Changed | Rebuild/Test Impact | Notes |
|-------------|-------------------|-------|
| `config.py` | **EVERYTHING** | Change paths/thresholds → retrain + retest all |
| `data/loaders.py` | All data modules, all models, train_pipeline | Column name changes break everything downstream |
| `data/joining.py` | train_pipeline, test_joining.py | Join logic change affects all downstream analysis |
| `data/cleaning.py` | train_pipeline, test_cleaning.py | |
| `data/enrichment.py` | train_pipeline | TMDb API changes affect sentiment + timing + poster stages |
| `features/movie_features.py` | rating_model, boxoffice_model, predict, test_features | Feature count changes need model retraining |
| `features/cast_crew_network.py` | (standalone, only used by network analysis) | |
| `models/model_selection.py` | **ALL model training**, predict, train_pipeline | Core — adding models, changing CV logic |
| `models/rating_model.py` | train_pipeline | |
| `models/boxoffice_model.py` | train_pipeline | |
| `nlp/plot_sentiment.py` | train_pipeline, _3_Industry_Trends | |
| `cv/poster_classifier.py` | train_pipeline, _3_Industry_Trends | |
| `network/bankability_score.py` | boxoffice_model, chemistry_pairs, train_pipeline, predict | **HIGH IMPACT** — affects model + dashboard + API |
| `network/chemistry_pairs.py` | train_pipeline, predict, _2_Star_Network | |
| `timing/festival_calendar.py` | train_pipeline, test_festival_calendar, _3_Industry_Trends | |
| `timing/release_scenario.py` | _1_Predict_a_Release | |
| `evaluation/metrics.py` | train_pipeline | |
| `predict.py` | **ALL app pages + ALL API routers** | **HIGHEST IMPACT** after config |
| `train_pipeline.py` | (no one imports it) | Only run as script/`make train` |
| `app/` pages | (no one imports them except streamlit_app.py) | |
| `api/` routers | api/main.py | |
| `api/schemas.py` | api/routers | |
