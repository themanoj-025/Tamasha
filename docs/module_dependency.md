# Tamasha — Module Dependency Map

## Core package (src/tamasha) internal dependencies

```
tamasha.config          ← imported by every module that needs settings
tamasha.exceptions      ← imported by data, features, models, predict
tamasha.cache           ← used only by predict.py / api routers (prediction caching)
tamasha.data.loaders    ← used by features.movie_features, predict, train_pipeline
tamasha.data.cleaning   ← used by features.movie_features, train_pipeline
tamasha.data.joining    ← used by features.movie_features, train_pipeline
tamasha.data.enrichment ← used by predict (posters/photos), train_pipeline
tamasha.features.movie_features      ← depends on data.* and features.cast_crew_network
tamasha.models.*        ← depends on features.* ; model_selection depends on models.*
tamasha.network.*       ← depends on data.loaders/joining + features.cast_crew_network
tamasha.nlp.plot_sentiment ← depends on data.loaders (plot text)
tamasha.cv.poster_classifier ← depends on config (paths) only
tamasha.timing.*        ← pure date/calendar logic, depends only on config
tamasha.evaluation.metrics ← depends on models.* (used by model_selection)
tamasha.predict         ← **facade**: depends on models, network, features, data, cache
tamasha.train_pipeline  ← depends on data, features, models, evaluation
```

## Interface layer → core

```
api.main        → tamasha.config, tamasha.predict   (PredictionService DI)
api.routers.*   → api.main (get_prediction_service), api.schemas, tamasha.cache
app.streamlit_app → tamasha.predict (PredictionService), app.pages.*
app.pages.*     → tamasha.* (predict, data.enrichment, data.loaders, config)
app.components.* → pure UI, no core imports
```

## Dependency rules (why)

- `tamasha.predict` is the **only** sanctioned business-logic entry for both
  front-ends — this prevents interface modules from reaching into internals.
- `data.*` never imports `models.*` or `predict` (no upward coupling) —
  domain layers depend only downward or on shared infra (`config`, `exceptions`).
- `timing` and `cv` are deliberately leaf modules with no peers — they are
  pulled in only by `predict`/`train_pipeline`.
- No circular imports exist between package domains. The only historical
  risk point (FastAPI app factory ↔ routers) is resolved via `get_prediction_service`
  dependency provider in `api.main`.

## External dependencies (for the dependency graph)

- **FastAPI + uvicorn/gunicorn** — API hosting
- **Streamlit** — UI hosting
- **pandas / numpy / scikit-learn / joblib** — data + modeling
- **slowapi** — rate limiting; **prometheus-fastapi-instrumentator** — metrics
- **structlog** — structured logging
- **SQLite** (stdlib `sqlite3`) — prediction cache
