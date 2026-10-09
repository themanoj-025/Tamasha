# Tamasha — Startup Flow

Tamasha has two independently deployed entry points. Both depend on the
installed `tamasha` package and the artifacts in `models/` + `data/processed/`.

## Entry point A — FastAPI service (`api.main:app`)

Deployed on Render (`render.yaml`: `gunicorn -w 1 -k uvicorn.workers.UvicornWorker api.main:app`).

1. **Import phase** — `api/main.py` imports `tamasha.config` (env loading) and
   `tamasha.predict` (module-level constants, no heavy I/O yet).
2. **App construction** — `FastAPI()` instance created; CORS, SlowAPI
   (rate-limit), and Prometheus instrumentation middleware registered.
3. **Lifespan startup** (`asynccontextmanager`) —
   a. `PredictionService()` is constructed: loads the box-office + rating
      models and feature manifests from `models/` (env-configurable paths),
      initializes the cache layer.
   b. The instance is stored in app state and exposed to routers via
      `get_prediction_service()`.
4. **Router registration** — `api.routers.{predict, network, model_info}` are
   included (`include_router`), each declaring Pydantic schemas from
   `api/schemas.py`.
5. **Ready to serve** — `/docs` (OpenAPI), rate-limited + auth-guarded
   prediction endpoints live.

## Entry point B — Streamlit app (`app/streamlit_app.py`)

Deployed on Docker (`CMD ["streamlit", "run", "app/streamlit_app.py", ...]`).

1. `streamlit_app.py` imports `tamasha.predict.PredictionService` and
   instantiates it once at import time (model + manifest loading).
2. Multipage router (`st.navigation`-style dispatch in the entry script)
   registers the four page modules from `app/pages/` (`_1_Predict_a_Release`,
   `_2_Star_Network_Explorer`, `_3_Industry_Trends`, `_4_Model_Performance`).
3. Each page lazily imports `tamasha.*` facades on first render and calls
   `show()`; shared widgets come from `app/components/`; theme from
   `app/assets/theme.css` and `.streamlit/secrets.toml` (gitignored; falls
   back to `.example`).
4. The page selector renders and the app serves on port 8501.

## Entry point C — Training pipeline (operational)

`python -m tamasha.train_pipeline` (Makefile `make train`):
raw CSVs → `data.cleaning/joining/enrichment` → `features.movie_features`
→ `models.*` fit + `model_selection` comparison → artifacts written to
`models/` and comparison reports to `reports/`.

## What must exist at startup

- `models/best_boxoffice_model.pkl`, `models/best_rating_model.pkl`,
  `models/*_features.json`, `models/director_encoder.pkl`
- `data/processed/*.parquet` (for training; prediction falls back to
  embedded/manifest data where applicable)
- Env keys consumed by `tamasha.config` — see `.env.example` /
  `.streamlit/secrets.toml.example`
