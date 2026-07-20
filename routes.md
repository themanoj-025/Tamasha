# Tamasha — Route Map

## 1. Streamlit Pages

The Streamlit app uses a multi-page structure with page files in `app/pages/` prefixed with `_1_`, `_2_`, etc. for ordering.

### Navigation (Sidebar)

```python
pages = ["Predict a Release", "Star Network Explorer", "Industry Trends", "Model Performance"]
st.sidebar.selectbox("Navigate", pages)
```

### Page Routing Table

| # | Page Name | File | Display Name | Purpose |
|:-:|-----------|------|-------------|---------|
| 1 | `1_Predict_a_Release` | `app/pages/_1_Predict_a_Release.py` | 🔮 Predict a Release | Movie form → predicted rating + box office + scenario comparison |
| 2 | `2_Star_Network_Explorer` | `app/pages/_2_Star_Network_Explorer.py` | ⭐ Star Network Explorer | Searchable actor graph, Bankability Scores, chemistry pairs |
| 3 | `3_Industry_Trends` | `app/pages/_3_Industry_Trends.py` | 📈 Industry Trends | Genre trends, festival analysis, plot-tone findings, poster CV results |
| 4 | `4_Model_Performance` | `app/pages/_4_Model_Performance.py` | 📊 Model Performance | Comparison charts, SHAP plots (global + per-prediction), model info table |

### Page Entry Points

Each page exposes a `show()` function called by the main app:

```python
# In streamlit_app.py:
PAGES = {
    "Predict a Release": page_1.show,
    "Star Network Explorer": page_2.show,
    "Industry Trends": page_3.show,
    "Model Performance": page_4.show,
}
PAGES[selected_page]()
```

### Page Dependencies

| Page | Depends On |
|------|-----------|
| Page 1 | `tamasha.predict.predict_rating()`, `tamasha.predict.predict_boxoffice()`, `plotly.graph_objects` |
| Page 2 | `tamasha.predict.get_actor_info()`, `tamasha.predict.get_bankability_scores()`, `tamasha.predict.get_chemistry_pairs()`, `networkx`, `pyvis` |
| Page 3 | `tamasha.data.loaders.load_imdb_india()`, `tamasha.predict.get_comparison_csv()`, `reports/genre_tone_correlation.csv`, `reports/release_timing_analysis.md`, `reports/chemistry_pairs.csv` |
| Page 4 | `tamasha.predict.get_comparison_csv()`, `tamasha.predict.get_model_info()`, `reports/figures/shap_*.png` |

---

## 2. FastAPI Routes

### Route Table

| Method | Path | Handler | Module | Auth | Purpose |
|--------|------|---------|--------|:----:|---------|
| `GET` | `/health` | `health()` | `api/main.py` | ❌ Exempt | Health check + version |
| `POST` | `/predict-rating` | `predict_rating_endpoint()` | `api/routers/predict.py` | ✅ X-API-Key | Predict IMDB rating |
| `POST` | `/predict-boxoffice` | `predict_boxoffice_endpoint()` | `api/routers/predict.py` | ✅ X-API-Key | Predict box office collection |
| `GET` | `/actor/{name}` | `get_actor_info_endpoint()` | `api/routers/network.py` | ✅ X-API-Key | Actor Bankability + chemistry |
| `GET` | `/model-info` | `get_model_info_endpoint()` | `api/routers/model_info.py` | ✅ X-API-Key | Deployed model metadata |
| `GET` | `/docs` | — | — | ❌ Exempt | Swagger UI |

### Route Details

#### 1. `GET /health`

**Response** (200):
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

#### 2. `POST /predict-rating`

**Request** (`PredictRatingRequest`):
```json
{
  "title": "My Bollywood Film",
  "genres": ["Drama", "Romance"],
  "cast": ["Shah Rukh Khan", "Deepika Padukone"],
  "director": "Sanjay Leela Bhansali",
  "budget_inr": 800000000,
  "runtime_minutes": 150
}
```

**Response** (`PredictRatingResponse`, 200):
```json
{
  "title": "My Bollywood Film",
  "predicted_rating": 7.42,
  "model_name": "GradientBoosting (tuned)",
  "model_mae": 0.9534,
  "shap_explanations": {
    "top_features": [
      {"feature": "genre_Drama", "value": 1.0, "shap_value": 0.42},
      {"feature": "director_encoded", "value": 142.0, "shap_value": 0.31},
      {"feature": "budget_inr", "value": 800000000.0, "shap_value": 0.18},
      {"feature": "cast_size", "value": 8.0, "shap_value": 0.12},
      {"feature": "runtime_minutes", "value": 150.0, "shap_value": 0.08}
    ]
  }
}
```

**Errors**:
- `401`: Missing or invalid X-API-Key header (all protected endpoints)
- `413`: Request body exceeds 100KB limit
- `503`: Model not found (run `make train`) or SHA-256 integrity check failed
- `500`: Prediction failure

#### 3. `POST /predict-boxoffice`

**Request** (`PredictBoxOfficeRequest`):
```json
{
  "title": "My Bollywood Film",
  "genres": ["Drama", "Romance"],
  "cast": ["Shah Rukh Khan", "Deepika Padukone"],
  "director": "Sanjay Leela Bhansali",
  "budget_inr": 800000000,
  "runtime_minutes": 150,
  "release_window": "Diwali"
}
```

**Response** (`PredictBoxOfficeResponse`, 200):
```json
{
  "title": "My Bollywood Film",
  "predicted_boxoffice_cr": 175.4,
  "model_name": "XGBoost (tuned)",
  "model_mae": 73.6,
  "scenarios": {
    "Normal": 175.4,
    "Diwali": 219.3,
    "Eid": 207.0,
    "Christmas": 196.4,
    "Independence Day": 189.4,
    "Republic Day": 184.2,
    "New Year": 193.0
  }
}
```

**Errors**:
- `401`: Missing or invalid X-API-Key header
- `413`: Request body exceeds 100KB limit
- `503`: Model not found or SHA-256 integrity check failed
- `500`: Prediction failure

#### 4. `GET /actor/{name}`

**Response** (`ActorInfoResponse`, 200):
```json
{
  "name": "Shah Rukh Khan",
  "bankability_score": 0.8712,
  "film_count": 42,
  "top_chemistry_pairs": [
    {
      "actor": "Kajol",
      "joint_films": 7,
      "chemistry_score": 0.1245
    },
    {
      "actor": "Deepika Padukone",
      "joint_films": 5,
      "chemistry_score": 0.0891
    }
  ]
}
```

**Errors**:
- `401`: Missing or invalid X-API-Key header
- `404`: Actor not found in Bankability dataset

#### 5. `GET /model-info`

**Response** (`ModelInfoResponse`, 200):
```json
{
  "rating_model": {
    "name": "GradientBoosting (tuned)",
    "algorithm": "GradientBoosting",
    "mae": 0.9534,
    "rmse": 1.2228,
    "r2": 0.2162,
    "features_used": ["genre", "cast_size", "director", "runtime", "budget", "decade"],
    "version": "v1",
    "train_date": "2026-01-15T10:30:00Z",
    "tuning_iterations": 15
  },
  "boxoffice_model": {
    "name": "XGBoost (tuned)",
    "algorithm": "XGBoost",
    "mae": 73.6,
    "rmse": 36.9,
    "r2": 0.21,
    "features_used": ["genre", "cast_size", "director", "runtime", "budget", "decade", "avg_bankability_score"],
    "version": "v1",
    "train_date": "2026-01-15T10:35:00Z",
    "tuning_iterations": 15
  }
}
```

---

## 3. Router Inclusion in FastAPI

```python
# api/main.py
from api.routers import predict, network, model_info

app.include_router(predict.router)     # /predict-rating, /predict-boxoffice
app.include_router(network.router)     # /actor/{name}
app.include_router(model_info.router)  # /model-info
```

Each router:

```python
# api/routers/predict.py
router = APIRouter(prefix="", tags=["predict"])

# api/routers/network.py
router = APIRouter(prefix="", tags=["network"])

# api/routers/model_info.py
router = APIRouter(prefix="", tags=["model-info"])
```

No prefix is applied — paths are absolute.

---

## 4. Request Body Size Limits

All POST endpoints enforce a **100KB** maximum request body to prevent resource
exhaustion. Requests exceeding this threshold receive HTTP 413 (Payload Too Large)
with the standard error response shape:

```json
{
  "error_code": "PAYLOAD_TOO_LARGE",
  "message": "Request body exceeds 100KB limit",
  "request_id": "req_abc123"
}
```

The 100KB threshold is generous for this API's typical payload (largest legitimate
payload is ~50KB even with 100+ actor names). The limit is enforced via incremental
body-reading middleware that aborts once the threshold is exceeded.

## 5. Authentication

All protected endpoints require the `X-API-Key` header:

```bash
curl -H "X-API-Key: tamasha-dev-key-2026" http://localhost:8000/predict-rating ...
```

The health endpoint (`/health`), documentation endpoints (`/docs`, `/openapi.json`, `/redoc`),
and `/metrics` are exempt from authentication.

Configured via `API_KEY` env var in `.env`. Default dev key: `tamasha-dev-key-2026`.
Change to a strong random value in production.

## 6. Rate Limiting

All endpoints are rate-limited to **60 requests per minute** via slowapi (configurable
via `RATE_LIMIT` env var, e.g. `120/minute`). Limits are keyed by the `X-API-Key` header
value, falling back to client IP. When exceeded, returns `429 Too Many Requests`.

Rate-limit headers on every response:
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Remaining requests in current window

## 7. Request ID

Every response includes an `X-Request-ID` header (format: `req_<uuid_hex>`).
Request IDs are passed to `structlog` context vars for structured log correlation.
Even 401/413 error responses include the `X-Request-ID` header for debuggability.

## 8. CORS Configuration

CORS is restricted to the origins configured in `ALLOWED_ORIGINS` env var:

```python
# Default: ALLOWED_ORIGINS=http://localhost:8501,http://localhost:8000
# In production: https://your-dashboard.streamlit.app,https://your-api.onrender.com
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,  # parsed from comma-separated env var
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 9. /metrics Endpoint

`GET /metrics` returns Prometheus text-exposition format with:
- Request count by endpoint
- Latency histogram (p50/p95/p99)
- Error rate by endpoint

This endpoint is **NOT** protected by X-API-Key authentication (designed for internal
Prometheus scraping). Exposed at `/metrics` with `include_in_schema=False`.

## 10. Observability (Local Docker)

```bash
docker compose -f docker-compose.observability.yml up -d
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)
```

Grafana auto-provisions a dashboard with request count, latency percentiles, and
error rate by endpoint. Dashboard JSON: `ops/grafana/dashboards/tamasha-overview.json`.

## 11. Streamlit Cloud Entry Point

- **File**: `app/streamlit_app.py`
- **Command**: `streamlit run app/streamlit_app.py`
- **Secrets**: `.streamlit/secrets.toml` for TMDb API credentials
- **System deps**: `packages.txt` (`libgl1-mesa-glx`, `libglib2.0-0` for OpenCV)
- **NLTK**: VADER lexicon downloaded at startup

## 12. Render Deployment (FastAPI)

- **Blueprint file**: `render.yaml`
- **Service**: Web Service
- **Build command**: `pip install -e . && python -m nltk.downloader vader_lexicon -d /opt/nltk_data`
- **Start command**: `gunicorn -w 1 -k uvicorn.workers.UvicornWorker api.main:app --bind 0.0.0.0:$PORT --timeout 120`
- **Env vars**: `TMDB_API_KEY`, `TMDB_ACCESS_TOKEN`, `API_KEY`, `ALLOWED_ORIGINS`, `RATE_LIMIT`
- **Python version**: 3.11
