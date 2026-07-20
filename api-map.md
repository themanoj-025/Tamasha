# Tamasha — API Inventory

## FastAPI Endpoints

| # | Method | Route | Input Schema | Output Schema | Auth | Rate Limit | Purpose |
|:-:|--------|-------|-------------|--------------|:----:|:----------:|---------|
| 1 | `GET` | `/health` | — | `dict` | ❌ Exempt | 60/min | Health check + version |
| 2 | `POST` | `/predict-rating` | `PredictRatingRequest` | `PredictRatingResponse` | ✅ Required | 60/min | Predict IMDB rating |
| 3 | `POST` | `/predict-boxoffice` | `PredictBoxOfficeRequest` | `PredictBoxOfficeResponse` | ✅ Required | 60/min | Predict box office with scenarios |
| 4 | `GET` | `/actor/{name}` | Path param `name: str` | `ActorInfoResponse` | ✅ Required | 60/min | Bankability Score + chemistry |
| 5 | `GET` | `/model-info` | — | `ModelInfoResponse` | ✅ Required | 60/min | Deployed model metadata |
| 6 | `GET` | `/metrics` | — | `text/plain` | ❌ Exempt | None | Prometheus metrics (internal scraper) |
| 7 | `GET` | `/docs` | — | HTML | ❌ Exempt | 60/min | Swagger UI documentation |

---

## Request Schemas

### PredictRatingRequest

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `title` | `str` | Required | Movie title (for display) |
| `genres` | `list[str]` | Required | List of genres |
| `cast` | `list[str]` | Required | List of cast members |
| `director` | `str` | `"Unknown"` | Director name |
| `budget_inr` | `float` | `0.0` | Budget in rupees |
| `runtime_minutes` | `int` | `150` | Runtime in minutes |

### PredictBoxOfficeRequest

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `title` | `str` | Required | Movie title |
| `genres` | `list[str]` | Required | List of genres |
| `cast` | `list[str]` | Required | List of cast members |
| `director` | `str` | `"Unknown"` | Director name |
| `budget_inr` | `float` | `0.0` | Budget in rupees |
| `runtime_minutes` | `int` | `150` | Runtime in minutes |
| `release_window` | `str` | `"Normal"` | Release window scenario |

---

## Response Schemas

### PredictRatingResponse

| Field | Type | Description |
|-------|------|-------------|
| `title` | `str` | Movie title |
| `predicted_rating` | `float` | Predicted IMDB rating (0-10) |
| `model_name` | `str` | Winning model name |
| `model_mae` | `float` | Model's MAE on validation |

### PredictBoxOfficeResponse

| Field | Type | Description |
|-------|------|-------------|
| `title` | `str` | Movie title |
| `predicted_boxoffice_cr` | `float` | Predicted box office in ₹ Crore |
| `model_name` | `str` | Winning model name |
| `model_mae` | `float` | Model's MAE (in ₹ Crore) |
| `scenarios` | `dict[str, float]` or `None` | Per-window predictions |

### ActorInfoResponse

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Actor name |
| `bankability_score` | `float` | Bankability Score (0-1) |
| `film_count` | `int` | Number of films in dataset |
| `top_chemistry_pairs` | `list[dict]` | Top chemistry pairings |

Each chemistry pair dict:
```json
{
  "actor": "Kajol",
  "joint_films": 7,
  "chemistry_score": 0.1245
}
```

### ModelInfoResponse

| Field | Type | Description |
|-------|------|-------------|
| `rating_model` | `dict` | Rating model: name, algorithm, mae, rmse, r2, features |
| `boxoffice_model` | `dict` | Box office model: name, algorithm, mae, rmse, r2, features |

---

## Shared Prediction Logic

Both the FastAPI endpoints and the Streamlit dashboard use the **same** underlying
`PredictionService` class from `tamasha.predict`:

- **FastAPI**: `PredictionService` is built once at startup via the lifespan context
  manager and injected into route handlers via FastAPI's `Depends()`.
- **Streamlit**: `PredictionService` is cached via `st.cache_resource` so models
  aren't reloaded on every rerun.
- Both access the same model files (`models/*.pkl`) directly — there is no
  HTTP boundary between the dashboard and the API.

```python
# api/routers/predict.py — receives service via dependency injection
from fastapi import Depends
from api.main import get_prediction_service
from tamasha.predict import PredictionService

@router.post("/predict-rating")
async def predict_rating_endpoint(...,
    service: PredictionService = Depends(get_prediction_service)):
    return await service.predict_rating_async(title, genres, cast, ...)

# app/pages/_1_Predict_a_Release.py — uses cached singleton
from tamasha.predict import PredictionService
@st.cache_resource
def get_service():
    svc = PredictionService()
    svc.load()
    return svc
```

---

## Authentication

All endpoints except `/health`, `/docs`, `/openapi.json`, and `/redoc` require the
`X-API-Key` header:

```bash
curl -H "X-API-Key: tamasha-dev-key-2026" http://localhost:8000/predict-rating ...
```

Configure via `API_KEY` env var (default dev key: `tamasha-dev-key-2026`).
Set a strong random value in production.

---

## Rate Limiting

All endpoints are rate-limited to **60 requests per minute** (configurable via
`RATE_LIMIT` env var). The rate limit is keyed by the `X-API-Key` header value
(or client IP if no key is provided).

Rate-limit headers are included on every response:
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Remaining requests in current window

When exceeded, returns `429 Too Many Requests`.

## Request Body Size Limits

All POST endpoints are protected by a **100KB** maximum request body size limit.
Requests exceeding this threshold receive HTTP 413 (Payload Too Large) with the
standard error response shape:

```json
{
  "error_code": "PAYLOAD_TOO_LARGE",
  "message": "Request body exceeds 100KB limit",
  "request_id": "req_abc123"
}
```

This protects against resource exhaustion from oversized payloads.

The 100KB threshold is generous for this API's typical payload (largest legitimate
payload is ~50KB even with 100+ actor names).

---

## Error Handling

| HTTP Status | Meaning | Example Scenario |
|:-----------:|---------|-----------------|
| `200` | Success | All endpoints |
| `401` | Unauthorized | Missing or invalid X-API-Key header |
| `404` | Not found | Actor not in Bankability dataset |
| `413` | Payload Too Large | Request body exceeds 100KB limit |
| `429` | Too Many Requests | Rate limit exceeded (60 req/min) |
| `503` | Service unavailable | Model not trained (missing .pkl) or SHA-256 integrity check failed |
| `500` | Internal error | Prediction failure, data error |

---

## API Dependencies

| Endpoint | Underlying Functions | Data Files Needed |
|----------|---------------------|-------------------|
| `/predict-rating` | `predict_rating()` | `models/best_rating_model.pkl`, `models/rating_features.json` |
| `/predict-boxoffice` | `predict_boxoffice()` | `models/best_boxoffice_model.pkl`, `models/boxoffice_features.json`, `reports/bankability_scores.csv` |
| `/actor/{name}` | `get_actor_info()` | `reports/bankability_scores.csv`, `reports/chemistry_pairs.csv` |
| `/model-info` | `get_model_info()` | `reports/model_comparison_*.csv` |

All data files are lazy-loaded and cached in memory after first access.

---

## Testing

API tests use FastAPI's `TestClient` across multiple test files:

| Test File | Focus | Test Count |
|-----------|-------|:----------:|
| `tests/test_api.py` | Smoke tests — health, prediction endpoints | 10+ |
| `tests/test_api_contract.py` | Schema validation — negative budget, oversized genres, empty cast, consistent error shapes | 10+ |
| `tests/test_auth.py` | API key validation, exempt paths, CORS, request body limits, /health degraded | 17 |
| `tests/test_predict_service.py` | PredictionService concurrency (20 threads), edge cases, graceful degradation | 15 |

```python
from fastapi.testclient import TestClient
from api.main import app
import os

# Protected endpoints require X-API-Key header
client = TestClient(app)
client.headers["X-API-Key"] = os.getenv("API_KEY", "tamasha-dev-key-2026")

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ("ok", "degraded")

def test_predict_rating():
    response = client.post("/predict-rating", json={...})
    assert response.status_code in (200, 503)  # 503 if model not trained

def test_auth_blocks_no_key():
    # Without API key, protected endpoints return 401
    noauth = TestClient(app)
    response = noauth.post("/predict-rating", json={...})
    assert response.status_code == 401
```
