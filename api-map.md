# Tamasha — API Inventory

## FastAPI Endpoints

| # | Method | Route | Input Schema | Output Schema | Auth | Rate Limit | Purpose |
|:-:|--------|-------|-------------|--------------|:----:|:----------:|---------|
| 1 | `GET` | `/health` | — | `dict` | ❌ Exempt | 60/min | Health check + version |
| 2 | `POST` | `/predict-rating` | `PredictRatingRequest` | `PredictRatingResponse` | ✅ Required | 60/min | Predict IMDB rating |
| 3 | `POST` | `/predict-boxoffice` | `PredictBoxOfficeRequest` | `PredictBoxOfficeResponse` | ✅ Required | 60/min | Predict box office with scenarios |
| 4 | `GET` | `/actor/{name}` | Path param `name: str` | `ActorInfoResponse` | ✅ Required | 60/min | Bankability Score + chemistry |
| 5 | `GET` | `/model-info` | — | `ModelInfoResponse` | ✅ Required | 60/min | Deployed model metadata |
| 6 | `GET` | `/docs` | — | HTML | ❌ Exempt | 60/min | Swagger UI documentation |

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

Both the FastAPI endpoints and the Streamlit dashboard call the **same** functions from `tamasha.predict`. There is no separate prediction implementation.

```python
# api/routers/predict.py
from tamasha.predict import predict_rating, predict_boxoffice

# api/routers/network.py
from tamasha.predict import get_actor_info

# api/routers/model_info.py
from tamasha.predict import get_model_info

# app/pages/_1_Predict_a_Release.py
from tamasha.predict import predict_rating, predict_boxoffice

# app/pages/_2_Star_Network_Explorer.py
from tamasha.predict import get_actor_info, get_bankability_scores, get_chemistry_pairs

# app/pages/_4_Model_Performance.py
from tamasha.predict import get_comparison_csv, get_model_info
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
- ~~`X-RateLimit-Reset`: (not yet available with slowapi)~~

When exceeded, returns `429 Too Many Requests`.

---

## Error Handling

| HTTP Status | Meaning | Example Scenario |
|:-----------:|---------|-----------------|
| `200` | Success | All endpoints |
| `401` | Unauthorized | Missing or invalid X-API-Key header |
| `404` | Not found | Actor not in Bankability dataset |
| `429` | Too Many Requests | Rate limit exceeded (60 req/min) |
| `503` | Service unavailable | Model not trained (missing .pkl) |
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

API tests in `tests/test_api.py` use FastAPI's `TestClient`:

```python
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_rating():
    response = client.post("/predict-rating", json={...})
    assert response.status_code in (200, 503)  # 503 if model not trained
```
