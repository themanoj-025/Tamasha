# Tamasha — Folder Structure

```
Tamasha/
├── api/                          # FastAPI service (interface layer)
│   ├── main.py                   # App factory: lifespan, middleware, DI, routers
│   ├── schemas.py                # Pydantic request/response DTOs
│   └── routers/                  # Thin route handlers
│       ├── predict.py            #   POST /predict endpoints
│       ├── network.py            #   star-network / bankability endpoints
│       └── model_info.py         #   model metadata endpoints
├── app/                          # Streamlit UI (interface layer)
│   ├── streamlit_app.py          # Multipage entry (launched by Docker CMD)
│   ├── pages/                    # Page modules (predict, network, trends, perf)
│   ├── components/               # Shared UI components (metric cards, graph)
│   └── assets/                   # theme.css
├── src/tamasha/                  # Core package (src-layout, installed package)
│   ├── __init__.py               # Version marker
│   ├── config.py                 # Settings / env-driven config (single source)
│   ├── cache.py                  # Prediction cache (SQLite + TTL)
│   ├── exceptions.py             # Domain exception hierarchy
│   ├── predict.py                # Public prediction service facade
│   ├── train_pipeline.py         # End-to-end training pipeline entry
│   ├── cv/                       # Poster-image classification
│   ├── data/                     # loaders · cleaning · joining · enrichment
│   ├── evaluation/               # metrics
│   ├── features/                 # movie_features · cast_crew_network
│   ├── models/                   # boxoffice_model · rating_model · selection
│   ├── network/                  # bankability_score · chemistry_pairs
│   ├── nlp/                      # plot_sentiment
│   └── timing/                   # festival_calendar · release_scenario
├── tests/                        # 18 pytest modules (unit + integration + API contract)
├── data/
│   ├── raw/                      # Source CSVs (IMDb, box-office)
│   └── processed/                # Parquet datasets, posters, TMDB cache, prediction cache
├── models/                       # Trained artifacts (*.pkl) + feature manifests
├── reports/                      # Analysis & model-comparison outputs
├── ops/                          # Prometheus config + Grafana provisioning/dashboards
├── docs/                         # Full documentation suite (see docs/README)
│   ├── migration/                # Migration records
│   └── ...                       # architecture, design, project, technical, reference
├── .github/workflows/ci.yml      # CI: lint + pytest w/ coverage gate
├── .streamlit/secrets.toml.example
├── docker-compose.yml            # App + observability (see docker-compose.observability.yml)
├── Dockerfile                    # Multi-stage build; CMD = streamlit app
├── Makefile                      # train / test / lint / clean targets
├── render.yaml                   # Render deploy: gunicorn api.main:app
├── pyproject.toml                # Packaging, pytest & coverage config
├── requirements.txt              # Runtime deps
└── packages.txt                  # System packages (Docker)
```

## Layout rules

- **Interface layer** (`api/`, `app/`) contains no business logic — it delegates
  to `tamasha.*`.
- **Core package** is organized by domain (`data`, `models`, `network`, `nlp`,
  `timing`, `cv`) rather than by file type — a feature's code lives in one place.
- **Artifacts** (trained models, processed data, reports) have dedicated
  top-level directories; nothing is written into the source tree at runtime.
- **Secrets never tracked** — `.env` and `.streamlit/secrets.toml` are
  gitignored; only `.example` variants are committed.
