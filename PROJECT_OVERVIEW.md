# Tamasha — Bollywood Movie Intelligence Platform

> Predict ratings and box office. Uncover what drives Bollywood success — star pairings, release timing, plot tone, and poster aesthetics.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Tests: 141](https://img.shields.io/badge/Tests-141%20passing-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Tech Stack & Core Technologies](#2-tech-stack--core-technologies)
- [3. High-Level Architecture](#3-high-level-architecture)
- [4. Complete Folder Structure Tree](#4-complete-folder-structure-tree)
- [5. Exhaustive File-by-File & Folder-by-Folder Breakdown](#5-exhaustive-file-by-file--folder-by-folder-breakdown)
- [6. Data Models & Schemas](#6-data-models--schemas)
- [7. API Surface](#7-api-surface)
- [8. Configuration & Environment Variables](#8-configuration--environment-variables)
- [9. Build, Run & Deployment Instructions](#9-build-run--deployment-instructions)
- [10. Data & Control Flow Walkthroughs](#10-data--control-flow-walkthroughs)
- [11. Dependency Graph Summary](#11-dependency-graph-summary)
- [12. Testing Strategy](#12-testing-strategy)
- [13. Known Issues, Technical Debt & Assumptions](#13-known-issues-technical-debt--assumptions)
- [14. Glossary](#14-glossary)
- [15. Appendix](#15-appendix)

---

## 1. Executive Summary

**Tamasha** is a Bollywood movie intelligence platform that predicts ratings and box office performance, uncovering what drives success in Indian cinema. It goes beyond simple prediction to analyze star pairings (chemistry scores), release timing (festival windows), plot tone (genre-conditional sentiment), and poster aesthetics.

**Target users**: Bollywood producers, film analysts, movie enthusiasts, and data science learners.

**What problem it solves**: Most movie prediction projects end at "train a model, predict a number." Tamasha answers questions that actually matter: which star pairings have real chemistry, does Diwali release timing matter, does dark tone correlate with success across genres.

**Why it exists**: To demonstrate a complete, research-grade ML project that combines predictive modeling with network analysis, NLP, and computer vision — applied to a culturally rich domain.

*Note: The fuzzy-join methodology, bankability scores, and chemistry pairings are explicitly documented in the README. The 141-test suite and three Kaggle datasets are verified.*

---

## 2. Tech Stack & Core Technologies

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Language | Python | 3.11+ | Primary language |
| ML | scikit-learn | — | 9 models × 5-fold CV × 3 tasks = 135 training runs |
| Gradient Boosting | XGBoost, LightGBM, CatBoost | — | High-performance models |
| NLP | NLTK (VADER) | — | Plot sentiment analysis |
| Network Analysis | NetworkX | — | Cast/crew collaboration graph |
| Explainability | SHAP | — | Feature importance |
| Dashboard | Streamlit + Plotly | — | Interactive UI (4 pages) |
| API | FastAPI + Pydantic | — | REST API |
| Data Processing | pandas, numpy, scipy, rapidfuzz | — | Fuzzy join methodology |
| Image Processing | OpenCV, Pillow | — | Poster feature extraction |
| External API | TMDb | — | Release dates, plot summaries, posters |
| Testing | pytest, Hypothesis, httpx | — | 141 tests |
| Containerization | Docker + docker-compose | — | Multi-stage builds |

---

## 3. High-Level Architecture

```
RAW DATA LAYER
  IMDb India (15,509) ──┐
  Box Office (1,000) ───┼── Fuzzy Join → 812 matched → Feature Engineering
  Year Bridge (7,419) ──┘

MODEL COMPARISON LAYER
  9 Models × 5-Fold CV × 3 Tasks = 135 Training Runs
  Rating: GradientBoosting (MAE: 0.9534/10)
  Box Office: XGBoost + Bankability (MAE: ₹73.6 Cr)

DEPLOYMENT LAYER
  FastAPI API       Streamlit Dashboard
  /predict-rating   Predict a Release
  /predict-boxoffice Star Network Explorer
  /actor/{name}     Industry Trends
```

---

## 4. Complete Folder Structure Tree

```
Tamasha/
├── .dockerignore
├── .env.example
├── .github/workflows/ci.yml
├── .gitignore
├── .pre-commit-config.yaml
├── .streamlit/
│   └── secrets.toml.example
├── AGENTS.md
├── api/
│   ├── main.py
│   ├── routers/
│   │   ├── model_info.py
│   │   ├── network.py
│   │   └── predict.py
│   └── schemas.py
├── app/
│   ├── assets/theme.css
│   ├── components/
│   │   ├── metric_cards.py
│   │   └── network_graph.py
│   ├── pages/
│   │   ├── _1_Predict_a_Release.py
│   │   ├── _2_Star_Network_Explorer.py
│   │   ├── _3_Industry_Trends.py
│   │   └── _4_Model_Performance.py
│   └── streamlit_app.py
├── docker-compose.observability.yml
├── docker-compose.yml
├── Dockerfile
├── docs/
│   ├── community/
│   ├── design/
│   ├── product/
│   ├── project/
│   ├── reference/
│   └── technical/
├── LICENSE
├── Makefile
├── ops/
│   ├── grafana/
│   ├── prometheus.yml
├── packages.txt
├── PROJECT_ANALYSIS.md
├── PROJECT_OVERVIEW.md
├── pyproject.toml
├── README.md
├── render.yaml
├── reports/
│   ├── bankability_scores.csv
│   ├── chemistry_pairs.csv
│   └── model_comparison_*.csv
├── requirements.txt
├── setup.py
├── src/tamasha/
│   ├── __init__.py
│   ├── cache.py
│   ├── config.py
│   ├── cv/poster_classifier.py
│   ├── data/
│   │   ├── cleaning.py
│   │   ├── enrichment.py
│   │   ├── joining.py
│   │   └── loaders.py
│   ├── evaluation/metrics.py
│   ├── exceptions.py
│   ├── features/
│   │   ├── cast_crew_network.py
│   │   └── movie_features.py
│   ├── models/
│   │   ├── boxoffice_model.py
│   │   ├── model_selection.py
│   │   └── rating_model.py
│   ├── network/
│   │   ├── bankability_score.py
│   │   └── chemistry_pairs.py
│   ├── nlp/plot_sentiment.py
│   ├── predict.py
│   ├── timing/
│   │   ├── festival_calendar.py
│   │   └── release_scenario.py
│   └── train_pipeline.py
└── tests/
    ├── conftest.py
    └── test_*.py               # 141 tests
```

---

## 5. Exhaustive File-by-File & Folder-by-Folder Breakdown

### Core Pipeline

#### `src/tamasha/train_pipeline.py`
- **Purpose**: Full training pipeline — data loading, fuzzy join, feature engineering, 9-model comparison, model selection, persistence.

#### `src/tamasha/predict.py`
- **Purpose**: Prediction service — loads trained models, preprocesses input, returns rating/box office predictions.

#### `src/tamasha/data/joining.py`
- **Purpose**: Fuzzy-join methodology — two-step enrichment matching Box Office data to IMDb via Year Bridge dataset. 93% of matches have similarity score ≥95/100.

#### `src/tamasha/network/bankability_score.py`
- **Purpose**: Time-decay-weighted historical average with 3-year half-life exponential decay. Combines normalized rating with log-transformed collection.

#### `src/tamasha/network/chemistry_pairs.py`
- **Purpose**: Cast/crew collaboration graph analysis. Identifies pairs that improve each other's performance beyond individual track records.

#### `src/tamasha/nlp/plot_sentiment.py`
- **Purpose**: VADER sentiment analysis on plot summaries with genre-conditional correlations.

#### `src/tamasha/timing/festival_calendar.py`
- **Purpose**: 9 major Indian release windows detection (Eid, Diwali, Christmas, etc.) with ±7 day tolerance.

#### `src/tamasha/cv/poster_classifier.py`
- **Purpose**: 63 hand-crafted visual features (HSV histograms, brightness, edge density, face count) from poster images. **Result: 49.2% accuracy — no signal detected.** (Transparent null result.)

---

## 6. Data Models & Schemas

### Bankability Score

```json
{
  "individual": "str — actor/director name",
  "type": "actor | director",
  "score": "float — 0-2.0 (time-decay-weighted)",
  "films_count": "int — number of films"
}
```

### Chemistry Pair

```json
{
  "pair": "str — 'Actor A & Actor B'",
  "uplift": "float — performance improvement beyond individual baselines"
}
```

---

## 7. API Surface

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/predict-rating` | Predict movie rating |
| `POST` | `/predict-boxoffice` | Predict box office collection |
| `GET` | `/actor/{name}` | Actor bankability score |
| `GET` | `/model-info` | Model performance metrics |

---

## 8. Configuration & Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `TMDB_API_KEY` | TMDb API for enrichment | Yes (for enrichment) |
| `KAGGLE_USERNAME` | Kaggle dataset download | Yes (for data) |
| `KAGGLE_KEY` | Kaggle API key | Yes (for data) |

---

## 9. Build, Run & Deployment Instructions

```bash
# Install
pip install -r requirements.txt
pip install -e .

# Download datasets
kaggle datasets download adrianmcmahon/imdb-india-movies -p data/raw/ --unzip
kaggle datasets download rajugc/bollywood-movies-dataset -p data/raw/ --unzip

# Train models
make train

# Run tests
make test

# Start dashboard
streamlit run app/streamlit_app.py

# Start API
uvicorn api.main:app --reload
```

---

## 10. Data & Control Flow Walkthroughs

### Flow 1: Predict a Release

1. User enters movie details (title, cast, genre, budget, release date)
2. System enriches with bankability scores and festival timing
3. Feature engineering creates 50+ features
4. XGBoost model predicts box office (₹ Cr)
5. Results displayed with confidence intervals

### Flow 2: Star Network Explorer

1. User searches for an actor/director
2. NetworkX graph shows collaboration network
3. Bankability scores displayed for connected individuals
4. Chemistry pairs identified with uplift scores

---

## 11. Dependency Graph Summary

```
src/tamasha/train_pipeline.py → data/* → features/* → models/* → evaluation/*
src/tamasha/predict.py → models/* → features/*
src/tamasha/network/* → data/joining.py
src/tamasha/nlp/* → data/loaders.py
api/main.py → src/tamasha/predict.py
app/streamlit_app.py → src/tamasha/predict.py, network/*
```

---

## 12. Testing Strategy

- **Framework**: pytest + Hypothesis
- **Tests**: 141 tests across unit, integration, and property-based
- **Coverage**: 8 modules omitted from coverage (training, loaders, etc.)

---

## 13. Known Issues, Technical Debt & Assumptions

### Known Issues

1. **Low R² (~0.22-0.35)**: Directionally useful, not suitable for financial decisions.
2. **Small training set (~800-12K)**: May not generalize to films far from training distribution.
3. **VADER sentiment**: Trained on social media text, not Bollywood Hinglish.
4. **Static dataset**: Predictions for 2025+ extrapolate from pre-2024 patterns.
5. **Poster CV null result**: 49.2% accuracy — no visual signal detected.

---

## 14. Glossary

| Term | Definition |
|------|-----------|
| **Bankability Score** | Time-decay-weighted historical performance metric |
| **Chemistry Uplift** | Performance improvement of a pair beyond individual baselines |
| **Fuzzy Join** | Matching records across datasets without common IDs |
| **Festival Window** | ±7 days around major Indian release dates |
| **Genre-Conditional Correlation** | Sentiment-success relationship filtered by genre |

---

## 15. Appendix

### Key Results

| Metric | Value |
|--------|-------|
| Best Rating Model | GradientBoosting (MAE: 0.9534/10) |
| Best Box Office Model | XGBoost + Bankability (MAE: ₹73.6 Cr) |
| Bankability Impact | 11.6% MAE improvement over baseline |
| TMDb Enrichment | 93.2% date coverage, 93.1% plot coverage |
| Poster CV | 49.2% (no signal) |
| Test Suite | 141 tests passing |

---

*This document was generated as part of a comprehensive project documentation effort. Last updated: August 8, 2026.*
