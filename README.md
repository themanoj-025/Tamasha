# 🎬 Tamasha — Bollywood Movie Intelligence Platform

<p><em>Predict ratings and box office. Uncover what drives Bollywood success — star pairings, release timing, plot tone, and poster aesthetics.</em></p>

[![Python](https://img.shields.io/badge/Python_3.11+-3776AB?logo=python&logoColor=white)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)]()
[![Tests](https://img.shields.io/badge/Tests-141_passing-brightgreen)]()

---

## The Problem

Most movie rating and box-office prediction projects end at training a single model on a flat dataset — predict the number, done. Generic, forgettable, and indistinguishable from a Kaggle notebook.

**Tamasha goes further.** This project is built to answer questions that actually matter in Bollywood:

- 🎭 **Which star pairings have real chemistry?** Not just who works together most, but who *improves each other's performance* beyond their individual track records.
- 📅 **Does release timing matter?** Do Diwali and Eid releases actually outperform, or is that selection bias?
- 🎨 **Does plot tone correlate differently with success across genres?** A dark thriller may be great; a dark comedy may flop.
- 🖼 **Can a poster's visual features predict box office?** Color palettes, face count, text density — is there signal in the art?

The result is a platform with **three predictive models** (rating, baseline box office, bankability-enhanced box office), a **network analysis engine**, and an **interactive dashboard**.

---

## Key Results

| Metric | Value |
|--------|-------|
| ⭐ **Best Rating Model** | **GradientBoosting (tuned)** — MAE **0.9534** / 10 |
| 💰 **Best Box Office Model** | **XGBoost (tuned)** (+ Bankability) — MAE **₹73.6 Cr** |
| 🔥 **Bankability Impact** | **11.6% MAE improvement** over baseline |
| 👥 **Bankability Scores** | 1,010 actors & directors scored |
| 📡 **TMDb Enrichment** | **93.2%** date coverage, **93.1%** plot coverage |
| 🖼 **Poster CV** | No signal found — 49.2% (vs 51.1% baseline) |
| ✅ **Test Suite** | **141 tests** passing |

---

## Fuzzy-Join Methodology

The project uses **three separate Kaggle datasets** with no common ID:

| Dataset | Source | Rows |
|---------|--------|:----:|
| 🎬 IMDb India Movies | [adrianmcmahon/imdb-india-movies](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies) | 15,509 |
| 💰 Bollywood Box Office | [rajugc/bollywood-movies-dataset](https://www.kaggle.com/datasets/rajugc/bollywood-movies-dataset) | 1,000 |
| 🔗 Year Bridge | [vidhikishorwaghela/bollywood-movies-dataset](https://www.kaggle.com/datasets/vidhikishorwaghela/bollywood-movies-dataset) | 7,419 |

**Challenge:** The Box Office dataset has **no year column**. A two-step fuzzy enrichment strategy was used — first matching titles to the Year Bridge dataset (994/1,000 matched), then matching enriched records to IMDb India on title + year (812 high-confidence matches, 81.2%). **93% of matches have a similarity score >= 95/100.**

---

## Architecture

```
RAW DATA LAYER
  IMDb India (15,509) ──┐
  Box Office (1,000) ───┼── Fuzzy Join → 812 matched → Feature Engineering
  Year Bridge (7,419) ──┘

MODEL COMPARISON LAYER
  9 Models x 5-Fold CV x 3 Tasks = 135 Training Runs
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │LinearReg │ │  Ridge   │ │  Lasso   │ │DecisionT │ │RandomFor │
  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │GradBoost │ │ XGBoost  │ │ LightGBM │ │ CatBoost │
  └──────────┘ └──────────┘ └──────────┘ └──────────┘
          |                          |
  Rating: GradientBoosting    Box Office: XGBoost (+Bankability)
  MAE: 0.9534/10              MAE: 73.6 Cr

DEPLOYMENT LAYER
  FastAPI API       Streamlit Dashboard
  /predict-rating   Predict a Release
  /predict-boxoffice Star Network Explorer
  /actor/{name}     Industry Trends
  /model-info       Model Performance (with SHAP)
```

---

## The Bankability Score & Chemistry Pairing Network

This is the signature feature — a network analysis of the Bollywood cast/crew collaboration graph.

For every actor and director, we compute a **time-decay-weighted historical average** of their film performance using a 3-year half-life exponential decay. Scores combine normalized rating with log-transformed collection.

### Top Bankability Scores

| Rank | Individual | Type | Score |
|:----:|-----------|:----:|:-----:|
| 🥇 | Deepa Mehta | Director | 2.0000 |
| 🥇 | Lisa Ray | Actor | 2.0000 |
| 🥉 | Arun Bakshi | Actor | 1.9559 |

### Top Chemistry Pairs

| Rank | Pair | Uplift |
|:----:|------|:------:|
| 🥇 | Nawazuddin Siddiqui & Salman Khan | 0.0818 |
| 🥈 | Jimmy Sheirgill & Kangana Ranaut | 0.0559 |
| 🥉 | Mastan Alibhai Burmawalla & Saif Ali Khan | 0.0532 |

The Bankability Score ranks as the **second most important feature** in the box office model — right after budget — validated by SHAP analysis.

---

## Release Timing & Festival Analysis

Using the **TMDb API**, 812 box office movies were enriched with release dates (93.5% coverage). The module defines **9 major Indian release windows** (Eid, Diwali, Christmas, etc.) and detects whether a release falls within +/-7 days of a festival.

## Plot Tone & Genre-Conditional Analysis

VADER sentiment analysis on plot summaries produces genre-conditional correlations. Key findings:

| Genre | Correlation | Signal |
|:----:|:-----------:|:------:|
| 🎭 **Fantasy** | **+0.417** | Dark tone -> higher box office |
| 📜 **History** | **+0.404** | Serious tone -> higher box office |
| 🎶 **Music** | **-0.325** | Serious tone -> higher box office |
| 📖 **Drama** | **-0.007** | **No signal** (largest category, 536 movies) |

Drama (the largest category) shows **zero correlation** — confirming that genre-conditional analysis discovered real patterns a naive approach would miss.

---

## Poster CV Module

Using 200 poster images from the TMDb API, we extracted 63 hand-crafted visual features (HSV histograms, brightness, edge density, face count) and trained a Random Forest classifier. **Result: 49.2% accuracy** — no signal detected. This null result is reported transparently.

---

## How to Run

```bash
# Install
git clone https://github.com/themanoj-025/tamasha.git
cd tamasha
python -m venv .venv
pip install -r requirements.txt
pip install -e .

# Download datasets
kaggle datasets download adrianmcmahon/imdb-india-movies -p data/raw/ --unzip
kaggle datasets download rajugc/bollywood-movies-dataset -p data/raw/ --unzip
kaggle datasets download vidhikishorwaghela/bollywood-movies-dataset -p data/raw/ --unzip

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

## Tech Stack

| Category | Technologies |
|----------|:------------|
| **Language** | Python 3.11+ |
| **Data Processing** | pandas, numpy, scipy, rapidfuzz |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM, CatBoost |
| **NLP** | NLTK (VADER) |
| **Network Analysis** | NetworkX |
| **Explainability** | SHAP |
| **Dashboard** | Streamlit, Plotly |
| **API** | FastAPI, Pydantic, uvicorn |
| **Testing** | pytest, pytest-cov, Hypothesis, httpx (141 tests) |
| **DevOps** | Docker, docker-compose |
| **Image Processing** | OpenCV, Pillow |
| **External APIs** | TMDb (themoviedb.org) |

---

## Responsible AI & Known Limitations

| Limitation | Impact |
|------------|--------|
| **Low R²** (~0.22-0.35) | Directionally useful, not suitable for financial decisions |
| **Small training set** (~800-12K samples) | May not generalize to films far from training distribution |
| **VADER sentiment** | Trained on social media text, not Bollywood Hinglish plot summaries |
| **Static dataset** | Predictions for 2025+ films extrapolate from pre-2024 patterns |

---

## License

MIT License — see [LICENSE](LICENSE).
