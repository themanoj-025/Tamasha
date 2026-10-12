<p align="center">
  <img src="https://img.shields.io/badge/Tamasha-Bollywood%20Movie%20Intelligence-purple?style=for-the-badge" alt="Tamasha Logo" />
</p>

<h1 align="center">Tamasha</h1>

<p align="center">
  <strong>Predict ratings and box office success. Uncover what drives Bollywood outcomes — star pairings, release timing, plot tone, and poster aesthetics.</strong>
</p>

<p align="center">
  <a href="https://github.com/themanoj-025/Tamasha/actions"><img src="https://img.shields.io/github/actions/workflow/status/themanoj-025/Tamasha/ci.yml?style=flat-square&label=CI" alt="CI Status" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License MIT" /></a>
  <a href="https://github.com/themanoj-025/Tamasha/stargazers"><img src="https://img.shields.io/github/stars/themanoj-025/Tamasha?style=social" alt="GitHub Stars" /></a>
  <a href="https://github.com/themanoj-025/Tamasha/issues"><img src="https://img.shields.io/github/issues/themanoj-025/Tamasha?style=flat-square" alt="GitHub Issues" /></a>
  <a href="https://github.com/themanoj-025/Tamasha/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" alt="PRs Welcome" /></a>
</p>

---

## Overview

Most movie box-office prediction projects stop at training a single regressor on an uncurated tabular dataset. Tamasha goes further by investigating the empirical dynamics of Indian cinema:
- **Star Chemistry & Bankability**: Quantifies collaboration network lift and decay-weighted track records across 1,000+ actors and directors.
- **Festival Calendar Effects**: Measures whether holiday release windows (Diwali, Eid, Christmas) yield genuine uplift or selection bias.
- **Genre-Conditional Plot Tone**: Sentiment analysis revealing how plot tone correlates with box office differently across narrative genres.
- **Computer Vision Poster Analysis**: Empirically evaluates whether visual aesthetics (HSV histograms, face counts, brightness) hold predictive signal (reporting null findings transparently).

---

## Table of Contents

- [Key Empirical Results](#key-empirical-results)
- [Fuzzy-Join Data Engineering](#fuzzy-join-data-engineering)
- [Architecture](#architecture)
- [Bankability & Chemistry Network](#bankability--chemistry-network)
- [Release Timing & Sentiment Analysis](#release-timing--sentiment-analysis)
- [Computer Vision Null Findings](#computer-vision-null-findings)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Quick Start](#installation--quick-start)
- [Known Limitations & Responsible AI](#known-limitations--responsible-ai)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)
- [Support](#support)

---

## Key Empirical Results

All metrics are benchmarked on held-out test evaluation partitions:

| Metric | Measured Value | Implementation / Detail |
| :--- | :--- | :--- |
| ⭐ **Rating Prediction Model** | **MAE 0.9534 / 10** | Tuned GradientBoostingRegressor |
| 💰 **Box Office Model** | **MAE ₹73.6 Cr** | Tuned XGBoost Regressor with Bankability features |
| 🔥 **Bankability Feature Lift** | **11.6% MAE Improvement** | Outperforms baseline models lacking actor/director network features |
| 👥 **Bankability Coverage** | **1,010 Actors & Directors** | Exponential time-decay weighted scoring (3-year half-life) |
| 📡 **TMDb Metadata Coverage** | **93.2% Dates, 93.1% Plots** | Enriched via TMDb API integration |
| 🖼️ **Poster Computer Vision** | **49.2% (No Signal)** | 63 extracted visual features yielded no predictive edge over random chance |

---

## Fuzzy-Join Data Engineering

The system bridges three disconnected datasets that lack common primary keys:

| Dataset | Records | Role |
| :--- | :--- | :--- |
| **IMDb India Movies** | 15,509 | Primary ratings, genres, and cast metadata |
| **Bollywood Box Office** | 1,000 | Historic box office collections (originally missing release year) |
| **Year Bridge Dataset** | 7,419 | Intermediate lookup connecting titles to release years |

**Methodology**: A two-stage fuzzy matching pipeline using RapidFuzz first aligned titles to the Year Bridge (994/1,000 matched), then cross-referenced IMDb records on title and release year. This yielded 812 verified matches (81.2% link rate), with 93% of matches scoring `>= 95/100` string similarity confidence.

---

## Architecture

```text
RAW DATA LAYER
  IMDb India (15,509) ──┐
  Box Office (1,000) ───┼── Two-Stage Fuzzy Matching → 812 Verified Merged Records
  Year Bridge (7,419) ──┘
                             │
                             ▼
FEATURE ENGINEERING LAYER
  ├── Cast & Crew Collaboration Network (NetworkX Bankability Scores)
  ├── TMDb Release Timing (Holiday / Festival Window Proximity)
  └── VADER Plot Sentiment & Genre-Conditional Interaction Features
                             │
                             ▼
MODEL SELECTION & BENCHMARKING (5-Fold CV)
  Rating: GradientBoosting (MAE 0.95) │ Box Office: XGBoost (MAE ₹73.6 Cr)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   FastAPI REST Service              Streamlit Dashboard
   (api.main:app, Port 8000)         (app/streamlit_app.py)
```

---

## Bankability & Chemistry Network

To measure star power, Tamasha constructs a weighted historical performance ledger using an exponential decay function ($t_{1/2} = 3\text{ years}$):
- **Chemistry Uplift**: Identifies actor-director and co-star pairs whose combined performance reliably exceeds the sum of their individual expectations.
- **Top Verified Chemistry Pairs**:
  - 🥇 Nawazuddin Siddiqui & Salman Khan (Uplift: `+0.0818`)
  - 🥈 Jimmy Sheirgill & Kangana Ranaut (Uplift: `+0.0559`)
  - 🥉 Mastan Alibhai Burmawalla & Saif Ali Khan (Uplift: `+0.0532`)
- **SHAP Validation**: Bankability scores rank as the single most influential predictive feature for box-office outcomes behind production budget.

---

## Release Timing & Sentiment Analysis

- **Festival Proximity**: Calculates proximity ($\pm 7\text{ days}$) to major Indian cultural holidays (Diwali, Eid, Christmas, Republic Day, Independence Day).
- **Genre-Conditional Plot Tone**: VADER sentiment scores show stark divergence across genres:
  - 🎭 **Fantasy**: `+0.417` correlation (Darker plot tones correlate with higher revenues)
  - 📜 **Historical**: `+0.404` correlation (Serious narrative tones outperform light tones)
  - 🎶 **Musical**: `-0.325` correlation (Somber tones correlate with lower box office)
  - 📖 **Drama**: `-0.007` correlation (Zero correlation across 536 movies)

---

## Computer Vision Null Findings

Tamasha evaluated 63 visual features across 200 movie posters (HSV color distribution, edge density, brightness, face count). A Random Forest classifier achieved only **49.2% accuracy** (against a 51.1% majority baseline). This confirms that poster visual attributes contain negligible standalone signal for box office performance, a finding reported transparently rather than discarded.

---

## Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Language & Environment** | Python 3.11+ |
| **Data & Fuzzy Joins** | pandas, numpy, scipy, RapidFuzz |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM, CatBoost |
| **Network & NLP** | NetworkX, NLTK (VADER) |
| **Explainability** | SHAP |
| **Serving & UI** | Streamlit, Plotly, FastAPI, Uvicorn |
| **Testing & Quality** | Pytest, Hypothesis, Docker |

---

## Project Structure

```text
Tamasha/
├── api/                    # FastAPI routes, schemas, and serving logic
├── app/                    # Multi-page Streamlit dashboard
├── data/                   # Raw and processed datasets (gitignored)
├── models/                 # Serialized model weights and scalers
├── src/                    # Feature engineering, fuzzy joins, and modeling code
├── tests/                  # Pytest unit, fuzzy match, and API test suites
├── Dockerfile              # Multi-stage container build
├── docker-compose.yml      # Local container orchestration
├── pyproject.toml          # Build backend and test settings
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```

---

## Prerequisites

- **Python**: `>= 3.11`
- **Memory**: Minimum 2 GB RAM

---

## Installation & Quick Start

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/themanoj-025/Tamasha.git
cd Tamasha

# Create and activate virtual environment
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1

# Install requirements and package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 2. Run the Applications

**Run Streamlit Interactive Explorer**:
```bash
streamlit run app/streamlit_app.py
```
*Open http://localhost:8501.*

**Run FastAPI REST Service**:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
*Open Swagger API documentation at http://localhost:8000/docs.*

---

## Known Limitations & Responsible AI

- **Moderate $R^2$ Variance Explained**: Models explain 22%–35% of variance, reflecting unquantifiable consumer cultural sentiment. Suitable for directional intelligence, not definitive financial underwriting.
- **Hinglish Nuance**: VADER sentiment models standard English vocabulary; nuances in colloquial Bollywood Hinglish may be attenuated.
- **Historical Horizon**: Models extrapolate from pre-2024 theatrical releases; post-pandemic streaming distribution shifts may introduce drift.

---

## Testing

```bash
# Execute full pytest suite (141 tests)
pytest tests/ -v
```

---

## Roadmap

- [x] Rating and box office prediction models
- [x] Time-decayed bankability score and chemistry network
- [x] Release timing festival calendar analysis
- [x] Genre-conditional plot tone evaluation
- [x] Transparent reporting of computer vision null result
- [ ] Post-theatrical OTT streaming window revenue modeling <!-- TODO: Track via GitHub issue -->
- [ ] Multilingual Hinglish sentiment transformer pipeline <!-- TODO: Track via GitHub issue -->

---

## Contributing

Contributions are welcomed! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for pull request guidelines.

---

## Security

Please report vulnerabilities following the instructions in [SECURITY.md](SECURITY.md).

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Support

- 🐛 **Issue Tracker**: [GitHub Issues](https://github.com/themanoj-025/Tamasha/issues)
- 💡 **Discussions & Feedback**: [GitHub Pull Requests](https://github.com/themanoj-025/Tamasha/pulls)
