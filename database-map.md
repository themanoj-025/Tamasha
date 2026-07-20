# Tamasha — Data Schema & Entity Relationships

## Overview

Tamasha does **not** use a traditional database. Data flows are:

```
Raw CSVs (Kaggle) → Cleaned Parquet files → In-memory DataFrames → Model artifacts
                                                                   ↓
                                                         CSV reports (for dashboard)
```

All persistence is file-based:
- **Parquet**: Cleaned intermediate datasets (`data/processed/*.parquet`)
- **Pickle/Joblib**: Trained models (`models/*.pkl`)
- **JSON**: Feature column names (`models/*_features.json`), TMDb cache (`data/processed/tmdb_cache.json`)
- **CSV**: Reports and analysis outputs (`reports/*.csv`)
- **PNG**: Charts and visualizations (`reports/figures/*.png`)
- **MD**: Analysis reports (`reports/*.md`)

---

## 1. Input Datasets

### IMDb India Movies (`IMDb Movies India.csv`)

| Original Column | Canonical Name | Type | Description |
|----------------|---------------|------|-------------|
| `Name` | `title` | `str` | Movie title |
| `Year` | `year` → parsed from `year_raw` | `int` | Release year (parsed from `(YYYY)` format) |
| `Duration` | `duration_minutes` → parsed from `duration_raw` | `int` | Runtime in minutes (parsed from `"109 min"`) |
| `Genre` | `genre` | `str` | Comma-separated genres |
| `Rating` | `rating` | `float` | IMDB rating (0-10) |
| `Votes` | `votes` | `int` | Number of votes |
| `Director` | `director` | `str` | Director name |
| `Actor 1` | → merged into `cast` | `str` | Lead actor |
| `Actor 2` | → merged into `cast` | `str` | Second actor |
| `Actor 3` | → merged into `cast` | `str` | Third actor |

**Loading**: `load_imdb_india()` in `data/loaders.py`
**Encoding**: `latin1`
**Size**: ~11,500+ movies

### Bollywood Box Office (`Top 1000 Bollywood Movies and their boxoffice.csv`)

| Original Column | Canonical Name | Type | Description |
|----------------|---------------|------|-------------|
| `Movie` | `title` | `str` | Movie title |
| `Worldwide` | `worldwide_collection_inr` | `int` | Worldwide box office in ₹ (all numerical) |
| `India Net` | `india_net_collection_inr` | `int` | India net collection in ₹ |
| `India Gross` | `india_gross_collection_inr` | `int` | India gross collection in ₹ |
| `Overseas` | `overseas_collection_inr` | `int` | Overseas collection in ₹ |
| `Budget` | `budget_inr` | `int` | Budget in ₹ (all numerical) |
| `Verdict` | `verdict` | `str` | Box office verdict (Hit, Flop, Super Hit, etc.) |

**Loading**: `load_bollywood_boxoffice()` in `data/loaders.py`
**Size**: ~1,000+ movies

### Year-Bridge Dataset (`bollywood_movies.csv`)

A third Kaggle dataset used as a bridge to add year information to box office movies before joining to IMDb.

| Canonical Column | Type | Description |
|-----------------|------|-------------|
| `title` | `str` | Movie title |
| `year` | `int` | Release year |
| Other fields | Various | Not used directly |

---

## 2. Joined Dataset

Created by the two-step fuzzy join in `train_pipeline.py`:

```
Step 1: Box Office → year-bridge (on title)
Step 2: Box Office (with years) → IMDb (on title + year tolerance=2)
```

### Joined Columns (after rename in pipeline)

| Column | Source | Type | Description |
|--------|--------|------|-------------|
| `title` | From IMDb side | `str` | Movie title |
| `genre` | From IMDb | `str` | Comma-separated genres |
| `rating` | From IMDb | `float` | IMDB rating |
| `director` | From IMDb | `str` | Director name |
| `year` | From year-bridge or IMDb | `int` | Release year |
| `cast` | From IMDb (merged from Actor 1/2/3) | `str` | Comma-separated cast |
| `duration_minutes` | From IMDb | `int` | Runtime |
| `worldwide_collection_inr` | From Box Office | `int` | Worldwide collection (₹) |
| `india_net_collection_inr` | From Box Office | `int` | India net (₹) |
| `india_gross_collection_inr` | From Box Office | `int` | India gross (₹) |
| `overseas_collection_inr` | From Box Office | `int` | Overseas (₹) |
| `budget_inr` | From Box Office | `int` | Budget (₹) |
| `verdict` | From Box Office | `str` | Hit/flop verdict |
| `_match_score` | From fuzzy join | `float` | Fuzzy match quality (0-100) |
| `plot_summary` | TMDb enrichment | `str` | Plot summary text |
| `release_date` | TMDb enrichment | `str` | Release date (YYYY-MM-DD) |
| `is_festival_release` | Festival analysis | `bool` | Near festival window? |
| `has_clash` | Clash analysis | `bool` | Clash with another release? |

**Match rate**: ~81.2% (varies with datasets)
**Saved as**: `data/processed/boxoffice_clean.parquet`

---

## 3. Feature Matrix (Rating)

Created by `build_feature_matrix(df_rating, target_column_rating="rating")`

| Feature Group | # Features | Columns |
|---------------|:----------:|---------|
| Genre dummies | ~20-25 | `genre_Action`, `genre_Comedy`, `genre_Drama`, etc. |
| Cast size | 1 | `cast_size` |
| Director (label-encoded) | 1 | `director_encoded` |
| Runtime | 1 | `runtime_minutes` |
| Decade dummies | ~8 | `decade_1990`, `decade_2000`, `decade_2010`, `decade_2020` |
| Budget | 1 | `budget_inr` |
| **Total** | **~36** | |

---

## 4. Feature Matrix (Box Office)

Created by `build_feature_matrix(df_box_model, target_column_boxoffice=...)`

### Baseline (no Bankability)

| Feature Group | # Features |
|---------------|:----------:|
| Genre dummies | ~20-25 |
| Cast size | 1 |
| Director (label-encoded) | 1 |
| Runtime | 1 |
| Decade dummies | ~8 |
| Budget | 1 |
| **Total** | **~31** |

### With Bankability Score

| Feature Group | # Features |
|---------------|:----------:|
| Same as baseline | ~31 |
| `avg_bankability_score` | +1 |
| **Total** | **~32** |

---

## 5. Bankability Scores

Created by `compute_bankability_scores()` in `network/bankability_score.py`

| Column | Type | Description |
|--------|------|-------------|
| `actor` | `str` | Actor/director name |
| `type` | `str` | `"actor"` or `"director"` |
| `bankability_score` | `float` | Time-decay-weighted performance score (0-1) |
| `film_count` | `int` | Number of films in dataset |

**Size**: 1,010 individuals
**Saved as**: `reports/bankability_scores.csv`

---

## 6. Chemistry Pairs

Created by `detect_chemistry_pairs()` in `network/chemistry_pairs.py`

| Column | Type | Description |
|--------|------|-------------|
| `actor_1` | `str` | First actor |
| `actor_2` | `str` | Second actor |
| `joint_films` | `int` | Number of joint appearances |
| `joint_avg_perf` | `float` | Average performance of joint films |
| `solo_avg_1` | `float` | Actor 1's solo average |
| `solo_avg_2` | `float` | Actor 2's solo average |
| `uplift` | `float` | Joint avg - max(solo_1, solo_2) |

**Size**: Top 10 pairs (configurable)
**Saved as**: `reports/chemistry_pairs.csv`

---

## 7. Model Comparison Results

All have the same schema:

| Column | Type | Description |
|--------|------|-------------|
| `model` | `str` | Model name |
| `MAE` | `float` | Mean Absolute Error |
| `MAE_std` | `float` | MAE standard deviation across folds |
| `RMSE` | `float` | Root Mean Squared Error |
| `RMSE_std` | `float` | RMSE standard deviation |
| `R2` | `float` | R² score |
| `R2_std` | `float` | R² standard deviation |
| `training_time_s` | `float` | Training time in seconds |

The comparison CSVs now include a boolean `tuned` column indicating whether the model
was optimized via RandomizedSearchCV before the CV comparison.

### Files:

| File | Contents |
|------|----------|
| `reports/model_comparison_rating.csv` | 9 models × 5-fold CV on rating task (includes `tuned` column) |
| `reports/model_comparison_boxoffice_baseline.csv` | 9 models × 5-fold CV on box office (no Bankability) |
| `reports/model_comparison_boxoffice_with_bankability.csv` | 9 models × 5-fold CV on box office (with Bankability) |

---

## 8. Sentiment Analysis

### Per-Movie Sentiment (`score_plot_sentiment()`)

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `compound` | `float` | -1 to +1 | Overall sentiment |
| `pos` | `float` | 0 to 1 | Positive proportion |
| `neu` | `float` | 0 to 1 | Neutral proportion |
| `neg` | `float` | 0 to 1 | Negative proportion |

### Genre-Tone Correlation (`genre_conditional_correlation()`)

| Column | Type | Description |
|--------|------|-------------|
| `genre` | `str` | Genre name |
| `n_movies` | `int` | Number of movies in this genre |
| `correlation` | `float` | Correlation of compound sentiment with target |
| `mean_compound` | `float` | Average compound score for the genre |

**Saved as**: `reports/genre_tone_correlation.csv`, `reports/genre_tone_correlation_rating.csv`

---

## 9. Entity Relationships

```
Movie
├── has → Genre (many-to-many via comma-separated string)
├── has → Cast (many-to-many via comma-separated string)
├── has → Director (many-to-one)
├── has → Rating (one-to-one, float 0-10)
├── has → Box Office Collection (one-to-one, ₹ int)
├── has → Budget (one-to-one, ₹ int)
├── has → Verdict (one-to-one, categorical)
├── has → Plot Summary (one-to-one, text via TMDb)
└── has → Release Date (one-to-one, date via TMDb)

Actor
├── acts_in → Movie (many-to-many via cast column)
├── has → Bankability Score (one-to-one, computed)
└── collaborates_with → Actor (many-to-many, weighted by joint film performance)

Director
├── directs → Movie (one-to-many)
└── has → Bankability Score (one-to-one, computed as director type)

Genre
└── categorizes → Movie (many-to-many)

Chemistry Pair (Actor_A - Actor_B)
├── has → Joint Films (int, count)
├── has → Joint Performance (float, average)
├── has → Uplift (float, improvement over solo)
└── has → Individual Solo Performances (float, per actor)
```

---

## 10. File Persistence Map

| Data | Format | Path | Cache? | Generated By |
|------|--------|------|:------:|-------------|
| Raw IMDb India | CSV | `data/raw/IMDb Movies India.csv` | No | Kaggle download |
| Raw Box Office | CSV | `data/raw/Top 1000 Bollywood Movies...csv` | No | Kaggle download |
| Year-bridge | CSV | `data/raw/bollywood_movies.csv` | No | Kaggle download |
| Cleaned Box Office | Parquet | `data/processed/boxoffice_clean.parquet` | Yes | `train_pipeline.py` Step 3 |
| Cleaned IMDb | Parquet | `data/processed/imdb_clean.parquet` | Yes | `train_pipeline.py` Step 3 |
| TMDb Cache | JSON | `data/processed/tmdb_cache.json` | Yes | `data/enrichment.py` |
| Poster Images | JPG | `data/processed/posters/` | Yes | `data/enrichment.py` → download |
| Rating Model | Pickle | `models/best_rating_model.pkl` | Yes | `train_pipeline.py` Step 4 |
| Box Office Model | Pickle | `models/best_boxoffice_model.pkl` | Yes | `train_pipeline.py` Step 7b |
| Feature Columns | JSON | `models/*_features.json` | Yes | `train_pipeline.py` Steps 4/7 |
| Comparison CSVs | CSV | `reports/model_comparison_*.csv` | Yes | `model_selection.py` |
| Bankability Scores | CSV | `reports/bankability_scores.csv` | Yes | `train_pipeline.py` Step 6 |
| Chemistry Pairs | CSV | `reports/chemistry_pairs.csv` | Yes | `train_pipeline.py` Step 6 |
| Join Report | Markdown | `reports/join_quality_report.md` | Yes | `joining.py` |
| Timing Report | Markdown | `reports/release_timing_analysis.md` | Yes | `train_pipeline.py` Step 8 |
| Figures | PNG | `reports/figures/*.png` | Yes | `train_pipeline.py` + `metrics.py` |
