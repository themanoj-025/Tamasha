"""Prediction utility methods -- bankability, actor info, and helpers."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


    def _compute_cast_avg_bankability(self, cast_list: list[str]) -> dict[str, Any]:
        """Compute average Bankability Score for a cast.

        Returns dict with ``avg_score``, ``fallback_count``, and ``total_count``.
        """
        scores: list[float] = []
        fallback_count = 0
        for actor in cast_list:
            key = actor.strip().lower()
            score = self._bankability_map.get(key)
            if score is not None:
                scores.append(score)
            else:
                fallback_count += 1
                all_scores = list(self._bankability_map.values())
                scores.append(float(np.mean(all_scores)) if all_scores else 0.3)

        avg = float(np.mean(scores)) if scores else 0.0
        return {
            "avg_score": round(avg, 4),
            "fallback_count": fallback_count,
            "total_count": len(cast_list),
        }

    # â”€â”€ predict_rating â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def predict_rating(
        self,
        genres: list[str],
        cast: list[str],
        director: str = "Unknown Director",
        budget_inr: float = 0.0,
        runtime_minutes: int = 150,
        year: int = 2024,
    ) -> dict[str, Any]:
        """Predict a movie's IMDB rating.

        Returns
        -------
        dict
            ``{"predicted_rating": float | None, "model_name": str, "model_mae": float | None}``
        """
        if self._rating_model is None:
            return {"predicted_rating": None, "model_name": "No model trained", "model_mae": None}
        if not self._rating_feature_cols:
            return {
                "predicted_rating": None,
                "model_name": "No feature columns saved",
                "model_mae": None,
            }

        X_vec = self._build_prediction_vector(
            genres,
            cast,
            director,
            budget_inr,
            runtime_minutes,
            year,
            self._rating_feature_cols,
        )
        if X_vec.size == 0:
            return {"predicted_rating": None, "model_name": "Feature error", "model_mae": None}

        try:
            pred = float(self._rating_model.predict(X_vec)[0])
            pred = max(0.0, min(10.0, pred))
            model_name = self._model_names.get("rating", "GradientBoosting")
            model_mae = self._model_metrics.get("rating", {}).get("mae", 0)
            return {
                "predicted_rating": round(pred, 2),
                "model_name": model_name,
                "model_mae": model_mae,
            }
        except (ValueError, OSError) as exc:
            logger.error("Rating prediction failed: %s", exc)
            return {"predicted_rating": None, "model_name": "Error", "model_mae": None}

    # â”€â”€ predict_boxoffice â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def predict_boxoffice(
        self,
        genres: list[str],
        cast: list[str],
        director: str = "Unknown Director",
        budget_inr: float = 0.0,
        runtime_minutes: int = 150,
        year: int = 2024,
        release_window: str = "Normal",
    ) -> dict[str, Any]:
        """Predict a movie's box office collection (in â‚¹ Crore).

        Returns
        -------
        dict
            ``{"predicted_boxoffice_cr": float | None, "model_name": str,
                "model_mae": float | None, "bankability_info": dict,
                "scenarios": dict, "fallback_actors": bool}``
        """
        if self._boxoffice_model is None:
            return {
                "predicted_boxoffice_cr": None,
                "model_name": "No model trained",
                "model_mae": None,
            }
        if not self._boxoffice_feature_cols:
            return {
                "predicted_boxoffice_cr": None,
                "model_name": "No feature columns saved",
                "model_mae": None,
            }

        bank_info = self._compute_cast_avg_bankability(cast)

        X_vec = self._build_prediction_vector(
            genres,
            cast,
            director,
            budget_inr,
            runtime_minutes,
            year,
            self._boxoffice_feature_cols,
            bank_info["avg_score"],
        )
        if X_vec.size == 0:
            return {
                "predicted_boxoffice_cr": None,
                "model_name": "Feature error",
                "model_mae": None,
            }

        try:
            pred = float(self._boxoffice_model.predict(X_vec)[0])
            pred_cr = pred / 1e7
            pred_cr = max(0.0, pred_cr)

            model_name = self._model_names.get("boxoffice_bank", "XGBoost")
            model_mae_cr = self._model_metrics.get("boxoffice_bank", {}).get("mae", 0) / 1e7

            # Festival multipliers â€” documented as domain-expert priors
            festival_multipliers = {
                "Normal": 1.0,
                "Diwali": 1.25,
                "Eid": 1.18,
                "Christmas": 1.12,
                "Independence Day": 1.08,
                "Republic Day": 1.05,
                "New Year": 1.10,
            }
            base_pred = pred_cr

            scenarios = {}
            for scenario, mult in festival_multipliers.items():
                scenarios[scenario] = round(base_pred * mult, 1)

            return {
                "predicted_boxoffice_cr": round(base_pred, 1),
                "model_name": model_name,
                "model_mae": round(model_mae_cr, 1),
                "bankability_info": bank_info,
                "scenarios": scenarios,
                "fallback_actors": bank_info["fallback_count"] > 0,
            }
        except (ValueError, OSError) as exc:
            logger.error("Box office prediction failed: %s", exc)
            return {"predicted_boxoffice_cr": None, "model_name": "Error", "model_mae": None}

    # â”€â”€ get_actor_info â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def get_actor_info(self, name: str) -> dict[str, Any]:
        """Get Bankability Score and chemistry pairs for an actor."""
        key = name.strip().lower()

        score_row = None
        if not self._bankability_scores.empty:
            match = self._bankability_scores[
                self._bankability_scores["actor"].str.lower().str.strip() == key
            ]
            if not match.empty:
                score_row = match.iloc[0]

        chemistry: list[dict[str, Any]] = []
        if not self._chemistry_pairs.empty:
            mask = self._chemistry_pairs["actor_1"].str.lower().str.strip().isin(
                [key, name.lower()]
            ) | self._chemistry_pairs["actor_2"].str.lower().str.strip().isin([key, name.lower()])
            chem_matches = self._chemistry_pairs[mask]
            for _, row in chem_matches.iterrows():
                partner = (
                    row["actor_2"] if row["actor_1"].lower().strip() == key else row["actor_1"]
                )
                chemistry.append(
                    {
                        "actor": partner,
                        "joint_films": int(row["joint_films"]),
                        "chemistry_score": float(row["uplift"]),
                    }
                )

        return {
            "name": name.strip().title(),
            "bankability_score": float(score_row["bankability_score"])
            if score_row is not None
            else None,
            "film_count": int(score_row["film_count"]) if score_row is not None else 0,
            "type": str(score_row["type"]) if score_row is not None else "unknown",
            "top_chemistry_pairs": chemistry,
            "found": score_row is not None,
        }

    # â”€â”€ get_model_info â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def get_model_info(self) -> dict[str, Any]:
        """Get info about currently deployed models."""
        return {
            "rating_model": {
                "name": self._model_names.get("rating", "Not trained"),
                "algorithm": self._model_names.get("rating", "N/A"),
                "mae": self._model_metrics.get("rating", {}).get("mae"),
                "rmse": self._model_metrics.get("rating", {}).get("rmse"),
                "r2": self._model_metrics.get("rating", {}).get("r2"),
                "features_used": ["genre", "cast_size", "director", "runtime", "budget", "decade"],
            },
            "boxoffice_model": {
                "name": self._model_names.get("boxoffice_bank", "Not trained"),
                "algorithm": self._model_names.get("boxoffice_bank", "N/A"),
                "mae": self._model_metrics.get("boxoffice_bank", {}).get("mae"),
                "rmse": self._model_metrics.get("boxoffice_bank", {}).get("rmse"),
                "r2": self._model_metrics.get("boxoffice_bank", {}).get("r2"),
                "features_used": [
                    "genre",
                    "cast_size",
                    "director",
                    "runtime",
                    "budget",
                    "decade",
                    "avg_bankability_score",
                ],
            },
        }

    # â”€â”€ data accessors (for dashboard tables / charts) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def get_bankability_scores(self) -> pd.DataFrame:
        """Return the full Bankability scores DataFrame."""
        return self._bankability_scores

    def get_chemistry_pairs(self) -> pd.DataFrame:
        """Return the full chemistry pairs DataFrame."""
        return self._chemistry_pairs

    def get_comparison_csv(self, task: str) -> pd.DataFrame | None:
        """Return a model comparison DataFrame.

        Parameters
        ----------
        task : str
            One of ``"rating"``, ``"boxoffice_baseline"``, ``"boxoffice_bank"``.
        """
        return {
            "rating": self._rating_comparison,
            "boxoffice_baseline": self._boxoffice_baseline_comparison,
            "boxoffice_bank": self._boxoffice_bank_comparison,
        }.get(task)


#  Moduleâ€‘level singleton + thin wrappers  (Streamlit dashboard
#  manages lifecycle via ``st.cache_resource``)

_service: PredictionService | None = None


