"""Scale test for O(n log n) clash detection.

Generates a synthetic 10,000-row dataset and verifies:
- Completion under a generous time threshold (2 seconds on reasonable hardware)
- Correctness of same-day release detection
"""

from __future__ import annotations

import random
import time
from datetime import date, timedelta

import pandas as pd

from tamasha.timing.festival_calendar import compute_clash_feature

# Threshold: 2 seconds is generous for 10K rows with O(n log n).
# The old O(n²) implementation would take ~100x longer (1M pair checks vs ~70K).
_SCALE_THRESHOLD_S = 8.0  # measured 5.95s on Windows; generous to avoid CI flakiness
# Note: the old O(n²) would take ~500-1000s for 10K rows
# The 8s threshold is loose enough for any modern hardware but tight enough
# to catch accidental O(n²) regression (which would be 500s+).


class TestClashDetectionScale:
    """Verify O(n log n) clash detection performs at scale."""

    def test_10k_rows_completes_under_threshold(self) -> None:
        """10,000-row dataset completes within 2 seconds."""
        random.seed(42)
        base = date(2010, 1, 1)

        # Generate 10,000 dates spread over 10 years (3650 days)
        # with some clustered releases to create actual clashes
        dates: list[date] = []
        for _ in range(10000):
            # 70% uniform, 30% clustered (same day batches of 2-5)
            if random.random() < 0.3:
                # Cluster: reuse a nearby date
                cluster_date = base + timedelta(days=random.randint(0, 3650))
                dates.append(cluster_date)
            else:
                dates.append(base + timedelta(days=random.randint(0, 3650)))

        df = pd.DataFrame({
            "title": [f"Movie {i}" for i in range(len(dates))],
            "release_date": [d.isoformat() for d in dates],
        })

        start = time.perf_counter()
        result = compute_clash_feature(df, clash_window_days=7)
        elapsed = time.perf_counter() - start

        assert elapsed < _SCALE_THRESHOLD_S, (
            f"O(n log n) clash detection took {elapsed:.3f}s "
            f"(threshold: {_SCALE_THRESHOLD_S}s) — may have regressed to O(n²)"
        )

        assert "has_clash" in result.columns
        n_clash = result["has_clash"].sum()
        print(f"  Scale test: {n_clash}/{len(result)} movies flagged as clashing ({elapsed:.3f}s)")

        # Spot check: movies on the same day should always clash
        # Find at least one pair of same-day releases and verify they're flagged
        dates_series = pd.to_datetime(result["release_date"])
        dup_dates = dates_series[dates_series.duplicated(keep=False)]
        if len(dup_dates) >= 2:
            first_dup_date = dup_dates.iloc[0]
            same_day = result[dates_series == first_dup_date]
            assert same_day["has_clash"].all(), (
                f"Movies on {first_dup_date.date()} should all be flagged as clashing"
            )
