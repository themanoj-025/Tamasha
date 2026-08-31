"""Tests for festival calendar utility."""


from datetime import date

from tamasha.timing.festival_calendar import (
    compute_festival_features,
    get_major_release_windows,
    is_festival_release,
)


class TestIsFestivalRelease:
    """Tests for is_festival_release."""

    def test_normal_release(self) -> None:
        """A June release should not be in any festival window."""
        windows = get_major_release_windows(2024)
        result = is_festival_release(date(2024, 6, 15), windows)
        assert isinstance(result, tuple)
        assert len(result) == 3  # (is_festival, festival_name, days_to_festival)

    def test_returns_tuple(self) -> None:
        """Return type should always be a 3-tuple."""
        windows = get_major_release_windows(2024)
        result = is_festival_release(date(2024, 1, 1), windows)
        assert isinstance(result, tuple)
        assert len(result) == 3


class TestComputeFestivalFeatures:
    """Tests for compute_festival_features."""

    def test_is_callable(self) -> None:
        """compute_festival_features should be callable."""
        assert callable(compute_festival_features)


class TestGetMajorReleaseWindows:
    """Tests for get_major_release_windows."""

    def test_returns_dict(self) -> None:
        windows = get_major_release_windows(2024)
        assert isinstance(windows, dict)

    def test_has_festival_entries(self) -> None:
        windows = get_major_release_windows(2024)
        assert len(windows) > 0
