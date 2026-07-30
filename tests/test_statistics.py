"""
test_statistics.py

Unit tests for src/statistics.py. Each test uses a small series with a
known, hand-computable answer, so a wrong result is a clear code bug
rather than ambiguous synthetic-data noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.statistics import (
    compare_two_groups_ttest,
    compute_descriptive_stats,
    confidence_interval_mean,
    detect_outliers_iqr,
    detect_outliers_zscore,
)


class TestDescriptiveStats:
    def test_known_mean_and_median(self):
        series = pd.Series([1, 2, 3, 4, 5])
        result = compute_descriptive_stats(series)
        assert result["mean"] == 3
        assert result["median"] == 3
        assert result["count"] == 5

    def test_drops_nan_before_computing(self):
        series = pd.Series([1, 2, 3, np.nan])
        result = compute_descriptive_stats(series)
        assert result["count"] == 3
        assert result["mean"] == 2


class TestOutlierDetection:
    def test_iqr_flags_extreme_value(self):
        series = pd.Series([10, 11, 12, 13, 14, 500])
        flags = detect_outliers_iqr(series)
        assert flags.iloc[-1] == True  # noqa: E712
        assert flags.iloc[:-1].sum() == 0

    def test_zscore_flags_extreme_value(self):
        series = pd.Series([10.0] * 20 + [1000.0])
        flags = detect_outliers_zscore(series, threshold=3.0)
        assert flags.iloc[-1] == True  # noqa: E712

    def test_zscore_no_false_positives_on_uniform_data(self):
        series = pd.Series([10.0] * 20)
        flags = detect_outliers_zscore(series, threshold=3.0)
        assert flags.sum() == 0


class TestConfidenceInterval:
    def test_ci_contains_sample_mean(self):
        series = pd.Series(np.random.default_rng(1).normal(50, 5, 200))
        lower, upper = confidence_interval_mean(series, confidence=0.95)
        assert lower < series.mean() < upper

    def test_wider_confidence_gives_wider_interval(self):
        series = pd.Series(np.random.default_rng(1).normal(50, 5, 200))
        low_90, high_90 = confidence_interval_mean(series, confidence=0.90)
        low_99, high_99 = confidence_interval_mean(series, confidence=0.99)
        assert (high_99 - low_99) > (high_90 - low_90)


class TestTTest:
    def test_detects_significant_difference(self):
        rng = np.random.default_rng(2)
        group_a = pd.Series(rng.normal(10, 1, 100))
        group_b = pd.Series(rng.normal(20, 1, 100))
        result = compare_two_groups_ttest(group_a, group_b)
        assert result["significant"] is True
        assert result["p_value"] < 0.05

    def test_no_significant_difference_for_identical_distributions(self):
        rng = np.random.default_rng(3)
        group_a = pd.Series(rng.normal(10, 1, 200))
        group_b = pd.Series(rng.normal(10, 1, 200))
        result = compare_two_groups_ttest(group_a, group_b)
        assert result["significant"] is False
