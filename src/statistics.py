"""
statistics.py

Statistical analysis functions used to move from "here's what the chart
shows" (EDA) to "here's whether that pattern is statistically defensible"
(this module). Every function is written to be reused on any numeric
column/grouping -- the business-specific interpretation happens where
these functions are called, not inside them.

Uses scipy.stats for the hypothesis testing (independent-samples t-test
and the t-distribution critical value for confidence intervals) --
reimplementing those from scratch would add risk of a subtle math error
for no real benefit; scipy's implementations are the standard, trusted
choice here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_descriptive_stats(series: pd.Series) -> dict:
    """
    Core descriptive statistics for a numeric column: mean, median, mode,
    variance, standard deviation, skewness.

    Reporting mean AND median together is deliberate -- a large gap
    between them (as seen with `sales` in Phase 4) is itself a business
    signal that the mean is being distorted by outliers.

    Args:
        series: A numeric pandas Series (NaNs are dropped before computing).

    Returns:
        Dict of statistic name -> value.
    """
    clean = series.dropna()
    return {
        "count": len(clean),
        "mean": clean.mean(),
        "median": clean.median(),
        "mode": clean.mode().iloc[0] if not clean.mode().empty else None,
        "std_dev": clean.std(),
        "variance": clean.var(),
        "skewness": clean.skew(),
        "min": clean.min(),
        "max": clean.max(),
    }


def compute_correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Pearson correlation matrix across the given numeric columns.

    Returned as a DataFrame (not a heatmap) so it can be reused both by
    visualization.plot_correlation_heatmap() and by any report that needs
    the raw numbers.
    """
    return df[columns].corr()


def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """
    Flag outliers using the IQR (interquartile range) method: any value
    below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.

    This is the standard, robust-to-skew outlier method -- preferred over
    z-score for a right-skewed column like `sales` (see Phase 4, Chart 9),
    since z-score assumes a roughly normal distribution and is itself
    distorted by the same extreme values it's meant to detect.

    Args:
        series: Numeric column to check.

    Returns:
        Boolean Series, same index as input, True where the value is an
        outlier.
    """
    clean = series.dropna()
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (series < lower_bound) | (series > upper_bound)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Flag outliers using the z-score method: any value more than
    `threshold` standard deviations from the mean.

    Best suited to roughly-normal columns (e.g. profit_margin, which is
    centered rather than heavily skewed) -- for a skewed column, prefer
    detect_outliers_iqr() instead.

    Args:
        series: Numeric column to check.
        threshold: Number of standard deviations that defines an outlier.
                   3.0 is the conventional default.

    Returns:
        Boolean Series, same index as input, True where the value is an
        outlier. NaN inputs are never flagged.
    """
    clean = series.dropna()
    z_scores = (clean - clean.mean()) / clean.std()
    is_outlier = z_scores.abs() > threshold
    return is_outlier.reindex(series.index, fill_value=False)


def confidence_interval_mean(series: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """
    Compute a confidence interval for the population mean using the
    t-distribution (appropriate for a sample, as opposed to the z-distribution
    which assumes a known population standard deviation).

    Args:
        series: Numeric column.
        confidence: Confidence level, e.g. 0.95 for a 95% CI.

    Returns:
        (lower_bound, upper_bound) tuple.
    """
    clean = series.dropna()
    n = len(clean)
    mean = clean.mean()
    sem = stats.sem(clean)  # standard error of the mean
    margin = sem * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return (mean - margin, mean + margin)


def compare_two_groups_ttest(
    group_a: pd.Series, group_b: pd.Series, alpha: float = 0.05
) -> dict:
    """
    Independent-samples t-test (Welch's, unequal variance assumed by
    default) comparing the means of two groups.

    Welch's t-test is used instead of the standard Student's t-test because
    it does not assume the two groups have equal variance -- a safer
    default when comparing business segments (e.g. one region vs. the
    rest) that have no reason to share identical variance.

    Args:
        group_a: Numeric values for group A (e.g. West region profit margin).
        group_b: Numeric values for group B (e.g. all other regions' profit margin).
        alpha: Significance threshold, default 0.05 (95% confidence).

    Returns:
        Dict with: mean_a, mean_b, mean_difference, t_statistic, p_value,
        significant (bool), and a plain-language interpretation string.
    """
    a, b = group_a.dropna(), group_b.dropna()
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    mean_a, mean_b = a.mean(), b.mean()
    significant = bool(p_value < alpha)

    interpretation = (
        f"The difference between the two group means ({mean_a:.4f} vs. {mean_b:.4f}) "
        f"is {'statistically significant' if significant else 'not statistically significant'} "
        f"at the {int((1 - alpha) * 100)}% confidence level (p = {p_value:.4f})."
    )

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_difference": mean_a - mean_b,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": significant,
        "interpretation": interpretation,
    }
