"""
test_forecasting.py

Unit tests for src/forecasting.py. Where possible these use synthetic
series with a known, hand-computable answer (a pure linear trend, a
constant series) so a wrong result points at a specific code bug rather
than ambiguous noise from real sales data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting import (
    aggregate_monthly_sales,
    build_time_features,
    chronological_train_test_split,
    decompose_time_series,
    evaluate_forecast,
    exponential_smoothing_forecast,
    forecast_confidence_interval,
    generate_sales_forecast,
    moving_average_forecast,
    train_linear_trend_model,
)


def _monthly_index(n_months: int, start: str = "2021-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n_months, freq="MS")


def _linear_series(n_months: int, intercept: float = 1000.0, slope: float = 50.0) -> pd.Series:
    """A perfectly linear series with zero noise and zero seasonality."""
    index = _monthly_index(n_months)
    values = intercept + slope * np.arange(n_months)
    return pd.Series(values, index=index, name="sales")


class TestAggregateMonthlySales:
    def test_sums_within_month(self):
        df = pd.DataFrame({
            "order_month": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-02-01"]),
            "sales": [100.0, 50.0, 200.0],
        })
        result = aggregate_monthly_sales(df)
        assert result.loc[pd.Timestamp("2023-01-01")] == 150.0
        assert result.loc[pd.Timestamp("2023-02-01")] == 200.0

    def test_sorted_ascending_by_month(self):
        df = pd.DataFrame({
            "order_month": pd.to_datetime(["2023-03-01", "2023-01-01", "2023-02-01"]),
            "sales": [10.0, 20.0, 30.0],
        })
        result = aggregate_monthly_sales(df)
        assert list(result.index) == sorted(result.index)


class TestDecomposition:
    def test_trend_recovers_linear_series(self):
        series = _linear_series(36, intercept=1000.0, slope=50.0)
        result = decompose_time_series(series)

        # Centered moving average can't be computed at the edges (NaN there
        # by construction) -- check the well-defined middle of the series.
        mid = result["trend"].dropna()
        expected_mid = series.loc[mid.index]
        assert np.allclose(mid.values, expected_mid.values, atol=1e-6)

    def test_seasonal_near_zero_for_non_seasonal_series(self):
        series = _linear_series(36)
        result = decompose_time_series(series)
        assert abs(result["seasonal"].mean()) < 1e-6

    def test_output_has_expected_columns(self):
        series = _linear_series(24)
        result = decompose_time_series(series)
        assert set(result.columns) == {"observed", "trend", "seasonal", "residual"}


class TestBuildTimeFeatures:
    def test_time_index_is_sequential(self):
        index = _monthly_index(12)
        features = build_time_features(index)
        assert list(features["time_index"]) == list(range(12))

    def test_start_time_index_offsets_correctly(self):
        index = _monthly_index(6, start="2024-01-01")
        features = build_time_features(index, start_time_index=24)
        assert list(features["time_index"]) == list(range(24, 30))

    def test_month_sin_cos_bounded(self):
        index = _monthly_index(24)
        features = build_time_features(index)
        assert features["month_sin"].between(-1.0, 1.0).all()
        assert features["month_cos"].between(-1.0, 1.0).all()

    def test_january_and_december_are_adjacent_in_cyclical_encoding(self):
        # December (month 12) and January (month 1) should be close in
        # (sin, cos) space, unlike their raw integer distance of 11.
        dec = build_time_features(pd.DatetimeIndex(["2023-12-01"]))
        jan = build_time_features(pd.DatetimeIndex(["2024-01-01"]))
        dist = np.hypot(
            dec["month_sin"].iloc[0] - jan["month_sin"].iloc[0],
            dec["month_cos"].iloc[0] - jan["month_cos"].iloc[0],
        )
        assert dist < 0.6  # much smaller than e.g. Jan-vs-July distance


class TestChronologicalSplit:
    def test_test_set_is_last_n_months_in_order(self):
        index = _monthly_index(12)
        features = build_time_features(index)
        target = _linear_series(12)

        _, X_test, _, y_test = chronological_train_test_split(features, target, test_size=3)

        assert list(X_test.index) == list(index[-3:])
        assert list(y_test.index) == list(index[-3:])

    def test_train_and_test_do_not_overlap(self):
        index = _monthly_index(12)
        features = build_time_features(index)
        target = _linear_series(12)

        X_train, X_test, _, _ = chronological_train_test_split(features, target, test_size=4)

        assert set(X_train.index).isdisjoint(set(X_test.index))
        assert len(X_train) == 8
        assert len(X_test) == 4

    def test_raises_when_test_size_too_large(self):
        index = _monthly_index(6)
        features = build_time_features(index)
        target = _linear_series(6)

        with pytest.raises(ValueError):
            chronological_train_test_split(features, target, test_size=6)


class TestBaselineForecasts:
    def test_moving_average_on_constant_series(self):
        series = pd.Series([100.0] * 12)
        forecast = moving_average_forecast(series, n_periods=4, window=3)
        assert np.allclose(forecast, 100.0)
        assert len(forecast) == 4

    def test_exponential_smoothing_on_constant_series(self):
        series = pd.Series([100.0] * 12)
        forecast = exponential_smoothing_forecast(series, n_periods=4, alpha=0.3)
        assert np.allclose(forecast, 100.0)

    def test_moving_average_uses_only_recent_window(self):
        series = pd.Series([0.0] * 20 + [100.0, 200.0, 300.0])
        forecast = moving_average_forecast(series, n_periods=1, window=3)
        assert forecast[0] == pytest.approx(200.0)  # mean of last 3 values


class TestEvaluateForecast:
    def test_perfect_prediction_has_zero_error(self):
        y_true = pd.Series([100.0, 200.0, 300.0])
        result = evaluate_forecast(y_true, y_true)
        assert result["mae"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["mape"] == pytest.approx(0.0)

    def test_known_mae_and_mape(self):
        y_true = pd.Series([100.0, 200.0])
        y_pred = pd.Series([110.0, 180.0])
        result = evaluate_forecast(y_true, y_pred)
        # errors: 10 and 20 -> MAE = 15
        assert result["mae"] == pytest.approx(15.0)
        # percentage errors: 10% and 10% -> MAPE = 10%
        assert result["mape"] == pytest.approx(10.0)


class TestLinearTrendModel:
    def test_recovers_slope_on_noiseless_linear_series(self):
        series = _linear_series(36, intercept=1000.0, slope=50.0)
        features = build_time_features(series.index)
        model = train_linear_trend_model(features, series)

        # month_sin/month_cos coefficients should contribute ~0 since there's
        # no seasonality; the time_index coefficient should recover the slope.
        assert model.coef_[0] == pytest.approx(50.0, abs=0.5)

    def test_predicts_future_points_accurately_on_noiseless_series(self):
        series = _linear_series(36, intercept=1000.0, slope=50.0)
        features = build_time_features(series.index)
        model = train_linear_trend_model(features, series)

        future_index = _monthly_index(3, start="2024-01-01")
        future_features = build_time_features(future_index, start_time_index=36)
        predictions = model.predict(future_features)
        expected = 1000.0 + 50.0 * np.arange(36, 39)

        assert np.allclose(predictions, expected, atol=1.0)


class TestForecastConfidenceInterval:
    def test_interval_widens_with_higher_confidence(self):
        residuals = pd.Series(np.random.default_rng(1).normal(0, 10, 30))
        forecast = pd.Series([500.0])

        low_90, high_90 = forecast_confidence_interval(residuals, forecast, confidence=0.90)
        low_99, high_99 = forecast_confidence_interval(residuals, forecast, confidence=0.99)

        assert (high_99.iloc[0] - low_99.iloc[0]) > (high_90.iloc[0] - low_90.iloc[0])

    def test_interval_is_centered_on_forecast(self):
        residuals = pd.Series(np.random.default_rng(2).normal(0, 5, 24))
        forecast = pd.Series([500.0])
        lower, upper = forecast_confidence_interval(residuals, forecast, confidence=0.95)
        midpoint = (lower.iloc[0] + upper.iloc[0]) / 2
        assert midpoint == pytest.approx(500.0)


class TestGenerateSalesForecast:
    def _sample_analytics_view(self, n_months: int = 30) -> pd.DataFrame:
        """Synthetic flat-view-shaped data: several sale rows per month."""
        rng = np.random.default_rng(42)
        months = _monthly_index(n_months)
        rows = []
        for month in months:
            n_sales = 5
            base = 1000 + 20 * (month.year - 2021) * 12 + 20 * month.month
            for _ in range(n_sales):
                rows.append({"order_month": month, "sales": base / n_sales + rng.normal(0, 5)})
        return pd.DataFrame(rows)

    def test_returns_expected_keys(self):
        df = self._sample_analytics_view()
        result = generate_sales_forecast(df, forecast_periods=4, test_size=6)

        expected_keys = {
            "monthly_sales", "decomposition", "backtest_evaluation",
            "future_forecast", "future_forecast_ci", "model",
        }
        assert set(result.keys()) == expected_keys

    def test_future_forecast_has_requested_length_and_continues_chronologically(self):
        df = self._sample_analytics_view()
        result = generate_sales_forecast(df, forecast_periods=4, test_size=6)

        future = result["future_forecast"]
        last_historical_month = result["monthly_sales"].index[-1]

        assert len(future) == 4
        assert future.index[0] == last_historical_month + pd.DateOffset(months=1)

    def test_backtest_evaluation_covers_all_three_methods(self):
        df = self._sample_analytics_view()
        result = generate_sales_forecast(df, forecast_periods=3, test_size=6)

        assert set(result["backtest_evaluation"].keys()) == {
            "linear_regression", "moving_average", "exponential_smoothing",
        }
        for metrics in result["backtest_evaluation"].values():
            assert set(metrics.keys()) == {"mae", "rmse", "mape"}
