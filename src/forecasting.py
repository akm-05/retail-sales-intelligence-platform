"""
forecasting.py

Phase 6 of the pipeline: turns the flat analytics view from
feature_engineering.build_analytics_view() into a monthly revenue forecast.

Design decisions (worth explaining in an interview):

- No statsmodels. The declared tech stack for this project is pandas /
  numpy / scikit-learn / scipy -- not statsmodels -- so ARIMA/SARIMA are
  deliberately out of scope. Trend and seasonality are instead captured as
  engineered features (a numeric time index for trend, sin/cos encoding of
  calendar month for seasonality) fed into a plain scikit-learn
  LinearRegression. That's easier to explain end-to-end on a whiteboard
  than justifying an ARIMA order-selection process, and for a monthly
  series with a few years of history it's a legitimate forecasting
  approach in its own right -- not a fallback.
- Two naive baselines (moving average, simple exponential smoothing) are
  computed alongside the regression model. A forecast is only meaningful
  relative to a baseline -- if the regression model can't beat a moving
  average on the holdout period, that's itself a useful, honest finding to
  report rather than something to hide.
- The train/test split is chronological (last N months held out), never a
  random split. Shuffling a time series before splitting leaks future
  information into training and silently inflates accuracy -- a classic
  mistake worth being explicit about avoiding.
- Classical (moving-average-based) decomposition into trend / seasonal /
  residual is included as a diagnostic step, run before any model is fit.
  It's the "look at the data first" step that justifies using a 12-month
  seasonal period everywhere else in this module, rather than assuming it.
- Prediction intervals around the future forecast reuse the same
  t-distribution approach as statistics.confidence_interval_mean() --
  keeping the one statistical technique for "how much should I trust this
  estimate" consistent across the whole project instead of introducing a
  second one just for forecasting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt

from src.visualization import ACCENT_COLOR, PRIMARY_COLOR, _save

# Monthly data, annual seasonality -- fixed rather than parameterized, since
# every other assumption in this module (the 2x12 centered moving average,
# the sin/cos month encoding) is built specifically around a 12-month cycle.
SEASONAL_PERIOD = 12

DEFAULT_TEST_SIZE = 6
DEFAULT_FORECAST_PERIODS = 6
DEFAULT_MA_WINDOW = 3
DEFAULT_EXP_ALPHA = 0.3


# ----------------------------------------------------------------------------
# 1. MONTHLY AGGREGATION
# ----------------------------------------------------------------------------

def aggregate_monthly_sales(df: pd.DataFrame, value_col: str = "sales") -> pd.Series:
    """
    Collapse the flat analytics view (one row per sale) into one total per
    calendar month -- the series every downstream function in this module
    operates on.

    Args:
        df: Flat analytics view from feature_engineering.build_analytics_view()
            (must contain 'order_month' and value_col).
        value_col: Column to sum per month, default 'sales'.

    Returns:
        A pd.Series indexed by order_month (monthly Timestamps, sorted
        ascending), named value_col.
    """
    monthly = df.groupby("order_month")[value_col].sum().sort_index()
    monthly.index.name = "order_month"
    monthly.name = value_col
    return monthly


# ----------------------------------------------------------------------------
# 2. CLASSICAL DECOMPOSITION (DIAGNOSTIC STEP)
# ----------------------------------------------------------------------------

def decompose_time_series(series: pd.Series) -> pd.DataFrame:
    """
    Classical additive decomposition: observed = trend + seasonal + residual.

    Trend is a 2x12 centered moving average (the standard way to center a
    moving average when the seasonal period is even). Seasonal is the
    average detrended value per calendar month, centered around zero so the
    twelve seasonal components sum to ~0 -- i.e. seasonality shifts sales
    between months without changing the annual total. Residual is whatever
    is left over.

    Args:
        series: Monthly series from aggregate_monthly_sales(). Needs at
                least ~2 full years of data for the trend/seasonal split to
                be meaningful.

    Returns:
        DataFrame indexed the same as `series`, with columns:
        observed, trend, seasonal, residual.
    """
    trend = series.rolling(window=SEASONAL_PERIOD, center=True).mean()
    trend = trend.rolling(window=2, center=True).mean()

    # pandas' rolling(center=True) is NOT symmetric for an even-length
    # window: each pass places its result half a period earlier than the
    # window's true midpoint (verified empirically -- window=4 centered on
    # index i averages s[i-2 : i+2), whose midpoint is i-0.5, not i). Two
    # passes compound to a full one-period lag, so the result is shifted
    # forward by one month to land back on the correct month. Without this,
    # `trend` silently lines up with the wrong month -- worth flagging
    # explicitly rather than leaving as a subtle off-by-one.
    trend = trend.shift(-1)

    detrended = series - trend
    seasonal_index = detrended.groupby(detrended.index.month).mean()
    seasonal_index = seasonal_index - seasonal_index.mean()  # center around zero

    seasonal = series.index.month.map(seasonal_index)
    seasonal = pd.Series(seasonal, index=series.index, name="seasonal")

    residual = series - trend - seasonal

    return pd.DataFrame({
        "observed": series,
        "trend": trend,
        "seasonal": seasonal,
        "residual": residual,
    })


# ----------------------------------------------------------------------------
# 3. FEATURE ENGINEERING FOR THE FORECASTING MODEL
# ----------------------------------------------------------------------------

def build_time_features(index: pd.DatetimeIndex, start_time_index: int = 0) -> pd.DataFrame:
    """
    Build the model's feature matrix from a DatetimeIndex alone.

    Taking an index (rather than the series itself) is what lets this same
    function generate features for historical months (to train on) and for
    future months that don't have a sales value yet (to predict on).

    Features:
        time_index: sequential integer capturing the linear trend
                     (0, 1, 2, ... continuing across historical -> future).
        month_sin / month_cos: cyclical encoding of calendar month, so
                     December (12) and January (1) are recognized as
                     adjacent rather than 11 units apart, as a plain integer
                     month number would imply.

    Args:
        index: Monthly DatetimeIndex to build features for.
        start_time_index: Value time_index should start counting from --
                           pass len(historical_series) when building
                           features for future months, so trend continues
                           smoothly rather than resetting to 0.

    Returns:
        DataFrame indexed the same as `index`, columns:
        time_index, month_sin, month_cos.
    """
    n = len(index)
    time_index = np.arange(start_time_index, start_time_index + n)
    month = index.month.to_numpy()
    month_sin = np.sin(2 * np.pi * month / SEASONAL_PERIOD)
    month_cos = np.cos(2 * np.pi * month / SEASONAL_PERIOD)

    return pd.DataFrame(
        {"time_index": time_index, "month_sin": month_sin, "month_cos": month_cos},
        index=index,
    )


# ----------------------------------------------------------------------------
# 4. TRAIN/TEST SPLIT (CHRONOLOGICAL -- NEVER RANDOM)
# ----------------------------------------------------------------------------

def chronological_train_test_split(
    features: pd.DataFrame, target: pd.Series, test_size: int = DEFAULT_TEST_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Hold out the last `test_size` months as the test set, in time order.

    Deliberately does not shuffle. A random split would let the model train
    on months that come after some of its test months, which is a form of
    lookahead bias -- the model would be evaluated on a task it could never
    actually face in production (predicting the future from the past only).

    Args:
        features: Output of build_time_features(), one row per month.
        target: Monthly series from aggregate_monthly_sales(), same index.
        test_size: Number of most recent months to hold out.

    Returns:
        (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If test_size >= len(features), leaving no training data.
    """
    if test_size >= len(features):
        raise ValueError(
            f"test_size ({test_size}) must be smaller than the number of "
            f"months available ({len(features)})"
        )
    X_train, X_test = features.iloc[:-test_size], features.iloc[-test_size:]
    y_train, y_test = target.iloc[:-test_size], target.iloc[-test_size:]
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------------
# 5. BASELINE FORECASTING METHODS
# ----------------------------------------------------------------------------

def moving_average_forecast(train_series: pd.Series, n_periods: int, window: int = DEFAULT_MA_WINDOW) -> np.ndarray:
    """
    Naive baseline: forecast every future period as the average of the last
    `window` observed values. Flat by construction -- it carries no trend
    or seasonality forward, which is exactly why it's a useful benchmark:
    the regression model earns its keep only if it beats this.
    """
    last_average = train_series.iloc[-window:].mean()
    return np.full(shape=n_periods, fill_value=last_average)


def exponential_smoothing_forecast(train_series: pd.Series, n_periods: int, alpha: float = DEFAULT_EXP_ALPHA) -> np.ndarray:
    """
    Naive baseline: simple exponential smoothing. Each observation updates
    a running "level" with weight `alpha`, and every future period is
    forecast as that final level (also flat -- no trend/seasonality).

    Args:
        train_series: Historical values, in time order.
        n_periods: Number of future periods to forecast.
        alpha: Smoothing weight in (0, 1]. Higher alpha weights recent
               observations more heavily; 0.3 is a common, mild default.
    """
    level = train_series.iloc[0]
    for value in train_series.iloc[1:]:
        level = alpha * value + (1 - alpha) * level
    return np.full(shape=n_periods, fill_value=level)


# ----------------------------------------------------------------------------
# 6. MODEL TRAINING & EVALUATION
# ----------------------------------------------------------------------------

def train_linear_trend_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Fit the trend + seasonality regression model on training features."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def _mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAPE, guarded against division by zero the same way
    feature_engineering.add_profit_margin() guards profit / sales -- months
    with zero actual sales are excluded from the percentage-error average
    rather than producing inf.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero = y_true != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)


def evaluate_forecast(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict:
    """
    Compute the three standard forecast-accuracy metrics for a set of
    predictions:

    - MAE: average absolute error, in the same units as sales (USD) --
      the easiest one to explain to a business stakeholder.
    - RMSE: penalizes large misses more heavily than MAE, useful for
      flagging a model that's usually close but occasionally very wrong.
    - MAPE: error as a percentage of the actual value -- lets you compare
      forecast quality across series of very different scale (e.g. a small
      region vs. a large one) on one common axis.

    Returns:
        Dict with keys: mae, rmse, mape.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = _mean_absolute_percentage_error(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "mape": mape}


# ----------------------------------------------------------------------------
# 7. FUTURE FORECAST GENERATION
# ----------------------------------------------------------------------------

def forecast_confidence_interval(
    residuals: pd.Series, forecast: pd.Series, confidence: float = 0.95
) -> tuple[pd.Series, pd.Series]:
    """
    Build a symmetric prediction interval around the future forecast, using
    the same t-distribution approach as
    statistics.confidence_interval_mean() -- the margin is a t-critical
    value times the standard deviation of the model's in-sample residuals.

    This is an approximation (it assumes constant residual variance across
    future horizons, i.e. it doesn't widen the further out you forecast),
    which is a reasonable, explainable simplification for a project at this
    scope -- worth naming as a limitation rather than hiding.

    Args:
        residuals: In-sample (training) residuals: y_train - model.predict(X_train).
        forecast: The future point forecast to build an interval around.
        confidence: Confidence level, e.g. 0.95 for a 95% interval.

    Returns:
        (lower_bound, upper_bound), each a pd.Series aligned to `forecast`.
    """
    n = len(residuals)
    resid_std = residuals.std(ddof=1)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * resid_std
    return forecast - margin, forecast + margin


# ----------------------------------------------------------------------------
# 8. ORCHESTRATION
# ----------------------------------------------------------------------------

def generate_sales_forecast(
    df: pd.DataFrame,
    forecast_periods: int = DEFAULT_FORECAST_PERIODS,
    test_size: int = DEFAULT_TEST_SIZE,
    ma_window: int = DEFAULT_MA_WINDOW,
    exp_alpha: float = DEFAULT_EXP_ALPHA,
) -> dict:
    """
    End-to-end forecasting pipeline: aggregate -> decompose -> backtest
    three methods on a holdout period -> refit on full history -> forecast
    forward.

    Backtesting (step 2) and the real future forecast (step 3) deliberately
    use two different model fits. The backtest model only ever sees
    training months, so its holdout accuracy is a fair estimate of
    real-world performance. Once that accuracy is reported, refitting on
    the FULL history before forecasting the true future is correct practice
    -- there's no reason to withhold recent months from the model once
    they're no longer being used to test it.

    Args:
        df: Flat analytics view from feature_engineering.build_analytics_view().
        forecast_periods: Number of future months to forecast.
        test_size: Number of most recent historical months held out for backtesting.
        ma_window: Window size for the moving-average baseline.
        exp_alpha: Smoothing weight for the exponential-smoothing baseline.

    Returns:
        Dict with keys:
            monthly_sales: pd.Series, historical monthly totals.
            decomposition: pd.DataFrame from decompose_time_series().
            backtest_evaluation: dict of method_name -> evaluate_forecast() dict,
                                  for 'linear_regression', 'moving_average',
                                  and 'exponential_smoothing'.
            future_forecast: pd.Series, point forecast for the next forecast_periods months.
            future_forecast_ci: (lower, upper) tuple of pd.Series, 95% prediction interval.
            model: the LinearRegression fitted on the full history (used for future_forecast).
    """
    monthly_sales = aggregate_monthly_sales(df)
    decomposition = decompose_time_series(monthly_sales)

    # --- Backtest: three methods, evaluated on the same held-out months ---
    features = build_time_features(monthly_sales.index)
    X_train, X_test, y_train, y_test = chronological_train_test_split(features, monthly_sales, test_size)

    backtest_model = train_linear_trend_model(X_train, y_train)
    regression_pred = backtest_model.predict(X_test)
    ma_pred = moving_average_forecast(y_train, test_size, window=ma_window)
    exp_pred = exponential_smoothing_forecast(y_train, test_size, alpha=exp_alpha)

    backtest_evaluation = {
        "linear_regression": evaluate_forecast(y_test, regression_pred),
        "moving_average": evaluate_forecast(y_test, ma_pred),
        "exponential_smoothing": evaluate_forecast(y_test, exp_pred),
    }

    # --- Future forecast: refit on the full history, then forecast forward ---
    full_model = train_linear_trend_model(features, monthly_sales)
    future_index = pd.date_range(
        start=monthly_sales.index[-1] + pd.DateOffset(months=1),
        periods=forecast_periods,
        freq="MS",
    )
    future_features = build_time_features(future_index, start_time_index=len(monthly_sales))
    future_forecast = pd.Series(
        full_model.predict(future_features), index=future_index, name="sales_forecast"
    )

    in_sample_residuals = monthly_sales - full_model.predict(features)
    future_forecast_ci = forecast_confidence_interval(in_sample_residuals, future_forecast)

    return {
        "monthly_sales": monthly_sales,
        "decomposition": decomposition,
        "backtest_evaluation": backtest_evaluation,
        "future_forecast": future_forecast,
        "future_forecast_ci": future_forecast_ci,
        "model": full_model,
    }


# ----------------------------------------------------------------------------
# 9. VISUALIZATION
# ----------------------------------------------------------------------------
# Reuses visualization.py's color palette and _save() helper rather than
# redefining them, so a forecast chart looks like part of the same report
# as the Phase 4 EDA charts instead of a visually separate module.

def plot_decomposition(decomposition: pd.DataFrame, save_path: str | Path) -> plt.Figure:
    """
    Business question: is the revenue trend genuinely growing/declining
    once seasonality is removed, and how large is the seasonal swing
    relative to that trend?

    Four stacked line charts sharing an x-axis: observed, trend, seasonal,
    residual -- the standard classical-decomposition view.
    """
    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    components = ["observed", "trend", "seasonal", "residual"]
    titles = ["Observed", "Trend", "Seasonal", "Residual"]

    for ax, component, title in zip(axes, components, titles):
        ax.plot(decomposition.index, decomposition[component], color=PRIMARY_COLOR)
        ax.set_ylabel(title)
        ax.axhline(0, color="#CCCCCC", linewidth=0.8) if component in ("seasonal", "residual") else None

    axes[0].set_title("Monthly revenue: classical decomposition")
    axes[-1].set_xlabel("Month")
    fig.autofmt_xdate()

    _save(fig, save_path)
    return fig


def plot_forecast(
    historical: pd.Series,
    future_forecast: pd.Series,
    save_path: str | Path,
    future_forecast_ci: tuple[pd.Series, pd.Series] | None = None,
) -> plt.Figure:
    """
    Business question: given the trend and seasonality observed so far,
    what does revenue look like over the next few months, and how much
    uncertainty should be attached to that number?

    Line chart of historical monthly revenue, with the future forecast
    continuing as a dashed line, and an optional shaded prediction
    interval around it.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(historical.index, historical.values, color=PRIMARY_COLOR, marker="o", markersize=3, label="Historical")
    ax.plot(
        future_forecast.index, future_forecast.values,
        color=ACCENT_COLOR, marker="o", markersize=3, linestyle="--", label="Forecast",
    )

    if future_forecast_ci is not None:
        lower, upper = future_forecast_ci
        ax.fill_between(future_forecast.index, lower, upper, color=ACCENT_COLOR, alpha=0.15, label="95% interval")

    ax.set_title("Monthly revenue forecast")
    ax.set_ylabel("Revenue (USD)")
    ax.set_xlabel("Month")
    ax.legend()
    fig.autofmt_xdate()

    _save(fig, save_path)
    return fig
