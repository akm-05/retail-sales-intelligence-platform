"""
recommendations.py

Generates the dashboard's "Business Recommendations" panel content.

Every recommendation here is computed from whatever data is currently
passed in -- nothing is hardcoded prose about "the Superstore dataset."
That matters for two reasons: it means the panel updates when the sidebar
filters change (recommendations for "West region, Technology only" look
different from the unfiltered view), and it means this module is testable
the same way statistics.py and forecasting.py are -- build a small
synthetic DataFrame with a known pattern baked in, assert the expected
recommendation comes out.

Kept deliberately rule-based rather than templated-but-vague. Each
function answers one concrete business question ("which category should
we be worried about") and returns a recommendation only when the data
actually supports one -- e.g. a discount-threshold recommendation is
skipped entirely if there's no discount tier where the average margin
turns negative, rather than forcing a finding that isn't there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _format_currency(value: float) -> str:
    return f"${value:,.0f}"


def _format_pct(value: float) -> str:
    return f"{value:.1f}%"


def recommend_category_focus(df: pd.DataFrame) -> dict | None:
    """
    Flags the category that earns the most revenue but has a margin below
    the portfolio's overall average -- a strong revenue contributor that is
    quietly underperforming on profitability relative to the rest of the
    business.
    """
    by_category = df.groupby("category_name").agg(sales=("sales", "sum"), profit=("profit", "sum"))
    by_category["margin"] = by_category["profit"] / by_category["sales"]
    overall_margin = df["profit"].sum() / df["sales"].sum()

    top_revenue_category = by_category["sales"].idxmax()
    category_margin = by_category.loc[top_revenue_category, "margin"]

    if category_margin >= overall_margin:
        return None

    return {
        "title": f"Investigate margin in {top_revenue_category}",
        "detail": (
            f"{top_revenue_category} is the single largest revenue driver "
            f"({_format_currency(by_category.loc[top_revenue_category, 'sales'])}), but its "
            f"margin ({_format_pct(category_margin * 100)}) trails the overall average "
            f"({_format_pct(overall_margin * 100)}). Even a small margin improvement here "
            "has outsized impact given the category's scale."
        ),
    }


def recommend_region_focus(df: pd.DataFrame) -> dict | None:
    """
    Flags the region with the lowest margin among regions that contribute
    a meaningful (>10%) share of total revenue -- filtering out small
    regions where a low margin might just be noise from a handful of
    orders.
    """
    by_region = df.groupby("region_name").agg(sales=("sales", "sum"), profit=("profit", "sum"))
    by_region["margin"] = by_region["profit"] / by_region["sales"]
    by_region["revenue_share"] = by_region["sales"] / by_region["sales"].sum()

    material = by_region[by_region["revenue_share"] > 0.10]
    if material.empty:
        return None

    worst_region = material["margin"].idxmin()
    row = material.loc[worst_region]

    return {
        "title": f"Review pricing or cost structure in {worst_region}",
        "detail": (
            f"{worst_region} contributes {_format_pct(row['revenue_share'] * 100)} of total "
            f"revenue but converts it at only {_format_pct(row['margin'] * 100)} margin -- "
            "the lowest of any region with material revenue share."
        ),
    }


def recommend_discount_threshold(df: pd.DataFrame, bucket_width: float = 0.1) -> dict | None:
    """
    Buckets sales by discount level and finds the lowest discount bucket
    where average profit margin turns negative -- a concrete, defensible
    "don't discount past this" number rather than a general warning.
    """
    bucketed = df.assign(
        discount_bucket=np.floor(np.round(df["discount"] / bucket_width, 8)) * bucket_width
    )
    by_bucket = bucketed.groupby("discount_bucket")["profit_margin"].mean().sort_index()

    negative_buckets = by_bucket[by_bucket < 0]
    if negative_buckets.empty:
        return None

    threshold = negative_buckets.index[0]
    return {
        "title": "Cap discounts below the margin break-even point",
        "detail": (
            f"Average profit margin turns negative once discounts reach "
            f"{_format_pct(threshold * 100)} and above. Discounting up to this level "
            "still preserves margin; beyond it, sales are being made at a loss on average."
        ),
    }


def recommend_segment_focus(df: pd.DataFrame) -> dict | None:
    """
    Identifies the top revenue-generating customer segment as a retention
    and account-management priority.
    """
    by_segment = df.groupby("segment")["sales"].sum().sort_values(ascending=False)
    if by_segment.empty:
        return None

    top_segment = by_segment.index[0]
    share = by_segment.iloc[0] / by_segment.sum()

    return {
        "title": f"Prioritize retention in the {top_segment} segment",
        "detail": (
            f"{top_segment} accounts for {_format_pct(share * 100)} of total revenue in the "
            "current view -- the largest single segment. Retention or expansion effort here "
            "has more leverage than an equivalent effort spread evenly across segments."
        ),
    }


def recommend_forecast_trend(future_forecast: pd.Series | None, monthly_sales: pd.Series | None) -> dict | None:
    """
    Compares the average of the forecasted future months against the
    average of the most recent equivalent-length historical window, and
    frames the forecast direction as a staffing/inventory planning signal.
    """
    if future_forecast is None or monthly_sales is None or future_forecast.empty:
        return None

    horizon = len(future_forecast)
    recent_actual = monthly_sales.iloc[-horizon:].mean()
    forecast_avg = future_forecast.mean()
    change = (forecast_avg - recent_actual) / recent_actual if recent_actual else 0.0

    if change > 0.03:
        direction, guidance = "growth", "consider scaling inventory and staffing ahead of the increase"
    elif change < -0.03:
        direction, guidance = "decline", "consider tightening inventory commitments to avoid excess stock"
    else:
        direction, guidance = "flat", "no major inventory or staffing shift appears warranted"

    return {
        "title": f"Forecast signals {direction} over the next {horizon} months",
        "detail": (
            f"Forecasted average monthly revenue ({_format_currency(forecast_avg)}) is "
            f"{_format_pct(abs(change) * 100)} {'above' if change >= 0 else 'below'} the most "
            f"recent {horizon}-month actual average ({_format_currency(recent_actual)}) -- {guidance}."
        ),
    }


def generate_recommendations(
    df: pd.DataFrame,
    future_forecast: pd.Series | None = None,
    monthly_sales: pd.Series | None = None,
) -> list[dict]:
    """
    Run every recommendation rule against the current (filtered) data and
    return whichever ones actually found something worth flagging.

    Args:
        df: Flat analytics view, already filtered to whatever the
            dashboard's sidebar selections currently are.
        future_forecast: Optional forecast series from
                          forecasting.generate_sales_forecast(), to enable
                          recommend_forecast_trend().
        monthly_sales: Optional historical monthly series, paired with
                        future_forecast for the same reason.

    Returns:
        List of {"title": str, "detail": str} dicts, in priority order.
        Can be shorter than the number of rules if some found nothing to
        flag -- an empty list is a valid, honest result.
    """
    candidates = [
        recommend_category_focus(df),
        recommend_region_focus(df),
        recommend_discount_threshold(df),
        recommend_segment_focus(df),
        recommend_forecast_trend(future_forecast, monthly_sales),
    ]
    return [rec for rec in candidates if rec is not None]
