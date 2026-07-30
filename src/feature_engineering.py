"""
feature_engineering.py

Builds a single flat "analytics view" from the eight normalized tables
produced by preprocessing.py, and adds the small set of derived features
(time parts, profit margin) that EDA, statistics, and forecasting all need.

This is deliberately the ONE place these joins happen. Every downstream
module (visualization.py, statistics.py, forecasting.py) consumes this
flat view rather than re-joining the normalized tables itself -- that
keeps the join logic in one place and guarantees every module is working
from the same definition of "a sale."
"""

from __future__ import annotations

import pandas as pd


def build_analytics_view(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join the normalized tables into one flat, analysis-ready DataFrame.

    One row = one sale (order line item), with every dimension attribute
    (customer, product, category, region) attached as columns.

    Args:
        tables: Dict of table_name -> DataFrame, as produced by
                preprocessing.build_normalized_tables().

    Returns:
        A flat DataFrame ready for EDA, statistics, and forecasting.
    """
    df = tables["sales"].merge(tables["orders"], on="order_id", how="left")
    df = df.merge(tables["customers"], on="customer_id", how="left")
    df = df.merge(tables["locations"], on="location_id", how="left")
    df = df.merge(tables["regions"], on="region_id", how="left")
    df = df.merge(tables["products"], on="product_id", how="left")
    df = df.merge(tables["sub_categories"], on="subcategory_id", how="left")
    df = df.merge(tables["categories"], on="category_id", how="left")

    df = add_time_features(df)
    df = add_profit_margin(df)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar features derived from order_date: year, month, quarter,
    and a first-of-month timestamp (order_month) used for grouping trend
    charts without the noise of individual order dates.
    """
    df = df.copy()
    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    df["order_quarter"] = df["order_date"].dt.to_period("Q").astype(str)
    return df


def add_profit_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add profit_margin = profit / sales.

    Guarded against division by zero (a sales value of exactly 0 would
    otherwise produce inf) -- in that edge case margin is set to NaN rather
    than a misleading value, and is excluded from margin-based aggregates.
    """
    df = df.copy()
    df["profit_margin"] = (df["profit"] / df["sales"]).where(df["sales"] != 0)
    return df
