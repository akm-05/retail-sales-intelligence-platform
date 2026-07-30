"""
test_dashboard_charts.py

Unit tests for src/dashboard_charts.py. Since these functions return
Plotly Figure objects rather than rendering anything, tests check
structural properties (a Figure was returned, it has the expected number
of traces, particular data made it into the chart) rather than pixels --
the same philosophy as test_statistics.py checking numbers rather than
"does this look right."
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.dashboard_charts import (
    plot_category_performance,
    plot_correlation_heatmap,
    plot_discount_vs_profit,
    plot_monthly_trend,
    plot_region_performance,
    plot_sales_distribution,
    plot_segment_share,
    plot_top_products,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "order_month": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"]),
        "category_name": ["Technology", "Furniture", "Technology", "Office Supplies"],
        "region_name": ["East", "West", "East", "South"],
        "segment": ["Consumer", "Corporate", "Consumer", "Home Office"],
        "product_name": ["Laptop", "Chair", "Phone", "Binder"],
        "sales": [1000.0, 200.0, 500.0, 50.0],
        "profit": [150.0, -20.0, 80.0, 10.0],
        "discount": [0.1, 0.3, 0.0, 0.2],
        "quantity": [2, 1, 3, 5],
        "profit_margin": [0.15, -0.10, 0.16, 0.20],
    })


class TestPlotMonthlyTrend:
    def test_returns_figure_with_two_traces(self, sample_df):
        fig = plot_monthly_trend(sample_df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # revenue line + profit line

    def test_aggregates_by_month(self, sample_df):
        fig = plot_monthly_trend(sample_df)
        revenue_trace = fig.data[0]
        assert len(revenue_trace.x) == 2  # two distinct months in sample_df


class TestPlotCategoryPerformance:
    def test_includes_all_categories(self, sample_df):
        fig = plot_category_performance(sample_df)
        assert set(fig.data[0].x) == {"Technology", "Furniture", "Office Supplies"}


class TestPlotRegionPerformance:
    def test_includes_all_regions(self, sample_df):
        fig = plot_region_performance(sample_df)
        assert set(fig.data[0].x) == {"East", "West", "South"}


class TestPlotTopProducts:
    def test_top_n_respects_n(self, sample_df):
        fig = plot_top_products(sample_df, n=2, metric="profit", ascending=False)
        assert len(fig.data[0].y) == 2

    def test_top_by_profit_orders_correctly(self, sample_df):
        fig = plot_top_products(sample_df, n=4, metric="profit", ascending=False)
        # highest profit (Laptop, 150) should be first
        assert fig.data[0].y[0] == "Laptop"

    def test_ascending_true_gives_worst_first(self, sample_df):
        fig = plot_top_products(sample_df, n=4, metric="profit", ascending=True)
        # lowest profit (Chair, -20) should be first
        assert fig.data[0].y[0] == "Chair"


class TestPlotDiscountVsProfit:
    def test_returns_scatter_with_all_points(self, sample_df):
        fig = plot_discount_vs_profit(sample_df)
        assert len(fig.data[0].x) == len(sample_df)


class TestPlotSegmentShare:
    def test_includes_all_segments(self, sample_df):
        fig = plot_segment_share(sample_df)
        assert set(fig.data[0].x) == {"Consumer", "Corporate", "Home Office"}


class TestPlotCorrelationHeatmap:
    def test_matrix_is_square_and_symmetric(self, sample_df):
        fig = plot_correlation_heatmap(sample_df)
        z = fig.data[0].z
        n = len(z)
        assert all(len(row) == n for row in z)
        for i in range(n):
            for j in range(n):
                assert z[i][j] == pytest.approx(z[j][i], abs=1e-9)


class TestPlotSalesDistribution:
    def test_without_outlier_mask_returns_single_trace(self, sample_df):
        fig = plot_sales_distribution(sample_df)
        assert len(fig.data) == 1

    def test_with_outlier_mask_splits_into_two_traces(self, sample_df):
        mask = pd.Series([True, False, False, False], index=sample_df.index)
        fig = plot_sales_distribution(sample_df, outlier_mask=mask)
        assert len(fig.data) == 2
