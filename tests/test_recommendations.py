"""
test_recommendations.py

Unit tests for src/recommendations.py. Each rule gets a small synthetic
DataFrame with a known pattern baked in (e.g. one category has above-
average revenue but below-average margin) so the test can assert the exact
recommendation fires -- and, just as importantly, that it does NOT fire
when the data doesn't support it.
"""

from __future__ import annotations

import pandas as pd

from src.recommendations import (
    generate_recommendations,
    recommend_category_focus,
    recommend_discount_threshold,
    recommend_forecast_trend,
    recommend_region_focus,
    recommend_segment_focus,
)


class TestRecommendCategoryFocus:
    def test_flags_high_revenue_low_margin_category(self):
        df = pd.DataFrame({
            "category_name": ["Technology"] * 4 + ["Furniture"] * 2,
            "sales": [1000, 1000, 1000, 1000, 100, 100],
            "profit": [50, 50, 50, 50, 40, 40],  # Technology margin 5%, Furniture margin 40%
        })
        result = recommend_category_focus(df)
        assert result is not None
        assert "Technology" in result["title"]

    def test_returns_none_when_top_category_already_above_average_margin(self):
        df = pd.DataFrame({
            "category_name": ["Technology"] * 2 + ["Furniture"] * 2,
            "sales": [1000, 1000, 100, 100],
            "profit": [400, 400, 5, 5],  # Technology margin 40%, well above overall
        })
        result = recommend_category_focus(df)
        assert result is None


class TestRecommendRegionFocus:
    def test_flags_low_margin_material_region(self):
        df = pd.DataFrame({
            "region_name": ["West"] * 3 + ["East"] * 3,
            "sales": [500, 500, 500, 500, 500, 500],
            "profit": [10, 10, 10, 200, 200, 200],  # West margin ~2%, East ~40%
        })
        result = recommend_region_focus(df)
        assert result is not None
        assert "West" in result["title"]

    def test_ignores_immaterial_small_regions(self):
        # "Tiny" region has terrible margin but only ~1% of revenue -- should be ignored.
        df = pd.DataFrame({
            "region_name": ["Big"] * 3 + ["Tiny"],
            "sales": [1000, 1000, 1000, 10],
            "profit": [300, 300, 300, -50],
        })
        result = recommend_region_focus(df)
        assert result is None or "Tiny" not in result["title"]


class TestRecommendDiscountThreshold:
    def test_finds_threshold_where_margin_turns_negative(self):
        df = pd.DataFrame({
            "discount": [0.0, 0.1, 0.2, 0.3, 0.4],
            "profit_margin": [0.30, 0.20, 0.10, -0.05, -0.20],
        })
        result = recommend_discount_threshold(df, bucket_width=0.1)
        assert result is not None
        assert "0.3" in result["detail"] or "30" in result["detail"]

    def test_returns_none_when_margin_never_negative(self):
        df = pd.DataFrame({
            "discount": [0.0, 0.1, 0.2],
            "profit_margin": [0.30, 0.25, 0.20],
        })
        result = recommend_discount_threshold(df)
        assert result is None


class TestRecommendSegmentFocus:
    def test_flags_top_segment_by_revenue(self):
        df = pd.DataFrame({
            "segment": ["Consumer", "Consumer", "Corporate"],
            "sales": [800, 800, 200],
        })
        result = recommend_segment_focus(df)
        assert result is not None
        assert "Consumer" in result["title"]


class TestRecommendForecastTrend:
    def test_flags_growth(self):
        monthly_sales = pd.Series([1000.0] * 12)
        future_forecast = pd.Series([1300.0, 1300.0, 1300.0])
        result = recommend_forecast_trend(future_forecast, monthly_sales)
        assert result is not None
        assert "growth" in result["title"].lower()

    def test_flags_decline(self):
        monthly_sales = pd.Series([1000.0] * 12)
        future_forecast = pd.Series([700.0, 700.0, 700.0])
        result = recommend_forecast_trend(future_forecast, monthly_sales)
        assert "decline" in result["title"].lower()

    def test_returns_none_when_forecast_missing(self):
        assert recommend_forecast_trend(None, None) is None


class TestGenerateRecommendations:
    def test_returns_only_non_none_results(self):
        df = pd.DataFrame({
            "category_name": ["Technology"] * 2,
            "region_name": ["East"] * 2,
            "segment": ["Consumer"] * 2,
            "sales": [500.0, 500.0],
            "profit": [100.0, 100.0],
            "discount": [0.1, 0.1],
            "profit_margin": [0.20, 0.20],
        })
        recommendations = generate_recommendations(df)
        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert "title" in rec and "detail" in rec
