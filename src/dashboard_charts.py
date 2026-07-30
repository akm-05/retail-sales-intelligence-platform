"""
dashboard_charts.py

Interactive chart functions for the Streamlit dashboard, built with Plotly
rather than matplotlib.

Why two charting libraries in one project: visualization.py's matplotlib
charts are static report assets (Phase 4) -- generated once, saved to
outputs/images/, meant to be dropped into a slide deck or a written report.
This module's charts live inside a dashboard where the user is actively
filtering by date range, region, category, and segment through the sidebar
-- hovering for exact values and seeing the chart redraw on every filter
change is what a live tool is for, and Plotly is the library in this
project's declared tech stack built for exactly that. Using Plotly here and
matplotlib there isn't inconsistency -- it's matching the tool to whether
the chart is read once or explored interactively.

Same design contract as visualization.py: every function takes the (already
filtered, by the caller) flat analytics view and returns a Figure object.
This module never calls st.plotly_chart() or anything else
Streamlit-specific -- app.py owns displaying figures, this module only
builds them. That keeps these functions unit-testable without a running
Streamlit session (see test_dashboard_charts.py).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.statistics import compute_correlation_matrix

# Same palette as visualization.py, so a chart doesn't look like it wandered
# in from a different project depending on which tab you're looking at.
PRIMARY_COLOR = "#2C6E8E"
ACCENT_COLOR = "#D9822B"
NEGATIVE_COLOR = "#B23A48"
PALETTE = ["#2C6E8E", "#4C9F70", "#D9822B", "#8E6C8A", "#B23A48", "#6C757D"]


def _apply_layout_defaults(fig: go.Figure, title: str) -> go.Figure:
    """Shared layout tweaks so every chart in the dashboard reads as one
    coherent tool rather than mismatched default-themed plots -- the
    Plotly equivalent of visualization.set_plot_style()."""
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified",
    )
    return fig


def plot_monthly_trend(df: pd.DataFrame) -> go.Figure:
    """
    Business question: how does revenue AND profit trend together over
    time -- is profit keeping pace with revenue growth, or falling behind?

    Dual-line chart of monthly revenue and profit, with hover tooltips
    giving exact values -- the interactivity a static line chart can't
    offer.
    """
    monthly = df.groupby("order_month")[["sales", "profit"]].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["order_month"], y=monthly["sales"],
        name="Revenue", mode="lines+markers", line=dict(color=PRIMARY_COLOR),
    ))
    fig.add_trace(go.Scatter(
        x=monthly["order_month"], y=monthly["profit"],
        name="Profit", mode="lines+markers", line=dict(color=ACCENT_COLOR),
    ))
    fig.update_yaxes(title_text="USD")
    return _apply_layout_defaults(fig, "Monthly revenue and profit")


def plot_category_performance(df: pd.DataFrame) -> go.Figure:
    """
    Business question: which categories generate the most revenue, and are
    the highest-revenue categories also the most profitable?

    Grouped bar chart, revenue vs. profit per category.
    """
    agg = df.groupby("category_name")[["sales", "profit"]].sum().sort_values("sales", ascending=False).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["category_name"], y=agg["sales"], name="Revenue", marker_color=PRIMARY_COLOR))
    fig.add_trace(go.Bar(x=agg["category_name"], y=agg["profit"], name="Profit", marker_color=ACCENT_COLOR))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="USD")
    return _apply_layout_defaults(fig, "Revenue vs. profit by category")


def plot_region_performance(df: pd.DataFrame) -> go.Figure:
    """
    Business question: which regions contribute the most revenue, and do
    any regions contribute disproportionately little profit relative to
    their revenue share?

    Grouped bar chart, revenue vs. profit per region.
    """
    agg = df.groupby("region_name")[["sales", "profit"]].sum().sort_values("sales", ascending=False).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["region_name"], y=agg["sales"], name="Revenue", marker_color=PRIMARY_COLOR))
    fig.add_trace(go.Bar(x=agg["region_name"], y=agg["profit"], name="Profit", marker_color=ACCENT_COLOR))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="USD")
    return _apply_layout_defaults(fig, "Revenue vs. profit by region")


def plot_top_products(df: pd.DataFrame, n: int = 10, metric: str = "profit", ascending: bool = False) -> go.Figure:
    """
    Business question: which specific products are the biggest profit (or
    revenue) drivers -- or, with ascending=True, which are actively losing
    money and worth re-pricing or discontinuing?

    Horizontal bar chart of the top/bottom N products by the chosen metric.
    Exposed as one function with a direction flag (rather than two
    functions, as visualization.py has) because in the dashboard this is
    driven by a single UI toggle -- one function keeps that wiring simple.

    Args:
        df: Flat analytics view (already filtered by the caller).
        n: Number of products to show.
        metric: Column to rank by, typically 'profit' or 'sales'.
        ascending: False for "top" (highest first), True for "worst"
                   (lowest/most negative first).
    """
    agg = df.groupby("product_name")[metric].sum().sort_values(ascending=ascending).head(n).reset_index()
    color = NEGATIVE_COLOR if ascending else PRIMARY_COLOR
    label = "Bottom" if ascending else "Top"

    fig = go.Figure(go.Bar(
        x=agg[metric], y=agg["product_name"], orientation="h", marker_color=color,
    ))
    fig.update_yaxes(autorange="reversed")  # largest value at the top of the chart
    fig.update_xaxes(title_text=f"Total {metric} (USD)")
    return _apply_layout_defaults(fig, f"{label} {n} products by {metric}")


def plot_discount_vs_profit(df: pd.DataFrame) -> go.Figure:
    """
    Business question: at what discount level does profit start turning
    negative -- i.e. is there a defensible maximum discount threshold?

    Scatter of discount vs. profit margin, one point per sale, with hover
    showing the product and category behind each point.
    """
    fig = px.scatter(
        df, x="discount", y="profit_margin",
        hover_data=["product_name", "category_name"],
        opacity=0.4,
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.add_hline(y=0, line_dash="dash", line_color=NEGATIVE_COLOR)
    fig.update_xaxes(title_text="Discount")
    fig.update_yaxes(title_text="Profit margin")
    return _apply_layout_defaults(fig, "Discount vs. profit margin")


def plot_segment_share(df: pd.DataFrame) -> go.Figure:
    """
    Business question: which customer segment drives the most revenue --
    where should account management/retention effort be concentrated?

    Bar chart of revenue by segment -- kept as a bar chart rather than a
    donut, for the same reason as visualization.py's segment chart: with
    only three segments, a bar chart makes the magnitude difference easier
    to read precisely than an angular comparison would.
    """
    agg = df.groupby("segment")["sales"].sum().sort_values(ascending=False).reset_index()

    fig = go.Figure(go.Bar(x=agg["segment"], y=agg["sales"], marker_color=PALETTE[: len(agg)]))
    fig.update_yaxes(title_text="Revenue (USD)")
    return _apply_layout_defaults(fig, "Revenue by customer segment")


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Business question: which numeric business metrics move together, and
    is discount actually correlated with profit loss (vs. just perceived
    to be)?

    Heatmap of the correlation matrix, reusing
    statistics.compute_correlation_matrix() rather than recomputing
    .corr() directly -- the one place that calculation happens, same as
    every other cross-module reuse in this project.
    """
    numeric_cols = ["sales", "quantity", "discount", "profit", "profit_margin"]
    corr = compute_correlation_matrix(df, numeric_cols)

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr.round(2).values, texttemplate="%{text}",
    ))
    return _apply_layout_defaults(fig, "Correlation matrix")


def plot_sales_distribution(df: pd.DataFrame, outlier_mask: pd.Series | None = None) -> go.Figure:
    """
    Business question: what does a "typical" sale look like, and how many
    extreme outliers are skewing average-based metrics?

    Histogram of sale amounts. When outlier_mask is supplied (from
    statistics.detect_outliers_iqr() or detect_outliers_zscore()), outlier
    and non-outlier sales are drawn as two overlaid, differently-colored
    histograms, so the dashboard visibly ties the statistics tab's outlier
    detector to what it's flagging -- rather than just reporting a count.
    """
    if outlier_mask is None:
        fig = px.histogram(df, x="sales", nbins=40, color_discrete_sequence=[PRIMARY_COLOR])
        return _apply_layout_defaults(fig, "Distribution of sale amounts")

    labeled = df.assign(is_outlier=outlier_mask.map({True: "Outlier", False: "Typical"}))
    fig = px.histogram(
        labeled, x="sales", color="is_outlier", nbins=40, barmode="overlay",
        color_discrete_map={"Typical": PRIMARY_COLOR, "Outlier": NEGATIVE_COLOR},
        opacity=0.7,
    )
    return _apply_layout_defaults(fig, "Distribution of sale amounts (outliers highlighted)")
