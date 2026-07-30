"""
visualization.py

Reusable chart functions for EDA. Every function takes the flat analytics
view from feature_engineering.build_analytics_view() and answers ONE
specific business question -- that mapping (function -> business question)
is documented in each docstring, and is the thing to walk an interviewer
through, not the matplotlib mechanics.

Design choices:
- matplotlib only (not seaborn) to match the project's declared tech stack
  and keep the dependency list minimal.
- Every function saves its figure to disk and also returns the Figure
  object, so it can be used both in a script (save to images/) and
  interactively in a notebook (display inline).
- A single set_plot_style() call establishes a consistent look across every
  chart, so the EDA reads as one coherent report rather than mismatched
  default-styled plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Consistent palette used across every chart -- avoids matplotlib's default
# rainbow cycling, which reads as arbitrary rather than intentional.
PRIMARY_COLOR = "#2C6E8E"
ACCENT_COLOR = "#D9822B"
NEGATIVE_COLOR = "#B23A48"
PALETTE = ["#2C6E8E", "#4C9F70", "#D9822B", "#8E6C8A", "#B23A48", "#6C757D"]


def set_plot_style() -> None:
    """Apply a consistent, presentation-ready style to every chart."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.dpi": 110,
    })


def _save(fig: plt.Figure, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")


def plot_revenue_profit_by_category(df: pd.DataFrame, save_path: str | Path) -> plt.Figure:
    """
    Business question: Which categories generate the most revenue, and are
    the highest-revenue categories also the most profitable?

    Grouped bar chart: revenue vs. profit per category, side by side.
    """
    agg = df.groupby("category_name")[["sales", "profit"]].sum().sort_values("sales", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(agg))
    width = 0.35
    ax.bar([i - width / 2 for i in x], agg["sales"], width, label="Revenue", color=PRIMARY_COLOR)
    ax.bar([i + width / 2 for i in x], agg["profit"], width, label="Profit", color=ACCENT_COLOR)
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg.index, rotation=0)
    ax.set_ylabel("USD")
    ax.set_title("Revenue vs. profit by category")
    ax.legend()

    _save(fig, save_path)
    return fig


def plot_monthly_sales_trend(df: pd.DataFrame, save_path: str | Path) -> plt.Figure:
    """
    Business question: How does revenue trend over time, and is there
    seasonality worth planning inventory/staffing around?

    Line chart of total monthly revenue.
    """
    monthly = df.groupby("order_month")["sales"].sum()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly.index, monthly.values, color=PRIMARY_COLOR, marker="o", markersize=3)
    ax.set_title("Monthly revenue trend")
    ax.set_ylabel("Revenue (USD)")
    ax.set_xlabel("Month")
    fig.autofmt_xdate()

    _save(fig, save_path)
    return fig


def plot_region_performance(df: pd.DataFrame, save_path: str | Path) -> plt.Figure:
    """
    Business question: Which regions contribute the most revenue, and do
    any regions contribute disproportionately little profit relative to
    their revenue share?

    Grouped bar chart: revenue vs. profit per region.
    """
    agg = df.groupby("region_name")[["sales", "profit"]].sum().sort_values("sales", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(agg))
    width = 0.35
    ax.bar([i - width / 2 for i in x], agg["sales"], width, label="Revenue", color=PRIMARY_COLOR)
    ax.bar([i + width / 2 for i in x], agg["profit"], width, label="Profit", color=ACCENT_COLOR)
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg.index)
    ax.set_ylabel("USD")
    ax.set_title("Revenue vs. profit by region")
    ax.legend()

    _save(fig, save_path)
    return fig


def plot_top_products(df: pd.DataFrame, save_path: str | Path, n: int = 10) -> plt.Figure:
    """
    Business question: Which specific products are the biggest profit
    drivers -- the candidates for guaranteed stock availability?

    Horizontal bar chart of the top N products by total profit.
    """
    agg = df.groupby("product_name")["profit"].sum().sort_values(ascending=False).head(n)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(agg.index[::-1], agg.values[::-1], color=PRIMARY_COLOR)
    ax.set_title(f"Top {n} products by profit")
    ax.set_xlabel("Total profit (USD)")

    _save(fig, save_path)
    return fig


def plot_worst_products(df: pd.DataFrame, save_path: str | Path, n: int = 10) -> plt.Figure:
    """
    Business question: Which products are actively losing money and should
    be re-priced, discounted less aggressively, or discontinued?

    Horizontal bar chart of the bottom N products by total profit.
    """
    agg = df.groupby("product_name")["profit"].sum().sort_values(ascending=True).head(n)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(agg.index[::-1], agg.values[::-1], color=NEGATIVE_COLOR)
    ax.set_title(f"Bottom {n} products by profit")
    ax.set_xlabel("Total profit (USD)")

    _save(fig, save_path)
    return fig


def plot_discount_vs_profit(df: pd.DataFrame, save_path: str | Path) -> plt.Figure:
    """
    Business question: At what discount level does profit start turning
    negative -- i.e. is there a defensible maximum discount threshold?

    Scatter plot of discount vs. profit margin, one point per sale.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df["discount"], df["profit_margin"], alpha=0.3, s=12, color=PRIMARY_COLOR)
    ax.axhline(0, color=NEGATIVE_COLOR, linewidth=1, linestyle="--")
    ax.set_title("Discount vs. profit margin")
    ax.set_xlabel("Discount")
    ax.set_ylabel("Profit margin")

    _save(fig, save_path)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str | Path) -> plt.Figure:
    """
    Business question: Which numeric business metrics move together, and
    is discount actually correlated with profit loss (vs. just perceived
    to be)?

    Heatmap of the correlation matrix across sales, quantity, discount,
    profit, and profit_margin.
    """
    numeric_cols = ["sales", "quantity", "discount", "profit", "profit_margin"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Correlation matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)

    _save(fig, save_path)
    return fig


def plot_customer_segment_contribution(df: pd.DataFrame, save_path: str | Path) -> plt.Figure:
    """
    Business question: Which customer segment drives the most revenue --
    where should account management/retention effort be concentrated?

    Bar chart of revenue by customer segment (chosen over a pie chart:
    with only 3 segments, a bar chart makes the magnitude difference
    easier to read precisely than angular comparison would).
    """
    agg = df.groupby("segment")["sales"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(agg.index, agg.values, color=PALETTE[: len(agg)])
    ax.set_title("Revenue by customer segment")
    ax.set_ylabel("Revenue (USD)")

    _save(fig, save_path)
    return fig


def plot_sales_distribution(df: pd.DataFrame, save_path: str | Path) -> plt.Figure:
    """
    Business question: What does a "typical" sale look like, and how many
    extreme outliers (bulk orders, heavily discounted clearance sales) are
    skewing average-based metrics?

    Histogram of sale amounts with a boxplot beneath it for outlier
    visibility.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 5.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax1.hist(df["sales"], bins=40, color=PRIMARY_COLOR, edgecolor="white")
    ax1.set_title("Distribution of sale amounts")
    ax1.set_ylabel("Number of sales")

    ax2.boxplot(df["sales"], vert=False, widths=0.6)
    ax2.set_xlabel("Sale amount (USD)")
    ax2.set_yticks([])

    _save(fig, save_path)
    return fig
