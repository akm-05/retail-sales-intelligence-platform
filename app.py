"""
app.py

Phase 7 of the pipeline: an interactive Streamlit dashboard for a business
manager audience, sitting on top of the exact same analytics view every
other module in this project uses.

Placed at the project root (rather than inside src/) so `streamlit run
app.py` works with zero configuration, both locally and on Streamlit
Community Cloud, which expects the entry point at the repo root. It still
imports everything from src., the same as the test suite does, since the
project root is on sys.path when Streamlit runs this file directly.

Design note on this file's role: app.py is intentionally thin. Every
number on this page is computed by a function in src/ that already has its
own unit tests (dashboard_data.compute_kpis, statistics.*,
forecasting.generate_sales_forecast, recommendations.generate_recommendations,
the dashboard_charts.* chart builders) -- this file's only job is UI
wiring: read the sidebar filters, call the right function, display what
it returns. Keeping business logic out of app.py is what makes the rest
of this project's test suite actually cover the dashboard's numbers too.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

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
from src.dashboard_data import compute_kpis, load_analytics_view
from src.forecasting import generate_sales_forecast, plot_decomposition, plot_forecast
from src.recommendations import generate_recommendations
from src.statistics import (
    compare_two_groups_ttest,
    compute_descriptive_stats,
    confidence_interval_mean,
    detect_outliers_iqr,
    detect_outliers_zscore,
)
from src.visualization import set_plot_style

st.set_page_config(
    page_title="Retail Sales Intelligence & Demand Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)
set_plot_style()  # applies to the matplotlib figures reused from forecasting.py


# ----------------------------------------------------------------------------
# DATA LOADING (cached -- reloading from Postgres/CSV on every filter click
# would make the sidebar feel sluggish for no benefit, since the underlying
# data doesn't change within a session)
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading data...")
def get_analytics_view() -> pd.DataFrame:
    return load_analytics_view()


try:
    full_df = get_analytics_view()
except FileNotFoundError as exc:
    st.error(
        "No data available. Start Postgres and run `python -m src.run_pipeline`, "
        f"or make sure data/processed/ contains the pipeline's output.\n\n{exc}"
    )
    st.stop()


# ----------------------------------------------------------------------------
# SIDEBAR: FILTERS + FORECAST SETTINGS
# ----------------------------------------------------------------------------

st.sidebar.title("Filters")

min_date, max_date = full_df["order_date"].min().date(), full_df["order_date"].max().date()
date_range = st.sidebar.date_input(
    "Order date range", value=(min_date, max_date), min_value=min_date, max_value=max_date,
)

region_options = sorted(full_df["region_name"].dropna().unique())
selected_regions = st.sidebar.multiselect("Region", region_options, default=region_options)

category_options = sorted(full_df["category_name"].dropna().unique())
selected_categories = st.sidebar.multiselect("Category", category_options, default=category_options)

segment_options = sorted(full_df["segment"].dropna().unique())
selected_segments = st.sidebar.multiselect("Segment", segment_options, default=segment_options)

with st.sidebar.expander("Forecast settings"):
    forecast_periods = st.slider("Months to forecast", min_value=1, max_value=12, value=6)
    test_size = st.slider("Backtest holdout (months)", min_value=3, max_value=12, value=6)

# A single-day date_input selection returns a length-1 tuple -- guard against
# it so the filter below doesn't crash mid-selection while the user is still
# picking the second date.
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered_df = full_df[
    (full_df["order_date"].dt.date >= start_date)
    & (full_df["order_date"].dt.date <= end_date)
    & (full_df["region_name"].isin(selected_regions))
    & (full_df["category_name"].isin(selected_categories))
    & (full_df["segment"].isin(selected_segments))
]

if filtered_df.empty:
    st.warning("No data matches the current filters. Widen the date range or selections in the sidebar.")
    st.stop()


# ----------------------------------------------------------------------------
# HEADER + KPI ROW
# ----------------------------------------------------------------------------

st.title("Retail Sales Intelligence & Demand Forecasting")
st.caption(
    f"{len(filtered_df):,} sales records | "
    f"{start_date:%b %Y} - {end_date:%b %Y} | "
    f"{len(selected_regions)} region(s), {len(selected_categories)} categor(y/ies), "
    f"{len(selected_segments)} segment(s) selected"
)

kpis = compute_kpis(filtered_df)
kpi_cols = st.columns(5)
kpi_cols[0].metric("Total Revenue", f"${kpis['total_revenue']:,.0f}")
kpi_cols[1].metric("Total Profit", f"${kpis['total_profit']:,.0f}")
kpi_cols[2].metric("Profit Margin", f"{kpis['profit_margin'] * 100:.1f}%")
kpi_cols[3].metric("Total Orders", f"{kpis['total_orders']:,}")
kpi_cols[4].metric("Avg Order Value", f"${kpis['avg_order_value']:,.2f}")

st.divider()


# ----------------------------------------------------------------------------
# FORECAST (computed once, shared by the Demand Forecast and
# Recommendations tabs below, rather than recomputed in each)
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner="Generating forecast...")
def get_forecast(df: pd.DataFrame, periods: int, holdout: int) -> dict:
    return generate_sales_forecast(df, forecast_periods=periods, test_size=holdout)


forecast_result, forecast_error = None, None
try:
    forecast_result = get_forecast(filtered_df, forecast_periods, test_size)
except ValueError as exc:
    forecast_error = str(exc)


# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------

tab_overview, tab_sales_profit, tab_stats, tab_forecast, tab_recommendations, tab_explorer = st.tabs(
    ["Overview", "Sales & Profit", "Statistical Insights", "Demand Forecast", "Recommendations", "Data Explorer"]
)

with tab_overview:
    st.plotly_chart(plot_monthly_trend(filtered_df), width="stretch")
    col_a, col_b = st.columns(2)
    col_a.plotly_chart(plot_category_performance(filtered_df), width="stretch")
    col_b.plotly_chart(plot_region_performance(filtered_df), width="stretch")

with tab_sales_profit:
    st.plotly_chart(plot_segment_share(filtered_df), width="stretch")

    st.subheader("Top / bottom products")
    control_cols = st.columns(3)
    metric_choice = control_cols[0].radio("Rank by", ["profit", "sales"], horizontal=True)
    direction_choice = control_cols[1].radio("Direction", ["Top", "Bottom"], horizontal=True)
    n_products = control_cols[2].slider("Number of products", min_value=5, max_value=25, value=10)
    st.plotly_chart(
        plot_top_products(filtered_df, n=n_products, metric=metric_choice, ascending=(direction_choice == "Bottom")),
        width="stretch",
    )

    st.plotly_chart(plot_discount_vs_profit(filtered_df), width="stretch")

with tab_stats:
    st.subheader("Descriptive statistics")
    numeric_columns = ["sales", "profit", "discount", "quantity", "profit_margin"]
    stats_col = st.selectbox("Column", numeric_columns, key="stats_col")
    stats_result = compute_descriptive_stats(filtered_df[stats_col])
    st.dataframe(pd.DataFrame([stats_result]).T.rename(columns={0: stats_col}), width="stretch")

    confidence = st.slider("Confidence level", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
    lower, upper = confidence_interval_mean(filtered_df[stats_col], confidence=confidence)
    st.write(
        f"{int(confidence * 100)}% confidence interval for the mean of **{stats_col}**: "
        f"[{lower:,.4f}, {upper:,.4f}]"
    )

    st.subheader("Outlier detection")
    outlier_method = st.radio("Method", ["IQR", "Z-score"], horizontal=True)
    outlier_mask = (
        detect_outliers_iqr(filtered_df["sales"])
        if outlier_method == "IQR"
        else detect_outliers_zscore(filtered_df["sales"])
    )
    st.write(f"{int(outlier_mask.sum())} outlier sale(s) flagged out of {len(filtered_df):,} ({outlier_mask.mean() * 100:.1f}%).")
    st.plotly_chart(plot_sales_distribution(filtered_df, outlier_mask), width="stretch")

    st.subheader("Compare two groups")
    group_dimension = st.selectbox("Group by", ["region_name", "category_name", "segment"])
    group_values = sorted(filtered_df[group_dimension].dropna().unique())
    if len(group_values) >= 2:
        compare_cols = st.columns(3)
        group_a_label = compare_cols[0].selectbox("Group A", group_values, index=0)
        group_b_label = compare_cols[1].selectbox("Group B", group_values, index=1)
        compare_metric = compare_cols[2].selectbox("Metric", numeric_columns, index=4)
        ttest_result = compare_two_groups_ttest(
            filtered_df.loc[filtered_df[group_dimension] == group_a_label, compare_metric],
            filtered_df.loc[filtered_df[group_dimension] == group_b_label, compare_metric],
        )
        st.write(ttest_result["interpretation"])
    else:
        st.info("Need at least two distinct values in the current filter to compare groups.")

    st.subheader("Correlation matrix")
    st.plotly_chart(plot_correlation_heatmap(filtered_df), width="stretch")

with tab_forecast:
    if forecast_error is not None:
        st.warning(
            f"Can't generate a forecast for the current filters: {forecast_error} "
            "Try widening the date range, or reducing the backtest holdout in the sidebar."
        )
    else:
        st.subheader("Backtest: forecast method comparison")
        backtest_df = pd.DataFrame(forecast_result["backtest_evaluation"]).T
        st.dataframe(backtest_df.style.format({"mae": "{:,.1f}", "rmse": "{:,.1f}", "mape": "{:.2f}%"}))
        st.caption(
            "Linear regression is compared against two naive baselines (moving average, exponential "
            "smoothing) on the same held-out months. Lower MAE/RMSE/MAPE is better."
        )

        st.subheader("Trend and seasonality")
        st.pyplot(plot_decomposition(forecast_result["decomposition"], save_path="/tmp/decomposition.png"))

        st.subheader(f"{forecast_periods}-month forecast")
        st.pyplot(plot_forecast(
            forecast_result["monthly_sales"], forecast_result["future_forecast"],
            save_path="/tmp/forecast.png", future_forecast_ci=forecast_result["future_forecast_ci"],
        ))

        lower_ci, upper_ci = forecast_result["future_forecast_ci"]
        forecast_table = pd.DataFrame({
            "forecast": forecast_result["future_forecast"],
            "lower_95": lower_ci,
            "upper_95": upper_ci,
        })
        st.dataframe(forecast_table.style.format("${:,.0f}"), width="stretch")

with tab_recommendations:
    st.subheader("Business recommendations")
    st.caption("Generated from the currently filtered data -- adjust the sidebar to see how these change.")

    forecast_series = forecast_result["future_forecast"] if forecast_result else None
    monthly_series = forecast_result["monthly_sales"] if forecast_result else None
    recommendations = generate_recommendations(filtered_df, forecast_series, monthly_series)

    if not recommendations:
        st.info("No specific flags for the current filters -- the data doesn't show a clear signal to act on.")
    else:
        for rec in recommendations:
            with st.container(border=True):
                st.markdown(f"**{rec['title']}**")
                st.write(rec["detail"])

with tab_explorer:
    st.subheader("Filtered data")
    st.dataframe(filtered_df, width="stretch")
    st.download_button(
        "Download filtered data as CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_sales_data.csv",
        mime="text/csv",
    )
