"""
dashboard_data.py

Phase 7's data-access layer: gets the same flat analytics view every other
module in this project uses (feature_engineering.build_analytics_view())
into the Streamlit app.

Deliberately has no `import streamlit` anywhere in this file. Keeping the
data-loading logic framework-agnostic means it can be unit-tested the same
way as every other module in src/ (see test_dashboard_data.py), instead of
requiring a running Streamlit session to exercise it. app.py is responsible
for the Streamlit-specific parts (caching with @st.cache_data, displaying
errors with st.error) -- this module just returns DataFrames or raises.

Two data sources, in preference order:
  1. Postgres, via sql_connector.get_engine() -- the "real" path, matching
     the rest of this project's architecture (CSV -> Postgres -> analytics).
  2. The processed CSVs in data/processed/, written by run_pipeline.py --
     a fallback so the dashboard is still demoable (e.g. in an interview,
     on a laptop with no Postgres instance running) without changing a
     single line of code. This mirrors the project philosophy that "every
     module should be runnable."
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.feature_engineering import build_analytics_view
from src.sql_connector import TABLE_LOAD_ORDER, get_engine

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = Path("data/processed")

# Tables whose CSVs contain columns that must be parsed back to datetime --
# read_csv leaves them as strings otherwise, and every downstream module
# (feature_engineering, forecasting) expects real Timestamps.
_DATE_COLUMNS_BY_TABLE: dict[str, list[str]] = {
    "orders": ["order_date", "ship_date"],
}


def load_tables_from_postgres() -> dict[str, pd.DataFrame]:
    """
    Read all eight normalized tables directly from Postgres.

    Returns:
        Dict of table_name -> DataFrame, keyed exactly like
        preprocessing.build_normalized_tables()'s output, so it can be
        passed straight into feature_engineering.build_analytics_view().

    Raises:
        Exception: Whatever SQLAlchemy/psycopg2 raises if the connection
                   fails (e.g. Postgres isn't running) -- deliberately not
                   caught here. load_analytics_view() decides how to react
                   to that failure; this function's job is just to try.
    """
    engine = get_engine()
    tables = {
        table_name: pd.read_sql(f"SELECT * FROM {table_name}", con=engine)
        for table_name in TABLE_LOAD_ORDER
    }
    for table_name, date_cols in _DATE_COLUMNS_BY_TABLE.items():
        for col in date_cols:
            tables[table_name][col] = pd.to_datetime(tables[table_name][col])
    logger.info("Loaded %d tables from Postgres", len(tables))
    return tables


def load_tables_from_processed_csv(processed_dir: str | Path = DEFAULT_PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    """
    Read all eight normalized tables from the processed CSVs written by
    run_pipeline.py.

    Args:
        processed_dir: Directory containing <table_name>.csv for each of
                        the eight tables in TABLE_LOAD_ORDER.

    Returns:
        Dict of table_name -> DataFrame, same shape as
        load_tables_from_postgres().

    Raises:
        FileNotFoundError: If any expected CSV is missing -- most likely
                            meaning run_pipeline.py hasn't been run yet.
    """
    processed_dir = Path(processed_dir)
    tables: dict[str, pd.DataFrame] = {}

    for table_name in TABLE_LOAD_ORDER:
        csv_path = processed_dir / f"{table_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Processed table not found: {csv_path}. "
                "Run `python -m src.run_pipeline` first, or start Postgres "
                "so the dashboard can load from there instead."
            )
        date_cols = _DATE_COLUMNS_BY_TABLE.get(table_name)
        tables[table_name] = pd.read_csv(csv_path, parse_dates=date_cols)

    logger.info("Loaded %d tables from %s", len(tables), processed_dir)
    return tables


def load_analytics_view(
    prefer_postgres: bool = True, processed_dir: str | Path = DEFAULT_PROCESSED_DIR
) -> pd.DataFrame:
    """
    Load the eight normalized tables from the best available source and
    join them into the flat analytics view -- the single entry point
    app.py needs to get from "no data" to "ready to chart."

    Args:
        prefer_postgres: If True (default), try Postgres first and fall
                          back to the processed CSVs only if that fails.
                          If False, read the CSVs directly.
        processed_dir: Passed through to load_tables_from_processed_csv().

    Returns:
        Flat analytics view from feature_engineering.build_analytics_view().

    Raises:
        FileNotFoundError: If both the Postgres connection and the CSV
                            fallback fail (i.e. there's genuinely no data
                            available anywhere).
    """
    if prefer_postgres:
        try:
            tables = load_tables_from_postgres()
            return build_analytics_view(tables)
        except Exception as exc:  # noqa: BLE001 -- intentionally broad: any
            # connection failure (down, unreachable, wrong credentials)
            # should fall back the same way, not just OperationalError.
            logger.warning("Postgres unavailable (%s) -- falling back to processed CSVs", exc)

    tables = load_tables_from_processed_csv(processed_dir)
    return build_analytics_view(tables)


def compute_kpis(df: pd.DataFrame) -> dict:
    """
    The five headline numbers for the dashboard's KPI row.

    Kept here rather than inline in app.py so the calculation is testable
    without a running Streamlit session -- app.py's job is only to format
    and display these values, not compute them.

    Args:
        df: Flat analytics view (already filtered by the caller).

    Returns:
        Dict with: total_revenue, total_profit, profit_margin (0-1 scale,
        NaN if revenue is 0), total_orders, avg_order_value (NaN if there
        are no orders).
    """
    total_revenue = df["sales"].sum()
    total_profit = df["profit"].sum()
    total_orders = df["order_id"].nunique()

    return {
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "profit_margin": (total_profit / total_revenue) if total_revenue else float("nan"),
        "total_orders": total_orders,
        "avg_order_value": (total_revenue / total_orders) if total_orders else float("nan"),
    }
