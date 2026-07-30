"""
sql_connector.py

Handles all communication with PostgreSQL: engine creation, running the
DDL from sql/schema.sql, and loading the normalized DataFrames produced by
preprocessing.py.

Credentials are read from environment variables rather than hardcoded, so
this module works unchanged across local dev, CI, and a deployed
environment (12-factor-app style config).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Tables must be loaded in this order -- every table after the first
# references at least one table before it via foreign key. Loading out of
# order would trigger FK constraint violations on insert.
TABLE_LOAD_ORDER: list[str] = [
    "regions",
    "categories",
    "locations",
    "sub_categories",
    "customers",
    "products",
    "orders",
    "sales",
]


def get_engine() -> Engine:
    """
    Build a SQLAlchemy engine from environment variables.

    Expected environment variables:
        DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

    Returns:
        A SQLAlchemy Engine connected to the target Postgres instance.
    """
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ.get("DB_NAME", "retail_analytics")
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "")

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    logger.info("Connecting to Postgres at %s:%s/%s", host, port, name)
    return create_engine(url)


def run_ddl(engine: Engine, ddl_path: str | Path) -> None:
    """
    Execute the DDL script (sql/schema.sql) to create all tables.

    Statements are split on ';' and run individually inside one transaction,
    so a mid-script failure rolls back cleanly rather than leaving a
    half-created schema behind.
    """
    ddl_path = Path(ddl_path)
    sql_text = ddl_path.read_text()
    statements = [s.strip() for s in sql_text.split(";") if s.strip() and not s.strip().startswith("--")]

    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))

    logger.info("Executed DDL from %s (%d statements)", ddl_path, len(statements))


def load_table(engine: Engine, df: pd.DataFrame, table_name: str) -> None:
    """
    Bulk-load a single DataFrame into its corresponding table using
    pandas.to_sql with method='multi' -- batches rows into multi-row INSERT
    statements rather than one INSERT per row, which is substantially
    faster for DataFrames in the tens-of-thousands-of-rows range.

    For very large datasets (millions of rows), the COPY-based approach in
    sql/insert_data.sql should be preferred over this method -- see that
    file's header comment for the throughput comparison.
    """
    df.to_sql(
        table_name,
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )
    logger.info("Loaded %d rows into %s", len(df), table_name)


def load_all_tables(engine: Engine, tables: dict[str, pd.DataFrame]) -> None:
    """
    Load every normalized table into Postgres in FK-safe dependency order.

    Args:
        engine: SQLAlchemy engine from get_engine().
        tables: Dict of table_name -> DataFrame, as produced by
                preprocessing.build_normalized_tables().
    """
    for table_name in TABLE_LOAD_ORDER:
        if table_name not in tables:
            raise KeyError(f"Expected table '{table_name}' not found in tables dict")
        load_table(engine, tables[table_name], table_name)

    logger.info("All %d tables loaded successfully", len(TABLE_LOAD_ORDER))
