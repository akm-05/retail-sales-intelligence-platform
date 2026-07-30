"""
run_pipeline.py

Orchestrates the full preprocessing pipeline end-to-end:

    raw CSV -> load -> clean -> normalize -> write processed CSVs -> load to Postgres

This is the single entry point a scheduler (cron, Airflow, etc.) would call
in a production setting. Each stage is logged so a failure is traceable to
the exact stage it occurred in.

Usage:
    python -m src.run_pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.data_loader import load_raw_data
from src.preprocessing import build_normalized_tables, clean_raw_data
from src.sql_connector import get_engine, load_all_tables, run_ddl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw/superstore.csv")
PROCESSED_DATA_DIR = Path("data/processed")
SCHEMA_PATH = Path("sql/schema.sql")


def main() -> None:
    logger.info("=== Retail Sales Intelligence Platform: preprocessing pipeline ===")

    # 1. Load
    raw_df = load_raw_data(RAW_DATA_PATH)

    # 2. Clean
    clean_df = clean_raw_data(raw_df)

    # 3. Normalize (also runs referential integrity validation internally)
    tables = build_normalized_tables(clean_df)

    # 4. Persist processed CSVs -- these are what sql/insert_data.sql's
    #    COPY commands read from.
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for table_name, table_df in tables.items():
        out_path = PROCESSED_DATA_DIR / f"{table_name}.csv"
        table_df.to_csv(out_path, index=False)
        logger.info("Wrote %s (%d rows)", out_path, len(table_df))

    # 5. Load into Postgres
    engine = get_engine()
    run_ddl(engine, SCHEMA_PATH)
    load_all_tables(engine, tables)

    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
