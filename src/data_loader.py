"""
data_loader.py

Responsible for exactly one thing: getting the raw Superstore CSV into a
pandas DataFrame safely, with column names standardized and the expected
schema validated before any downstream module touches it.

Keeping "load" and "clean" as separate modules (data_loader.py vs.
preprocessing.py) means a schema-validation failure is caught immediately,
before wasting time running cleaning logic on data that isn't what we think
it is.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# The exact columns we expect in the raw Superstore export, and the dtype
# we want pandas to infer them as on read. Declaring this explicitly (rather
# than letting pandas guess) avoids silent type-inference bugs -- e.g. a
# Postal Code column being read as float64 and rendering "10001" as
# "10001.0".
EXPECTED_RAW_COLUMNS: dict[str, str] = {
    "Row ID": "int64",
    "Order ID": "string",
    "Order Date": "string",   # parsed to datetime in preprocessing.py
    "Ship Date": "string",
    "Ship Mode": "string",
    "Customer ID": "string",
    "Customer Name": "string",
    "Segment": "string",
    "Country": "string",
    "City": "string",
    "State": "string",
    "Postal Code": "string",  # kept as string: postal codes are identifiers, not numbers
    "Region": "string",
    "Product ID": "string",
    "Category": "string",
    "Sub-Category": "string",
    "Product Name": "string",
    "Sales": "float64",
    "Quantity": "int64",
    "Discount": "float64",
    "Profit": "float64",
}


class SchemaValidationError(Exception):
    """Raised when the raw file doesn't match the columns we expect."""


def load_raw_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load the raw Superstore CSV from disk.

    Args:
        file_path: Path to the raw CSV file (typically data/raw/superstore.csv).

    Returns:
        A DataFrame with raw, unvalidated, uncleaned data -- exactly as it
        appears in the source file, aside from dtype coercion on read.

    Raises:
        FileNotFoundError: If file_path does not exist.
        SchemaValidationError: If required columns are missing.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {file_path}")

    logger.info("Loading raw data from %s", file_path)

    # encoding='latin-1' because the public Superstore export ships with
    # Windows-1252 characters (e.g. in some Product Name fields) that break
    # a strict utf-8 read.
    df = pd.read_csv(file_path, encoding="latin-1")

    _validate_schema(df)
    df = _standardize_column_names(df)

    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Ensure every expected raw column is present before we proceed."""
    missing = set(EXPECTED_RAW_COLUMNS) - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"Raw data is missing expected columns: {sorted(missing)}. "
            "Check that you're loading the correct Superstore export."
        )


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'Order Date' -> 'order_date', 'Sub-Category' -> 'sub_category',
    etc. Snake_case column names are used everywhere downstream (Python and
    SQL both prefer this convention over 'Order Date' with a literal space).
    """
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    return df
