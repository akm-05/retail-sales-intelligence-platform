"""
preprocessing.py

Takes the cleaned-column, raw-values DataFrame from data_loader.py and:
  1. Cleans it (missing values, duplicates, type casting, text normalization)
  2. Decomposes it into the eight normalized tables defined in sql/schema.sql
  3. Validates referential integrity across those tables before anything is
     written to disk or loaded into Postgres

The output of build_normalized_tables() is a dict[str, pd.DataFrame] whose
keys are the exact table names in schema.sql -- this keeps preprocessing.py
and schema.sql in lockstep by construction, rather than by convention.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataQualityError(Exception):
    """Raised when cleaned data fails a referential-integrity or business-rule check."""


# ----------------------------------------------------------------------------
# 1. CLEANING
# ----------------------------------------------------------------------------

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the raw (but column-standardized) DataFrame.

    Order matters here: duplicates are removed before type casting (so we're
    not casting rows we're about to discard), and missing-value handling
    happens after casting (so we can use proper NaT/NaN semantics rather
    than string comparisons).

    Args:
        df: Output of data_loader.load_raw_data().

    Returns:
        A cleaned DataFrame, same shape or smaller, ready for decomposition
        into normalized tables.
    """
    logger.info("Starting cleaning: %d rows in", len(df))

    df = _strip_whitespace(df)
    df = _remove_duplicates(df)
    df = _cast_types(df)
    df = _handle_missing_values(df)
    df = _validate_business_rules(df)

    logger.info("Cleaning complete: %d rows out", len(df))
    return df


def _strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace on every string column.

    Raw exports frequently have trailing spaces in fields like
    'Customer Name' or 'Product Name' (e.g. 'John Smith ') that would
    otherwise create phantom duplicate customers/products after grouping.
    """
    df = df.copy()
    string_cols = df.select_dtypes(include="string").columns
    for col in string_cols:
        df[col] = df[col].str.strip()
    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows.

    We deliberately do NOT dedupe on row_id alone -- row_id is a raw export
    artifact, not a business key. A true duplicate is a row that matches on
    every business-meaningful column.
    """
    before = len(df)
    business_cols = [c for c in df.columns if c != "row_id"]
    df = df.drop_duplicates(subset=business_cols, keep="first").reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.warning("Removed %d exact duplicate rows", dropped)
    return df


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast date columns to datetime and numeric columns to proper dtypes."""
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], format="%m/%d/%Y", errors="coerce")
    df["ship_date"] = pd.to_datetime(df["ship_date"], format="%m/%d/%Y", errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("Int64")
    df["discount"] = pd.to_numeric(df["discount"], errors="coerce")
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
    return df


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business-rule-driven missing value handling -- not blanket imputation.

    - order_date / ship_date / sales / quantity are non-negotiable for
      analytics and forecasting: rows missing these are dropped, since
      imputing a sale amount or an order date would fabricate business facts.
    - postal_code is the one field genuinely allowed to be missing (a real
      Superstore data quirk: some rows -- consistently in New York City --
      have no postal code recorded). We fill with 'UNKNOWN' rather than
      drop the row, since city/state/region are still valid and usable.
    """
    df = df.copy()

    critical_cols = ["order_date", "ship_date", "sales", "quantity", "customer_id", "product_id"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with missing critical fields", dropped)

    df["postal_code"] = df["postal_code"].fillna("UNKNOWN")

    return df


def _validate_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce business rules that mirror the CHECK constraints in schema.sql.
    Failing fast here (in Python) gives a much more specific error message
    than letting Postgres reject the row at insert time.
    """
    df = df.copy()

    before = len(df)
    df = df[df["ship_date"] >= df["order_date"]]
    invalid_dates = before - len(df)
    if invalid_dates:
        logger.warning("Dropped %d rows where ship_date < order_date", invalid_dates)

    before = len(df)
    df = df[(df["discount"] >= 0) & (df["discount"] <= 1)]
    invalid_discount = before - len(df)
    if invalid_discount:
        logger.warning("Dropped %d rows with discount outside [0, 1]", invalid_discount)

    before = len(df)
    df = df[df["quantity"] > 0]
    invalid_qty = before - len(df)
    if invalid_qty:
        logger.warning("Dropped %d rows with non-positive quantity", invalid_qty)

    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. DECOMPOSITION INTO NORMALIZED TABLES
# ----------------------------------------------------------------------------

def build_normalized_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Decompose the cleaned flat DataFrame into the eight normalized tables
    from sql/schema.sql. Order of construction matters: tables with no
    foreign key dependencies (regions, categories) are built first, so that
    dependent tables (locations, sub_categories) can look up their parent's
    surrogate keys via merge.

    Args:
        df: Output of clean_raw_data().

    Returns:
        Dict keyed by table name, matching schema.sql exactly:
        regions, locations, customers, categories, sub_categories,
        products, orders, sales.
    """
    regions_df = _build_regions(df)
    locations_df = _build_locations(df, regions_df)
    customers_df = _build_customers(df)
    categories_df = _build_categories(df)
    sub_categories_df = _build_sub_categories(df, categories_df)
    products_df = _build_products(df, sub_categories_df)
    orders_df = _build_orders(df, locations_df)
    sales_df = _build_sales(df)

    tables = {
        "regions": regions_df,
        "locations": locations_df,
        "customers": customers_df,
        "categories": categories_df,
        "sub_categories": sub_categories_df,
        "products": products_df,
        "orders": orders_df,
        "sales": sales_df,
    }

    _validate_referential_integrity(tables)
    return tables


def _build_regions(df: pd.DataFrame) -> pd.DataFrame:
    regions = df[["region"]].drop_duplicates().reset_index(drop=True)
    regions.insert(0, "region_id", range(1, len(regions) + 1))
    return regions.rename(columns={"region": "region_name"})


def _build_locations(df: pd.DataFrame, regions_df: pd.DataFrame) -> pd.DataFrame:
    locations = (
        df[["city", "state", "postal_code", "region"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    locations = locations.merge(
        regions_df, left_on="region", right_on="region_name", how="left"
    )
    locations.insert(0, "location_id", range(1, len(locations) + 1))
    return locations[["location_id", "city", "state", "postal_code", "region_id"]]


def _build_customers(df: pd.DataFrame) -> pd.DataFrame:
    customers = (
        df[["customer_id", "customer_name", "segment"]]
        .drop_duplicates(subset="customer_id")
        .reset_index(drop=True)
    )
    return customers


def _build_categories(df: pd.DataFrame) -> pd.DataFrame:
    categories = df[["category"]].drop_duplicates().reset_index(drop=True)
    categories.insert(0, "category_id", range(1, len(categories) + 1))
    return categories.rename(columns={"category": "category_name"})


def _build_sub_categories(df: pd.DataFrame, categories_df: pd.DataFrame) -> pd.DataFrame:
    sub_categories = (
        df[["sub_category", "category"]].drop_duplicates().reset_index(drop=True)
    )
    sub_categories = sub_categories.merge(
        categories_df, left_on="category", right_on="category_name", how="left"
    )
    sub_categories.insert(0, "subcategory_id", range(1, len(sub_categories) + 1))
    return sub_categories[["subcategory_id", "sub_category", "category_id"]].rename(
        columns={"sub_category": "subcategory_name"}
    )


def _build_products(df: pd.DataFrame, sub_categories_df: pd.DataFrame) -> pd.DataFrame:
    products = (
        df[["product_id", "product_name", "sub_category"]]
        .drop_duplicates(subset="product_id")
        .reset_index(drop=True)
    )
    products = products.merge(
        sub_categories_df,
        left_on="sub_category",
        right_on="subcategory_name",
        how="left",
    )
    return products[["product_id", "product_name", "subcategory_id"]]


def _build_orders(df: pd.DataFrame, locations_df: pd.DataFrame) -> pd.DataFrame:
    orders = (
        df[["order_id", "order_date", "ship_date", "ship_mode", "customer_id",
            "city", "state", "postal_code"]]
        .drop_duplicates(subset="order_id")
        .reset_index(drop=True)
    )
    orders = orders.merge(
        locations_df, on=["city", "state", "postal_code"], how="left"
    )
    return orders[
        ["order_id", "order_date", "ship_date", "ship_mode", "customer_id", "location_id"]
    ]


def _build_sales(df: pd.DataFrame) -> pd.DataFrame:
    sales = df[["order_id", "product_id", "sales", "quantity", "discount", "profit"]].copy()
    sales.insert(0, "sales_id", range(1, len(sales) + 1))
    return sales


# ----------------------------------------------------------------------------
# 3. REFERENTIAL INTEGRITY VALIDATION
# ----------------------------------------------------------------------------

def _validate_referential_integrity(tables: dict[str, pd.DataFrame]) -> None:
    """
    Verify every foreign key in every table resolves to an existing parent
    row -- mirroring the FOREIGN KEY constraints in schema.sql, but checked
    here so a violation is caught with a specific, debuggable message
    instead of a generic Postgres constraint-violation error mid-load.

    Raises:
        DataQualityError: If any foreign key reference is dangling.
    """
    checks = [
        ("locations.region_id", tables["locations"]["region_id"], tables["regions"]["region_id"]),
        ("sub_categories.category_id", tables["sub_categories"]["category_id"], tables["categories"]["category_id"]),
        ("products.subcategory_id", tables["products"]["subcategory_id"], tables["sub_categories"]["subcategory_id"]),
        ("orders.customer_id", tables["orders"]["customer_id"], tables["customers"]["customer_id"]),
        ("orders.location_id", tables["orders"]["location_id"], tables["locations"]["location_id"]),
        ("sales.order_id", tables["sales"]["order_id"], tables["orders"]["order_id"]),
        ("sales.product_id", tables["sales"]["product_id"], tables["products"]["product_id"]),
    ]

    for name, child_col, parent_col in checks:
        orphans = set(child_col.dropna()) - set(parent_col.dropna())
        if orphans:
            raise DataQualityError(
                f"Referential integrity violation in {name}: "
                f"{len(orphans)} value(s) with no matching parent row, "
                f"e.g. {list(orphans)[:5]}"
            )

    logger.info("Referential integrity validated across all %d tables", len(tables))
