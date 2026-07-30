"""
test_preprocessing.py

Unit tests for src/preprocessing.py. These use small, hand-built synthetic
DataFrames rather than the real Superstore CSV, so tests run fast and each
one isolates exactly one cleaning rule or integrity check.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import (
    DataQualityError,
    _handle_missing_values,
    _remove_duplicates,
    _validate_business_rules,
    _validate_referential_integrity,
    build_normalized_tables,
)


def _sample_raw_df(n_rows: int = 4) -> pd.DataFrame:
    """A minimal, already-typed DataFrame matching the post-_cast_types schema."""
    base = {
        "row_id": list(range(1, n_rows + 1)),
        "order_id": [f"ORD-{i}" for i in range(1, n_rows + 1)],
        "order_date": pd.to_datetime(["2023-01-01"] * n_rows),
        "ship_date": pd.to_datetime(["2023-01-05"] * n_rows),
        "ship_mode": ["Standard Class"] * n_rows,
        "customer_id": [f"CUST-{i}" for i in range(1, n_rows + 1)],
        "customer_name": [f"Customer {i}" for i in range(1, n_rows + 1)],
        "segment": ["Consumer"] * n_rows,
        "country": ["United States"] * n_rows,
        "city": ["New York"] * n_rows,
        "state": ["New York"] * n_rows,
        "postal_code": ["10001"] * n_rows,
        "region": ["East"] * n_rows,
        "product_id": [f"PROD-{i}" for i in range(1, n_rows + 1)],
        "category": ["Furniture"] * n_rows,
        "sub_category": ["Chairs"] * n_rows,
        "product_name": [f"Product {i}" for i in range(1, n_rows + 1)],
        "sales": [100.0] * n_rows,
        "quantity": [2] * n_rows,
        "discount": [0.1] * n_rows,
        "profit": [20.0] * n_rows,
    }
    return pd.DataFrame(base)


class TestRemoveDuplicates:
    def test_drops_exact_business_duplicates(self):
        df = _sample_raw_df(n_rows=2)
        df.loc[1] = df.loc[0]  # make row 1 an exact duplicate of row 0
        df.loc[1, "row_id"] = 99  # except for row_id, which shouldn't matter

        result = _remove_duplicates(df)

        assert len(result) == 1

    def test_keeps_rows_that_differ_on_business_columns(self):
        df = _sample_raw_df(n_rows=2)
        result = _remove_duplicates(df)
        assert len(result) == 2


class TestHandleMissingValues:
    def test_drops_rows_missing_critical_fields(self):
        df = _sample_raw_df(n_rows=2)
        df.loc[0, "sales"] = None

        result = _handle_missing_values(df)

        assert len(result) == 1

    def test_fills_missing_postal_code_instead_of_dropping(self):
        df = _sample_raw_df(n_rows=1)
        df.loc[0, "postal_code"] = None

        result = _handle_missing_values(df)

        assert len(result) == 1
        assert result.loc[0, "postal_code"] == "UNKNOWN"


class TestValidateBusinessRules:
    def test_drops_rows_where_ship_before_order(self):
        df = _sample_raw_df(n_rows=1)
        df.loc[0, "ship_date"] = pd.Timestamp("2022-12-31")  # before order_date

        result = _validate_business_rules(df)

        assert len(result) == 0

    def test_drops_rows_with_discount_out_of_range(self):
        df = _sample_raw_df(n_rows=1)
        df.loc[0, "discount"] = 1.5

        result = _validate_business_rules(df)

        assert len(result) == 0

    def test_drops_rows_with_non_positive_quantity(self):
        df = _sample_raw_df(n_rows=1)
        df.loc[0, "quantity"] = 0

        result = _validate_business_rules(df)

        assert len(result) == 0

    def test_valid_row_passes_through_unchanged(self):
        df = _sample_raw_df(n_rows=1)
        result = _validate_business_rules(df)
        assert len(result) == 1


class TestBuildNormalizedTables:
    def test_produces_all_eight_tables(self):
        df = _sample_raw_df(n_rows=3)
        tables = build_normalized_tables(df)

        expected_tables = {
            "regions", "locations", "customers", "categories",
            "sub_categories", "products", "orders", "sales",
        }
        assert set(tables.keys()) == expected_tables

    def test_no_duplicate_order_ids_in_orders_table(self):
        df = _sample_raw_df(n_rows=3)
        # simulate one order with two line items (same order_id, different product)
        df.loc[1, "order_id"] = df.loc[0, "order_id"]

        tables = build_normalized_tables(df)

        assert tables["orders"]["order_id"].is_unique

    def test_sales_row_count_matches_input_line_items(self):
        df = _sample_raw_df(n_rows=5)
        tables = build_normalized_tables(df)
        assert len(tables["sales"]) == 5


class TestReferentialIntegrity:
    def test_raises_on_dangling_foreign_key(self):
        df = _sample_raw_df(n_rows=2)
        tables = build_normalized_tables(df)

        # deliberately break the FK: point a sales row at a nonexistent order
        tables["sales"].loc[0, "order_id"] = "NONEXISTENT-ORDER"

        with pytest.raises(DataQualityError):
            _validate_referential_integrity(tables)

    def test_passes_on_clean_normalized_tables(self):
        df = _sample_raw_df(n_rows=3)
        tables = build_normalized_tables(df)
        # build_normalized_tables already calls this internally and would
        # have raised if it failed -- this re-asserts it explicitly.
        _validate_referential_integrity(tables)
