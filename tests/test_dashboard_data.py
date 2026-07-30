"""
test_dashboard_data.py

Unit tests for src/dashboard_data.py. The Postgres-backed loader
(load_tables_from_postgres) is intentionally not unit tested here -- it's a
thin pass-through to pd.read_sql with no branching logic of its own, and
exercising it for real would require a live database connection, which is
what the full pipeline's manual/integration testing already covers. What IS
unit tested: the CSV fallback (a real branch of logic worth protecting) and
compute_kpis (pure arithmetic with edge cases worth pinning down).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.dashboard_data import compute_kpis, load_analytics_view, load_tables_from_processed_csv
from src.sql_connector import TABLE_LOAD_ORDER


def _write_minimal_processed_tables(tmp_path):
    """Writes the smallest possible set of 8 valid, FK-consistent CSVs."""
    (tmp_path / "regions.csv").write_text("region_id,region_name\n1,East\n")
    (tmp_path / "categories.csv").write_text("category_id,category_name\n1,Technology\n")
    (tmp_path / "locations.csv").write_text(
        "location_id,city,state,postal_code,region_id\n1,New York,New York,10001,1\n"
    )
    (tmp_path / "sub_categories.csv").write_text(
        "subcategory_id,subcategory_name,category_id\n1,Phones,1\n"
    )
    (tmp_path / "customers.csv").write_text(
        "customer_id,customer_name,segment\nC-1,Jane Doe,Consumer\n"
    )
    (tmp_path / "products.csv").write_text(
        "product_id,product_name,subcategory_id\nP-1,Smartphone,1\n"
    )
    (tmp_path / "orders.csv").write_text(
        "order_id,order_date,ship_date,ship_mode,customer_id,location_id\n"
        "O-1,2023-01-05,2023-01-08,Standard Class,C-1,1\n"
    )
    (tmp_path / "sales.csv").write_text(
        "sales_id,order_id,product_id,sales,quantity,discount,profit\n"
        "1,O-1,P-1,500.0,2,0.1,120.0\n"
    )
    return tmp_path


class TestLoadTablesFromProcessedCsv:
    def test_loads_all_eight_tables(self, tmp_path):
        _write_minimal_processed_tables(tmp_path)
        tables = load_tables_from_processed_csv(tmp_path)
        assert set(tables.keys()) == set(TABLE_LOAD_ORDER)

    def test_order_date_parsed_as_datetime(self, tmp_path):
        _write_minimal_processed_tables(tmp_path)
        tables = load_tables_from_processed_csv(tmp_path)
        assert pd.api.types.is_datetime64_any_dtype(tables["orders"]["order_date"])

    def test_raises_clear_error_when_csv_missing(self, tmp_path):
        # Deliberately don't write any files.
        with pytest.raises(FileNotFoundError):
            load_tables_from_processed_csv(tmp_path)


class TestLoadAnalyticsView:
    def test_falls_back_to_csv_when_postgres_unavailable(self, tmp_path, monkeypatch):
        _write_minimal_processed_tables(tmp_path)

        def _raise_connection_error():
            raise ConnectionError("simulated: no Postgres running")

        monkeypatch.setattr("src.dashboard_data.load_tables_from_postgres", _raise_connection_error)

        result = load_analytics_view(prefer_postgres=True, processed_dir=tmp_path)
        assert len(result) == 1
        assert result.loc[0, "sales"] == 500.0

    def test_reads_csv_directly_when_prefer_postgres_false(self, tmp_path, monkeypatch):
        _write_minimal_processed_tables(tmp_path)

        def _fail_if_called():
            raise AssertionError("load_tables_from_postgres should not be called")

        monkeypatch.setattr("src.dashboard_data.load_tables_from_postgres", _fail_if_called)

        result = load_analytics_view(prefer_postgres=False, processed_dir=tmp_path)
        assert len(result) == 1


class TestComputeKpis:
    def test_basic_kpis(self):
        df = pd.DataFrame({
            "order_id": ["O-1", "O-1", "O-2"],
            "sales": [100.0, 50.0, 200.0],
            "profit": [20.0, 10.0, -40.0],
        })
        kpis = compute_kpis(df)
        assert kpis["total_revenue"] == 350.0
        assert kpis["total_profit"] == -10.0
        assert kpis["total_orders"] == 2
        assert kpis["profit_margin"] == pytest.approx(-10.0 / 350.0)
        assert kpis["avg_order_value"] == pytest.approx(175.0)

    def test_zero_revenue_does_not_raise(self):
        df = pd.DataFrame({"order_id": [], "sales": [], "profit": []})
        kpis = compute_kpis(df)
        assert pd.isna(kpis["profit_margin"])
        assert pd.isna(kpis["avg_order_value"])
