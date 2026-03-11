"""Shared fixtures for Sanitizer tests."""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from sanitizer.models import (
    AnalysisResult,
    ColumnMeta,
    ColumnRole,
    DimensionGroup,
    Relationship,
    TableMeta,
)


@pytest.fixture
def customers_polars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "region": ["North", "South", "North", "South", "North"],
            "country": ["US", "UK", "US", "UK", "US"],
            "email": [
                "a@test.com",
                "b@test.com",
                "c@test.com",
                "d@test.com",
                "e@test.com",
            ],
        }
    )


@pytest.fixture
def orders_polars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "order_id": [101, 102, 103, 104, 105, 106, 107, 108],
            "customer_id": [1, 1, 2, 3, 3, 3, 4, 5],
            "order_date": [
                "2024-01-01",
                "2024-01-15",
                "2024-02-01",
                "2024-02-15",
                "2024-03-01",
                "2024-03-15",
                "2024-04-01",
                "2024-04-15",
            ],
            "ship_date": [
                "2024-01-05",
                "2024-01-20",
                "2024-02-05",
                "2024-02-20",
                "2024-03-05",
                "2024-03-20",
                "2024-04-05",
                "2024-04-20",
            ],
            "amount": [100.0, 200.0, 150.0, 300.0, 250.0, 175.0, 225.0, 125.0],
        }
    )


@pytest.fixture
def customers_pandas(customers_polars) -> pd.DataFrame:
    return customers_polars.to_pandas()


@pytest.fixture
def orders_pandas(orders_polars) -> pd.DataFrame:
    return orders_polars.to_pandas()


@pytest.fixture
def sample_analysis(customers_polars, orders_polars) -> AnalysisResult:
    """Pre-built analysis result for customers + orders."""
    return AnalysisResult(
        tables={
            "customers": TableMeta(
                name="customers",
                file_path="test.xlsx",
                row_count=5,
                primary_key="customer_id",
                columns={
                    "customer_id": ColumnMeta(
                        name="customer_id",
                        sdtype="id",
                        role=ColumnRole.PRIMARY_KEY,
                        is_primary_key=True,
                        uniqueness_ratio=1.0,
                        sample_values=[1, 2, 3],
                    ),
                    "name": ColumnMeta(
                        name="name",
                        sdtype="unknown",
                        role=ColumnRole.TEXT,
                        uniqueness_ratio=1.0,
                        sample_values=["Alice", "Bob"],
                    ),
                    "region": ColumnMeta(
                        name="region",
                        sdtype="categorical",
                        role=ColumnRole.DIMENSION,
                        uniqueness_ratio=0.4,
                        sample_values=["North", "South"],
                    ),
                    "country": ColumnMeta(
                        name="country",
                        sdtype="categorical",
                        role=ColumnRole.DIMENSION,
                        uniqueness_ratio=0.4,
                        sample_values=["US", "UK"],
                    ),
                    "email": ColumnMeta(
                        name="email",
                        sdtype="unknown",
                        role=ColumnRole.TEXT,
                        uniqueness_ratio=1.0,
                        sample_values=["a@test.com"],
                    ),
                },
            ),
            "orders": TableMeta(
                name="orders",
                file_path="test.xlsx",
                row_count=8,
                primary_key="order_id",
                columns={
                    "order_id": ColumnMeta(
                        name="order_id",
                        sdtype="id",
                        role=ColumnRole.PRIMARY_KEY,
                        is_primary_key=True,
                        uniqueness_ratio=1.0,
                        sample_values=[101, 102],
                    ),
                    "customer_id": ColumnMeta(
                        name="customer_id",
                        sdtype="id",
                        role=ColumnRole.FOREIGN_KEY,
                        foreign_key_target="customers.customer_id",
                        uniqueness_ratio=0.625,
                        sample_values=[1, 1, 2],
                    ),
                    "order_date": ColumnMeta(
                        name="order_date",
                        sdtype="categorical",
                        role=ColumnRole.OTHER,
                        uniqueness_ratio=1.0,
                    ),
                    "ship_date": ColumnMeta(
                        name="ship_date",
                        sdtype="categorical",
                        role=ColumnRole.OTHER,
                        uniqueness_ratio=1.0,
                    ),
                    "amount": ColumnMeta(
                        name="amount",
                        sdtype="numerical",
                        role=ColumnRole.MEASURE,
                        uniqueness_ratio=1.0,
                        sample_values=[100.0, 200.0],
                    ),
                },
            ),
        },
        relationships=[
            Relationship(
                parent_table="customers",
                parent_column="customer_id",
                child_table="orders",
                child_column="customer_id",
                overlap_ratio=1.0,
            ),
        ],
        date_constraints=[],
        dimension_groups=[
            DimensionGroup(
                table_name="customers",
                column_names=["region", "country"],
                combination_count=2,
            ),
        ],
    )


@pytest.fixture
def sample_pandas_dfs(customers_pandas, orders_pandas) -> dict[str, pd.DataFrame]:
    return {"customers": customers_pandas, "orders": orders_pandas}


@pytest.fixture
def sample_polars_dfs(customers_polars, orders_polars) -> dict[str, pl.DataFrame]:
    return {"customers": customers_polars, "orders": orders_polars}
