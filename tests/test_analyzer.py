"""Tests for the analyzer module."""

from __future__ import annotations

import polars as pl

from sanitizer.analyzer import (
    analyze,
    classify_column,
    detect_dimension_groups,
    detect_dimensions,
    detect_foreign_keys,
    detect_primary_keys,
    detect_sensitivity,
    detect_text_columns,
)
from sanitizer.models import ColumnRole, SensitivityLevel


class TestDetectPrimaryKeys:
    def test_detects_id_column(self, customers_polars):
        candidates = detect_primary_keys("customers", customers_polars)
        assert len(candidates) >= 1
        assert candidates[0][0] == "customer_id"
        assert candidates[0][1] == 1.0

    def test_empty_dataframe(self):
        df = pl.DataFrame({"id": []}).cast({"id": pl.Int64})
        candidates = detect_primary_keys("empty", df)
        assert candidates == []

    def test_no_pk_suffix_falls_back_to_100_unique(self):
        df = pl.DataFrame({"code": [1, 2, 3, 4, 5]})
        candidates = detect_primary_keys("test", df)
        # "code" has no PK suffix, but 100% unique => fallback
        assert len(candidates) == 1
        assert candidates[0][0] == "code"

    def test_nulls_disqualify_pk(self):
        df = pl.DataFrame({"item_id": [1, 2, None, 4, 5]})
        candidates = detect_primary_keys("test", df)
        assert all(c[0] != "item_id" for c in candidates)

    def test_low_uniqueness_rejected(self):
        df = pl.DataFrame({"category_id": [1, 1, 1, 2, 2]})
        candidates = detect_primary_keys("test", df)
        assert all(c[0] != "category_id" for c in candidates)


class TestDetectForeignKeys:
    def test_detects_customer_id_fk(self, sample_polars_dfs):
        pks = {"customers": "customer_id"}
        rels = detect_foreign_keys(sample_polars_dfs, pks)
        assert len(rels) == 1
        assert rels[0].parent_table == "customers"
        assert rels[0].child_column == "customer_id"
        assert rels[0].child_table == "orders"

    def test_no_self_reference(self):
        df = pl.DataFrame({"item_id": [1, 2, 3], "value": [10, 20, 30]})
        pks = {"test": "item_id"}
        rels = detect_foreign_keys({"test": df}, pks)
        assert len(rels) == 0


class TestDetectDimensions:
    def test_finds_low_cardinality_strings(self, customers_polars):
        dims = detect_dimensions("customers", customers_polars)
        assert "region" in dims
        assert "country" in dims

    def test_excludes_specified_columns(self, customers_polars):
        dims = detect_dimensions("customers", customers_polars, exclude_cols={"region"})
        assert "region" not in dims

    def test_empty_df(self):
        df = pl.DataFrame({"region": []}).cast({"region": pl.String})
        dims = detect_dimensions("test", df)
        assert dims == []


class TestDetectDimensionGroups:
    def test_detects_functional_dependency(self, customers_polars):
        groups = detect_dimension_groups(
            "customers", customers_polars, ["region", "country"]
        )
        assert len(groups) == 1
        assert set(groups[0].column_names) == {"region", "country"}

    def test_single_column_no_group(self, customers_polars):
        groups = detect_dimension_groups("customers", customers_polars, ["region"])
        assert groups == []


class TestDetectTextColumns:
    def test_detects_high_cardinality_strings(self, customers_polars):
        text_cols = detect_text_columns("customers", customers_polars)
        assert "name" in text_cols
        assert "email" in text_cols

    def test_excludes_dimensions(self, customers_polars):
        text_cols = detect_text_columns(
            "customers", customers_polars, exclude_cols={"region", "country"}
        )
        assert "region" not in text_cols


class TestClassifyColumn:
    def test_pk_column(self, customers_polars):
        meta = classify_column(
            "customer_id", customers_polars, "customer_id", set(), set(), set()
        )
        assert meta.role == ColumnRole.PRIMARY_KEY
        assert meta.is_primary_key is True
        assert meta.sdtype == "id"

    def test_fk_column(self, orders_polars):
        meta = classify_column(
            "customer_id", orders_polars, "order_id", {"customer_id"}, set(), set()
        )
        assert meta.role == ColumnRole.FOREIGN_KEY
        assert meta.sdtype == "id"

    def test_dimension_column(self, customers_polars):
        meta = classify_column(
            "region", customers_polars, "customer_id", set(), {"region"}, set()
        )
        assert meta.role == ColumnRole.DIMENSION
        assert meta.sdtype == "categorical"

    def test_measure_column(self, orders_polars):
        meta = classify_column("amount", orders_polars, "order_id", set(), set(), set())
        assert meta.role == ColumnRole.MEASURE
        assert meta.sdtype == "numerical"


class TestAnalyze:
    def test_full_analysis(self, sample_polars_dfs):
        sources = {"customers": "test.xlsx", "orders": "test.xlsx"}
        result = analyze(sample_polars_dfs, sources)
        assert "customers" in result.tables
        assert "orders" in result.tables
        assert result.tables["customers"].primary_key == "customer_id"
        assert result.tables["orders"].primary_key == "order_id"
        assert len(result.relationships) >= 1

    def test_custom_thresholds(self, sample_polars_dfs):
        """Analysis should accept custom threshold parameters."""
        sources = {"customers": "test.xlsx", "orders": "test.xlsx"}
        # Very strict PK threshold — should still find customer_id (100% unique)
        result = analyze(
            sample_polars_dfs,
            sources,
            pk_threshold=0.99,
            fk_threshold=0.90,
            dim_threshold=0.03,
        )
        assert result.tables["customers"].primary_key == "customer_id"

    def test_relaxed_fk_threshold(self, sample_polars_dfs):
        """Lower FK threshold should still detect relationships."""
        sources = {"customers": "test.xlsx", "orders": "test.xlsx"}
        result = analyze(sample_polars_dfs, sources, fk_threshold=0.50)
        assert len(result.relationships) >= 1


class TestDetectSensitivity:
    def test_high_sensitivity_ssn(self):
        assert detect_sensitivity("ssn") == SensitivityLevel.HIGH
        assert detect_sensitivity("social_security_number") == SensitivityLevel.HIGH

    def test_high_sensitivity_credit_card(self):
        assert detect_sensitivity("credit_card") == SensitivityLevel.HIGH
        assert detect_sensitivity("card_number") == SensitivityLevel.HIGH

    def test_low_sensitivity_email(self):
        assert detect_sensitivity("email") == SensitivityLevel.LOW
        assert detect_sensitivity("email_address") == SensitivityLevel.LOW

    def test_low_sensitivity_phone(self):
        assert detect_sensitivity("phone_number") == SensitivityLevel.LOW

    def test_low_sensitivity_name(self):
        assert detect_sensitivity("first_name") == SensitivityLevel.LOW
        assert detect_sensitivity("last_name") == SensitivityLevel.LOW

    def test_no_sensitivity(self):
        assert detect_sensitivity("amount") == SensitivityLevel.NONE
        assert detect_sensitivity("region") == SensitivityLevel.NONE
        assert detect_sensitivity("order_id") == SensitivityLevel.NONE

    def test_classify_column_sets_sensitivity(self, customers_polars):
        """classify_column should set sensitivity on the resulting ColumnMeta."""
        meta = classify_column(
            "email", customers_polars, "customer_id", set(), set(), {"email"}
        )
        assert meta.sensitivity == SensitivityLevel.LOW
