"""Tests for the validator module."""

from __future__ import annotations

import pandas as pd

from sanitizer.models import AnalysisResult, DateConstraintPair
from sanitizer.validator import (
    validate_all,
    validate_date_constraints,
    validate_foreign_keys,
    validate_primary_keys,
    validate_row_counts,
)


class TestValidatePrimaryKeys:
    def test_unique_pks_pass(self, sample_analysis):
        synth = {
            "customers": pd.DataFrame({"customer_id": [10, 20, 30]}),
            "orders": pd.DataFrame({"order_id": [1, 2, 3]}),
        }
        results = validate_primary_keys(sample_analysis, synth)
        assert results["customers"].passed
        assert results["orders"].passed

    def test_duplicate_pks_fail(self, sample_analysis):
        synth = {
            "customers": pd.DataFrame({"customer_id": [10, 10, 30]}),
            "orders": pd.DataFrame({"order_id": [1, 2, 3]}),
        }
        results = validate_primary_keys(sample_analysis, synth)
        assert not results["customers"].passed
        assert results["customers"].duplicate_count == 1


class TestValidateForeignKeys:
    def test_valid_fks_pass(self, sample_analysis):
        synth = {
            "customers": pd.DataFrame({"customer_id": [10, 20, 30]}),
            "orders": pd.DataFrame(
                {"order_id": [1, 2, 3], "customer_id": [10, 20, 30]}
            ),
        }
        results = validate_foreign_keys(sample_analysis, synth)
        assert len(results) == 1
        key = next(iter(results.keys()))
        assert results[key].passed

    def test_orphaned_fks_fail(self, sample_analysis):
        synth = {
            "customers": pd.DataFrame({"customer_id": [10, 20]}),
            "orders": pd.DataFrame({"order_id": [1, 2], "customer_id": [10, 99]}),
        }
        results = validate_foreign_keys(sample_analysis, synth)
        key = next(iter(results.keys()))
        assert not results[key].passed
        assert results[key].orphan_count == 1


class TestValidateDateConstraints:
    def test_no_violations_pass(self):
        analysis = AnalysisResult(
            date_constraints=[
                DateConstraintPair(
                    table_name="orders",
                    low_column="start",
                    high_column="end",
                )
            ],
        )
        synth = {
            "orders": pd.DataFrame(
                {
                    "start": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                    "end": pd.to_datetime(["2024-01-15", "2024-02-15"]),
                }
            )
        }
        results = validate_date_constraints(analysis, synth)
        key = next(iter(results.keys()))
        assert results[key].passed

    def test_violations_fail(self):
        analysis = AnalysisResult(
            date_constraints=[
                DateConstraintPair(
                    table_name="orders",
                    low_column="start",
                    high_column="end",
                )
            ],
        )
        synth = {
            "orders": pd.DataFrame(
                {
                    "start": pd.to_datetime(["2024-03-01", "2024-02-01"]),
                    "end": pd.to_datetime(["2024-01-15", "2024-02-15"]),
                }
            )
        }
        results = validate_date_constraints(analysis, synth)
        key = next(iter(results.keys()))
        assert not results[key].passed
        assert results[key].violation_count == 1


class TestValidateRowCounts:
    def test_correct_scale(self, sample_analysis):
        synth = {
            "customers": pd.DataFrame({"customer_id": range(5)}),
            "orders": pd.DataFrame({"order_id": range(8)}),
        }
        results = validate_row_counts(sample_analysis, synth, scale=1.0)
        assert results["customers"].synthetic_rows == 5
        assert results["customers"].expected_rows == 5


class TestValidateAll:
    def test_all_passed(self, sample_analysis, sample_pandas_dfs):
        # Build synthetic data that passes all checks
        synth = {
            "customers": pd.DataFrame(
                {
                    "customer_id": [10, 20, 30, 40, 50],
                    "name": ["A", "B", "C", "D", "E"],
                    "region": ["N", "S", "N", "S", "N"],
                    "country": ["US", "UK", "US", "UK", "US"],
                    "email": ["x@test.com"] * 5,
                }
            ),
            "orders": pd.DataFrame(
                {
                    "order_id": [1, 2, 3, 4, 5, 6, 7, 8],
                    "customer_id": [10, 10, 20, 30, 30, 30, 40, 50],
                    "order_date": ["2024-01-01"] * 8,
                    "ship_date": ["2024-01-05"] * 8,
                    "amount": [100.0] * 8,
                }
            ),
        }
        result = validate_all(sample_analysis, synth, sample_pandas_dfs, scale=1.0)
        assert result.all_passed
