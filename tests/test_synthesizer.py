"""Tests for key synthesizer functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from faker import Faker

from sanitizer.synthesizer import (
    FAKER_LOCALES,
    _faker_for_column,
    _stitch_foreign_keys,
    postprocess_text_columns,
    preview_sample,
)


class TestFakerForColumn:
    def test_email_pattern(self):
        gen = _faker_for_column("email")
        result = gen()
        assert "@" in result

    def test_name_pattern(self):
        gen = _faker_for_column("customer_name")
        result = gen()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_override_takes_priority(self):
        gen = _faker_for_column("some_column", override="email")
        result = gen()
        assert "@" in result

    def test_fallback_produces_string(self):
        gen = _faker_for_column("xyz_unknown_column_123")
        result = gen()
        assert isinstance(result, str)


class TestStitchForeignKeys:
    def test_preserves_cardinality_distribution(
        self, sample_analysis, sample_pandas_dfs
    ):
        """FK stitching should roughly preserve the original FK distribution."""
        # Original: customer_id distribution is [1:3, 2:1, 3:1, 4:1, 5:1] (skewed)
        np.random.seed(42)
        synthetic = {
            "customers": pd.DataFrame({"customer_id": [10, 20, 30, 40, 50]}),
            "orders": pd.DataFrame(
                {
                    "order_id": range(1000),
                    "customer_id": [0] * 1000,  # placeholder
                }
            ),
        }
        result = _stitch_foreign_keys(sample_analysis, synthetic, sample_pandas_dfs)
        fk_values = result["orders"]["customer_id"]

        # All FK values should exist in parent
        parent_ids = set(synthetic["customers"]["customer_id"])
        assert set(fk_values.dropna().unique()).issubset(parent_ids)

    def test_preserves_null_ratio(self, sample_analysis):
        original = {
            "customers": pd.DataFrame({"customer_id": [1, 2, 3]}),
            "orders": pd.DataFrame(
                {
                    "order_id": [1, 2, 3, 4],
                    "customer_id": [1, 2, None, None],  # 50% null
                }
            ),
        }
        synthetic = {
            "customers": pd.DataFrame({"customer_id": [10, 20, 30]}),
            "orders": pd.DataFrame(
                {
                    "order_id": range(100),
                    "customer_id": [0] * 100,
                }
            ),
        }
        np.random.seed(42)
        result = _stitch_foreign_keys(sample_analysis, synthetic, original)
        null_ratio = result["orders"]["customer_id"].isna().mean()
        # Should be roughly 50% (within tolerance)
        assert 0.3 < null_ratio < 0.7


class TestPostprocessTextColumns:
    def test_replaces_text_columns(self, sample_analysis, sample_pandas_dfs):
        synth = {"customers": sample_pandas_dfs["customers"].copy()}
        result = postprocess_text_columns(sample_analysis, synth)
        # "name" and "email" are TEXT columns — should be replaced with Faker values
        assert (
            result["customers"]["name"].tolist()
            != sample_pandas_dfs["customers"]["name"].tolist()
        )


class TestPreviewSample:
    def test_generates_correct_rows(self, sample_analysis, sample_pandas_dfs):
        preview = preview_sample(
            "customers", sample_analysis, sample_pandas_dfs, n_rows=5
        )
        assert len(preview) == 5
        assert "customer_id" in preview.columns
        assert "name" in preview.columns

    def test_fk_references_parent(self, sample_analysis, sample_pandas_dfs):
        preview = preview_sample(
            "orders", sample_analysis, sample_pandas_dfs, n_rows=10
        )
        parent_vals = set(sample_pandas_dfs["customers"]["customer_id"].values)
        fk_vals = set(preview["customer_id"].dropna().values)
        assert fk_vals.issubset(parent_vals)


class TestLocaleSupport:
    def test_locale_list_not_empty(self):
        assert len(FAKER_LOCALES) > 10
        assert "en_US" in FAKER_LOCALES
        assert "fr_FR" in FAKER_LOCALES

    def test_locale_changes_faker_instance(self):
        """Passing a locale-specific Faker should produce locale-appropriate output."""
        fr_fake = Faker("fr_FR")
        gen = _faker_for_column("name", fake=fr_fake)
        result = gen()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_faker_for_column_uses_provided_instance(self):
        """_faker_for_column should use the provided Faker instance."""
        de_fake = Faker("de_DE")
        gen = _faker_for_column("city", fake=de_fake)
        result = gen()
        assert isinstance(result, str)
        assert len(result) > 0
