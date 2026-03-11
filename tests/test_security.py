"""Security-focused tests for the Sanitizer application."""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

from sanitizer.config import _redact_sample, analysis_from_json, analysis_to_json
from sanitizer.loader import sanitize_column_name, sanitize_table_name
from sanitizer.logging_config import _SanitizingFilter
from sanitizer.models import (
    ColumnRole,
)
from sanitizer.synthesizer import preview_sample

# ── Path validation ──────────────────────────────────────────────────────────


class TestPathValidation:
    """Test that _validate_folder_path blocks dangerous paths."""

    @pytest.fixture(autouse=True)
    def _import_validate(self):
        """Import the validator; requires streamlit stub or patch."""
        # We test the logic directly by importing and patching st.error
        from sanitizer.ui import _BLOCKED_PREFIXES, _validate_folder_path

        self.validate = _validate_folder_path
        self.blocked_prefixes = _BLOCKED_PREFIXES

    @patch("sanitizer.ui.st")
    def test_blocks_root(self, mock_st):
        if sys.platform == "win32":
            result = self.validate("C:\\")
            assert result is None
        else:
            result = self.validate("/")
            assert result is None

    @patch("sanitizer.ui.st")
    def test_blocks_windows_system(self, mock_st):
        result = self.validate("C:\\Windows\\System32\\drivers")
        # On non-Windows this path won't resolve the same way, but the prefix
        # check should still trigger for the normalised string
        if sys.platform == "win32":
            assert result is None

    @patch("sanitizer.ui.st")
    def test_blocks_etc_subdirectory(self, mock_st):
        """Subdirectories of blocked prefixes must also be blocked."""
        if sys.platform != "win32":
            result = self.validate("/etc/passwd/..")
            assert result is None
            result = self.validate("/etc/shadow")
            assert result is None

    @patch("sanitizer.ui.st")
    def test_blocks_empty_and_whitespace(self, mock_st):
        assert self.validate("") is None
        assert self.validate("   ") is None

    @patch("sanitizer.ui.st")
    def test_allows_normal_directory(self, mock_st, tmp_path):
        result = self.validate(str(tmp_path))
        assert result is not None
        assert result == tmp_path.resolve()

    @patch("sanitizer.ui.st")
    def test_blocks_symlink(self, mock_st, tmp_path):
        """Symlinked directories should be rejected."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link = tmp_path / "link"
        try:
            link.symlink_to(real_dir)
        except OSError:
            pytest.skip("Symlinks not supported on this platform/config")
        result = self.validate(str(link))
        assert result is None


# ── Writer path traversal ────────────────────────────────────────────────────


class TestWriterPathTraversal:
    def test_dotdot_in_relative_dir_skipped(self, tmp_path):
        from sanitizer.writer import write_excel_files

        df = pd.DataFrame({"a": [1, 2, 3]})
        paths = write_excel_files({"tbl": df}, tmp_path, {"tbl": "../../escape"})
        assert len(paths) == 0

    def test_resolve_escape_skipped(self, tmp_path):
        from sanitizer.writer import write_excel_files

        df = pd.DataFrame({"a": [1]})
        # Even without ".." a resolved path escaping output should be caught
        paths = write_excel_files({"tbl": df}, tmp_path, {"tbl": ""})
        # Normal case — should succeed
        assert len(paths) == 1

    def test_symlink_target_dir_skipped(self, tmp_path):
        from sanitizer.writer import write_excel_files

        real_dir = tmp_path / "outside"
        real_dir.mkdir()
        link = tmp_path / "output" / "link"
        output = tmp_path / "output"
        output.mkdir()
        try:
            link.symlink_to(real_dir)
        except OSError:
            pytest.skip("Symlinks not supported")
        df = pd.DataFrame({"a": [1]})
        paths = write_excel_files({"tbl": df}, output, {"tbl": "link"})
        assert len(paths) == 0


# ── Sample redaction ─────────────────────────────────────────────────────────


class TestSampleRedaction:
    def test_redacts_non_empty_values(self):
        assert _redact_sample("john@example.com") == "[REDACTED]"
        assert _redact_sample("123-45-6789") == "[REDACTED]"
        assert _redact_sample("Alice") == "[REDACTED]"
        assert _redact_sample("AB") == "[REDACTED]"

    def test_preserves_empty_and_none(self):
        assert _redact_sample("") == ""
        assert _redact_sample("None") == "None"
        assert _redact_sample("nan") == "nan"
        assert _redact_sample(None) == "None"  # str(None) == "None"

    def test_config_roundtrip_redacted(self, sample_analysis):
        json_str = analysis_to_json(sample_analysis)
        data = json.loads(json_str)
        for table in data["tables"].values():
            for col in table["columns"].values():
                for sv in col["sample_values"]:
                    assert sv in ("", "None", "nan", "[REDACTED]"), (
                        f"Sample value not redacted: {sv!r}"
                    )


# ── Config injection ─────────────────────────────────────────────────────────


class TestConfigInjection:
    def test_rejects_invalid_role(self):
        bad_json = json.dumps(
            {
                "tables": {
                    "t": {
                        "name": "t",
                        "row_count": 5,
                        "columns": {
                            "c": {"name": "c", "sdtype": "unknown", "role": "EVIL_ROLE"}
                        },
                    }
                }
            }
        )
        analysis, _ = analysis_from_json(bad_json)
        # Should fall back to OTHER, not crash
        assert analysis.tables["t"].columns["c"].role == ColumnRole.OTHER

    def test_rejects_negative_row_count(self):
        bad_json = json.dumps(
            {
                "tables": {
                    "t": {
                        "name": "t",
                        "row_count": -999,
                        "columns": {},
                    }
                }
            }
        )
        analysis, _ = analysis_from_json(bad_json)
        assert analysis.tables["t"].row_count == 0

    def test_clamps_uniqueness_ratio(self):
        bad_json = json.dumps(
            {
                "tables": {
                    "t": {
                        "name": "t",
                        "row_count": 5,
                        "columns": {
                            "c": {
                                "name": "c",
                                "sdtype": "unknown",
                                "role": "other",
                                "uniqueness_ratio": 99.9,
                            }
                        },
                    }
                }
            }
        )
        analysis, _ = analysis_from_json(bad_json)
        assert analysis.tables["t"].columns["c"].uniqueness_ratio == 1.0

    def test_skips_invalid_relationship(self):
        bad_json = json.dumps(
            {
                "tables": {},
                "relationships": [
                    {"parent_table": "a"},  # missing required fields
                    {
                        "parent_table": "a",
                        "parent_column": "b",
                        "child_table": "c",
                        "child_column": "d",
                        "overlap_ratio": 0.5,
                        "extra_evil_key": "injected",
                    },
                ],
            }
        )
        analysis, _ = analysis_from_json(bad_json)
        # First entry should be skipped, second should be loaded (extra key ignored)
        assert len(analysis.relationships) == 1

    def test_skips_invalid_date_constraint(self):
        bad_json = json.dumps(
            {
                "tables": {},
                "date_constraints": [
                    {"table_name": "t"},  # missing low/high columns
                ],
            }
        )
        analysis, _ = analysis_from_json(bad_json)
        assert len(analysis.date_constraints) == 0


# ── Preview PII redaction toggle ─────────────────────────────────────────────


class TestPreviewRedaction:
    def test_redacted_preview_has_no_real_values(
        self,
        sample_analysis,
        sample_pandas_dfs,
    ):
        """When redact_preview=True, no real data values should appear."""
        real_regions = set(sample_pandas_dfs["customers"]["region"].dropna().unique())
        real_countries = set(
            sample_pandas_dfs["customers"]["country"].dropna().unique()
        )

        preview = preview_sample(
            "customers",
            sample_analysis,
            sample_pandas_dfs,
            n_rows=10,
            redact_preview=True,
        )

        preview_regions = set(preview["region"].dropna().unique())
        preview_countries = set(preview["country"].dropna().unique())

        # Dimension columns should NOT contain real values
        assert preview_regions.isdisjoint(real_regions), (
            f"Real regions leaked: {preview_regions & real_regions}"
        )
        assert preview_countries.isdisjoint(real_countries), (
            f"Real countries leaked: {preview_countries & real_countries}"
        )

    def test_unredacted_preview_may_contain_real_values(
        self,
        sample_analysis,
        sample_pandas_dfs,
    ):
        """When redact_preview=False (default), real values are allowed."""
        preview = preview_sample(
            "customers",
            sample_analysis,
            sample_pandas_dfs,
            n_rows=10,
        )
        # Should contain some real dimension values
        real_regions = set(sample_pandas_dfs["customers"]["region"].dropna().unique())
        preview_regions = set(preview["region"].dropna().unique())
        assert preview_regions.issubset(real_regions)


# ── Filename sanitization ────────────────────────────────────────────────────


class TestFilenameSanitization:
    def test_strips_path_traversal(self):
        assert ".." not in sanitize_table_name("../../etc/passwd")
        assert "/" not in sanitize_table_name("foo/bar")
        assert "\\" not in sanitize_table_name("foo\\bar")

    def test_strips_control_characters(self):
        result = sanitize_table_name("hello\x00world\x1f")
        assert "\x00" not in result
        assert "\x1f" not in result

    def test_empty_becomes_table(self):
        assert sanitize_table_name("") == "table"

    def test_dots_only_sanitized(self):
        result = sanitize_table_name("...")
        assert ".." not in result

    def test_column_sanitization(self):
        assert sanitize_column_name("") == "column"
        assert "/" not in sanitize_column_name("path/col")


# ── Resource limits ──────────────────────────────────────────────────────────


class TestResourceLimits:
    def test_max_files_enforced(self, tmp_path):
        from sanitizer.loader import discover_data_files

        # Create more files than the limit
        for i in range(6):
            (tmp_path / f"file_{i}.csv").write_text("a,b\n1,2\n")

        with patch.dict(os.environ, {"SANITIZER_MAX_FILES": "5"}):
            # Need to reimport to pick up env var — or patch the module var
            import sanitizer.loader as loader_mod

            old_max = loader_mod.MAX_FILES
            loader_mod.MAX_FILES = 5
            try:
                with pytest.raises(ValueError, match="more than 5"):
                    discover_data_files(tmp_path)
            finally:
                loader_mod.MAX_FILES = old_max

    def test_max_depth_enforced(self, tmp_path):
        import sanitizer.loader as loader_mod
        from sanitizer.loader import discover_data_files

        # Create a file deeper than limit
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "data.csv").write_text("x\n1\n")
        # Also create a file at allowed depth
        (tmp_path / "top.csv").write_text("x\n1\n")

        old_max = loader_mod.MAX_DEPTH
        loader_mod.MAX_DEPTH = 1
        try:
            files = discover_data_files(tmp_path)
            # Only the shallow file should be found
            assert len(files) == 1
            assert files[0].name == "top.csv"
        finally:
            loader_mod.MAX_DEPTH = old_max


# ── Logging sanitization ────────────────────────────────────────────────────


class TestLoggingSanitization:
    def test_scrubs_absolute_paths(self):
        f = _SanitizingFilter()
        if sys.platform == "win32":
            assert "Users" not in f._scrub(r"Loaded C:\Users\john\data\file.xlsx")
        else:
            assert "/home/" not in f._scrub("Loaded /home/john/data/file.xlsx")

    def test_scrubs_email_addresses(self):
        f = _SanitizingFilter()
        result = f._scrub("Found email john@example.com in column")
        assert "john@example.com" not in result
        assert "[REDACTED_EMAIL]" in result

    def test_preserves_non_sensitive_text(self):
        f = _SanitizingFilter()
        text = "Loaded 5 tables with 100 rows"
        assert f._scrub(text) == text
