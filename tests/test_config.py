"""Tests for the config module (JSON serialization round-trip)."""

from __future__ import annotations

import json

from sanitizer.config import _redact_sample, analysis_from_json, analysis_to_json
from sanitizer.models import ColumnRole, SensitivityLevel


class TestConfigRoundTrip:
    def test_serialize_deserialize(self, sample_analysis):
        settings = {"scale": 1.5, "seed": 42}
        json_str = analysis_to_json(sample_analysis, settings)

        loaded_analysis, loaded_settings = analysis_from_json(json_str)

        # Tables preserved
        assert set(loaded_analysis.tables.keys()) == set(sample_analysis.tables.keys())

        # Primary keys preserved
        for name in sample_analysis.tables:
            assert (
                loaded_analysis.tables[name].primary_key
                == sample_analysis.tables[name].primary_key
            )

        # Column roles preserved
        for name, tm in sample_analysis.tables.items():
            for col_name, col_meta in tm.columns.items():
                loaded_col = loaded_analysis.tables[name].columns[col_name]
                assert loaded_col.role == col_meta.role
                assert loaded_col.sdtype == col_meta.sdtype

        # Relationships preserved
        assert len(loaded_analysis.relationships) == len(sample_analysis.relationships)

        # Dimension groups preserved
        assert len(loaded_analysis.dimension_groups) == len(
            sample_analysis.dimension_groups
        )

        # Settings preserved
        assert loaded_settings["scale"] == 1.5
        assert loaded_settings["seed"] == 42

    def test_invalid_role_falls_back_to_other(self):
        json_str = '{"version": "1.0", "tables": {"t": {"name": "t", "file_path": "", "row_count": 1, "columns": {"c": {"name": "c", "sdtype": "unknown", "role": "nonexistent_role"}}}}}'
        analysis, _ = analysis_from_json(json_str)
        assert analysis.tables["t"].columns["c"].role == ColumnRole.OTHER

    def test_empty_analysis(self):
        from sanitizer.models import AnalysisResult

        analysis = AnalysisResult()
        json_str = analysis_to_json(analysis)
        loaded, _settings = analysis_from_json(json_str)
        assert len(loaded.tables) == 0
        assert len(loaded.relationships) == 0

    def test_extended_settings_round_trip(self, sample_analysis):
        """Locale and threshold settings should survive round-trip."""
        settings = {
            "scale": 2.0,
            "seed": 123,
            "faker_locale": "fr_FR",
            "pk_threshold": 0.90,
            "fk_threshold": 0.70,
            "dim_threshold": 0.10,
        }
        json_str = analysis_to_json(sample_analysis, settings)
        _, loaded_settings = analysis_from_json(json_str)
        assert loaded_settings["faker_locale"] == "fr_FR"
        assert loaded_settings["pk_threshold"] == 0.90
        assert loaded_settings["fk_threshold"] == 0.70
        assert loaded_settings["dim_threshold"] == 0.10

    def test_sample_values_redacted(self, sample_analysis):
        """Config export should redact sample values to prevent PII leakage."""
        json_str = analysis_to_json(sample_analysis)
        data = json.loads(json_str)
        for _table_name, table_data in data["tables"].items():
            for _col_name, col_data in table_data["columns"].items():
                for sv in col_data["sample_values"]:
                    if sv and sv not in ("None", "nan", ""):
                        assert sv == "[REDACTED]"

    def test_file_paths_stripped(self, sample_analysis):
        """Config export should not contain absolute file paths."""
        # Set an absolute path on the table
        sample_analysis.tables["customers"].file_path = "C:/Users/data/secret/test.xlsx"
        json_str = analysis_to_json(sample_analysis)
        data = json.loads(json_str)
        file_path = data["tables"]["customers"]["file_path"]
        assert "Users" not in file_path
        assert "secret" not in file_path
        assert file_path == "test.xlsx"

    def test_sensitivity_preserved(self, sample_analysis):
        """Sensitivity level should be serialized and deserialized."""
        sample_analysis.tables["customers"].columns[
            "email"
        ].sensitivity = SensitivityLevel.LOW
        json_str = analysis_to_json(sample_analysis)
        loaded, _ = analysis_from_json(json_str)
        assert (
            loaded.tables["customers"].columns["email"].sensitivity
            == SensitivityLevel.LOW
        )


class TestRedactSample:
    def test_redacts_normal_string(self):
        assert _redact_sample("john@example.com") == "[REDACTED]"

    def test_short_string(self):
        assert _redact_sample("ab") == "[REDACTED]"

    def test_single_char(self):
        assert _redact_sample("x") == "[REDACTED]"

    def test_none_passthrough(self):
        assert _redact_sample("None") == "None"

    def test_empty_passthrough(self):
        assert _redact_sample("") == ""
