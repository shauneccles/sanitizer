"""Tests for the writer module."""

from __future__ import annotations

import json
import zipfile

import pandas as pd

from sanitizer.writer import (
    _safe_filename,
    build_manifest,
    create_zip_buffer,
    write_excel_files,
)


class TestSafeFilename:
    def test_normal_name(self):
        assert _safe_filename("customers") == "customers"

    def test_special_chars(self):
        result = _safe_filename('test<>:"/\\|?*file')
        assert "<" not in result
        assert ">" not in result

    def test_path_traversal(self):
        result = _safe_filename("../../etc/passwd")
        assert ".." not in result

    def test_empty_string(self):
        assert _safe_filename("") == "table"


class TestWriteExcelFiles:
    def test_writes_files(self, tmp_path):
        data = {"t1": pd.DataFrame({"a": [1, 2]})}
        paths = write_excel_files(data, tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".xlsx"

    def test_preserves_subdirs(self, tmp_path):
        data = {"t1": pd.DataFrame({"a": [1]})}
        dirs = {"t1": "sub/folder"}
        paths = write_excel_files(data, tmp_path, dirs)
        assert "sub" in str(paths[0])

    def test_blocks_path_traversal(self, tmp_path):
        data = {"t1": pd.DataFrame({"a": [1]})}
        dirs = {"t1": "../../escape"}
        paths = write_excel_files(data, tmp_path, dirs)
        # Should skip the table due to ".." in path
        assert len(paths) == 0


class TestBuildManifest:
    def test_basic_manifest(self):
        data = {"t1": pd.DataFrame({"a": [1, 2, 3]})}
        m = build_manifest(data, scale=1.0, seed=42)
        assert m["total_synthetic_rows"] == 3
        assert m["scale_factor"] == 1.0
        assert m["seed"] == 42
        assert "generated_at" in m

    def test_manifest_with_analysis(self, sample_analysis):
        data = {
            "customers": pd.DataFrame({"customer_id": range(5)}),
            "orders": pd.DataFrame({"order_id": range(8)}),
        }
        m = build_manifest(data, sample_analysis)
        assert m["tables"]["customers"]["original_rows"] == 5
        assert len(m["relationships"]) == 1


class TestCreateZipBuffer:
    def test_creates_valid_zip(self):
        data = {"t1": pd.DataFrame({"a": [1, 2]})}
        buf = create_zip_buffer(data)
        with zipfile.ZipFile(buf) as zf:
            assert "t1.xlsx" in zf.namelist()

    def test_includes_manifest(self):
        data = {"t1": pd.DataFrame({"a": [1]})}
        manifest = {"test": True}
        buf = create_zip_buffer(data, manifest=manifest)
        with zipfile.ZipFile(buf) as zf:
            assert "manifest.json" in zf.namelist()
            loaded = json.loads(zf.read("manifest.json"))
            assert loaded["test"] is True
