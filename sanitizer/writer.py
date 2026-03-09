from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

# Characters unsafe for filenames on Windows/Linux/macOS
_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_filename(name: str) -> str:
    """Sanitize a table name for use as a filename."""
    name = name.replace("..", "").replace("/", "_").replace("\\", "_")
    return _UNSAFE_FILENAME_RE.sub("_", name).strip("_") or "table"


def write_excel_files(
    synthetic_data: dict[str, pd.DataFrame],
    output_folder: Path,
    table_relative_dirs: dict[str, str] | None = None,
) -> list[Path]:
    """Write synthetic tables to .xlsx files, preserving relative subdirectory structure."""
    output_folder.mkdir(parents=True, exist_ok=True)
    rel_dirs = table_relative_dirs or {}
    paths: list[Path] = []
    for table_name, df in synthetic_data.items():
        safe_name = _safe_filename(table_name)
        rel_dir = rel_dirs.get(table_name, "")
        if rel_dir:
            target_dir = output_folder / rel_dir
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = output_folder
        path = target_dir / f"{safe_name}.xlsx"
        df.to_excel(path, index=False, engine="openpyxl")
        paths.append(path)
    return paths


def dataframe_to_excel_buffer(df: pd.DataFrame) -> io.BytesIO:
    """Write a single DataFrame to an in-memory Excel buffer."""
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf


def create_zip_buffer(
    synthetic_data: dict[str, pd.DataFrame],
    table_relative_dirs: dict[str, str] | None = None,
) -> io.BytesIO:
    """Create a ZIP archive of synthetic tables, preserving relative subdirectory structure."""
    rel_dirs = table_relative_dirs or {}
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for table_name, df in synthetic_data.items():
            safe_name = _safe_filename(table_name)
            rel_dir = rel_dirs.get(table_name, "")
            zip_path = f"{rel_dir}/{safe_name}.xlsx" if rel_dir else f"{safe_name}.xlsx"
            excel_buf = dataframe_to_excel_buffer(df)
            zf.writestr(zip_path, excel_buf.read())
    zip_buf.seek(0)
    return zip_buf
