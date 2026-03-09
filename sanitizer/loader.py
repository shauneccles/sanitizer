from __future__ import annotations

import logging
import re
from pathlib import Path

import fastexcel
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

EXCEL_EXTENSIONS = {".xlsx", ".xls"}

# Characters unsafe for filenames on Windows/Linux/macOS
_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_table_name(name: str) -> str:
    """Remove characters that are unsafe for filenames and path traversal sequences."""
    name = name.replace("..", "").replace("/", "_").replace("\\", "_")
    return _UNSAFE_FILENAME_RE.sub("_", name).strip("_") or "table"


def discover_excel_files(folder: Path) -> list[Path]:
    """Recursively find all Excel files under *folder* and its subdirectories."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXCEL_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No Excel files found in {folder} (searched recursively)")
    return files


def _get_sheet_names(path: Path) -> list[str]:
    wb = fastexcel.read_excel(str(path))
    return wb.sheet_names


def load_excel_file(path: Path) -> dict[str, pl.DataFrame]:
    sheets = _get_sheet_names(path)
    stem = path.stem
    result: dict[str, pl.DataFrame] = {}

    for sheet in sheets:
        df = pl.read_excel(path, sheet_name=sheet)
        if df.height == 0:
            continue
        if len(sheets) == 1:
            table_name = sanitize_table_name(stem)
        else:
            safe_sheet = sheet.replace(" ", "_")
            table_name = sanitize_table_name(f"{stem}_{safe_sheet}")
        result[table_name] = df

    return result


def load_all(
    folder: Path,
) -> tuple[dict[str, pl.DataFrame], dict[str, pd.DataFrame], dict[str, str], dict[str, str]]:
    """Load all Excel files from folder (recursively).

    Returns:
        (polars_dfs, pandas_dfs, table_sources, table_relative_dirs)
        - table_sources maps table_name -> absolute file path string
        - table_relative_dirs maps table_name -> posix-style relative subdirectory
          from the root folder (empty string for files in the root)
    """
    files = discover_excel_files(folder)
    polars_dfs: dict[str, pl.DataFrame] = {}
    table_sources: dict[str, str] = {}
    table_relative_dirs: dict[str, str] = {}

    for file_path in files:
        # Compute relative subdirectory (posix-style for consistency)
        rel_dir = file_path.parent.relative_to(folder).as_posix()
        if rel_dir == ".":
            rel_dir = ""

        tables = load_excel_file(file_path)
        for name, df in tables.items():
            # Disambiguate duplicate table names
            if name in polars_dfs:
                original = name
                suffix = 2
                while name in polars_dfs:
                    name = f"{original}_{suffix}"
                    suffix += 1
                logger.warning("Duplicate table name '%s', renamed to '%s'", original, name)
            polars_dfs[name] = df
            table_sources[name] = str(file_path)
            table_relative_dirs[name] = rel_dir

    pandas_dfs = {}
    for name, df in polars_dfs.items():
        pdf = df.to_pandas()
        # SDV requires datetime64[ns]; polars .to_pandas() produces [ms]
        for col in pdf.columns:
            if pd.api.types.is_datetime64_any_dtype(pdf[col]):
                pdf[col] = pdf[col].astype("datetime64[ns]")
        pandas_dfs[name] = pdf
    return polars_dfs, pandas_dfs, table_sources, table_relative_dirs
