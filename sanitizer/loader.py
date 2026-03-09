from __future__ import annotations

from pathlib import Path

import fastexcel
import pandas as pd
import polars as pl


EXCEL_EXTENSIONS = {".xlsx", ".xls"}


def discover_excel_files(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXCEL_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No Excel files found in {folder}")
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
            table_name = stem
        else:
            safe_sheet = sheet.replace(" ", "_")
            table_name = f"{stem}_{safe_sheet}"
        result[table_name] = df

    return result


def load_all(folder: Path) -> tuple[dict[str, pl.DataFrame], dict[str, pd.DataFrame], dict[str, str]]:
    """Load all Excel files from folder.

    Returns:
        (polars_dfs, pandas_dfs, table_sources) where table_sources maps table_name -> file path string.
    """
    files = discover_excel_files(folder)
    polars_dfs: dict[str, pl.DataFrame] = {}
    table_sources: dict[str, str] = {}

    for file_path in files:
        tables = load_excel_file(file_path)
        for name, df in tables.items():
            polars_dfs[name] = df
            table_sources[name] = str(file_path)

    pandas_dfs = {}
    for name, df in polars_dfs.items():
        pdf = df.to_pandas()
        # SDV requires datetime64[ns]; polars .to_pandas() produces [ms]
        for col in pdf.columns:
            if pd.api.types.is_datetime64_any_dtype(pdf[col]):
                pdf[col] = pdf[col].astype("datetime64[ns]")
        pandas_dfs[name] = pdf
    return polars_dfs, pandas_dfs, table_sources
