from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

import fastexcel
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

EXCEL_EXTENSIONS = {".xlsx", ".xls"}
CSV_EXTENSIONS = {".csv", ".tsv"}
ALL_EXTENSIONS = EXCEL_EXTENSIONS | CSV_EXTENSIONS

# Configurable resource limits (via environment variables)
MAX_DEPTH = int(os.environ.get("SANITIZER_MAX_DEPTH", "10"))
MAX_FILES = int(os.environ.get("SANITIZER_MAX_FILES", "500"))
DISCOVERY_TIMEOUT = int(os.environ.get("SANITIZER_DISCOVERY_TIMEOUT", "60"))
FILE_SIZE_WARN_MB = 200
FILE_SIZE_REJECT_MB = int(os.environ.get("SANITIZER_MAX_FILE_MB", "500"))
MAX_TOTAL_MB = int(os.environ.get("SANITIZER_MAX_TOTAL_MB", "2000"))

# Characters unsafe for filenames on Windows/Linux/macOS
_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_table_name(name: str) -> str:
    """Remove characters that are unsafe for filenames and path traversal sequences."""
    name = name.replace("..", "").replace("/", "_").replace("\\", "_")
    return _UNSAFE_FILENAME_RE.sub("_", name).strip("_") or "table"


def sanitize_column_name(name: str) -> str:
    """Sanitize a column name, preserving underscores and alphanumerics."""
    return _UNSAFE_FILENAME_RE.sub("_", name).strip("_") or "column"


def discover_data_files(folder: Path) -> list[Path]:
    """Recursively find data files under *folder* with configurable safety limits.

    Limits (all configurable via env vars):
      - SANITIZER_MAX_DEPTH: max directory depth (default 10)
      - SANITIZER_MAX_FILES: max file count (default 500)
      - SANITIZER_DISCOVERY_TIMEOUT: seconds before abort (default 60)
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    folder = folder.resolve()
    files: list[Path] = []
    start = time.monotonic()

    for p in folder.rglob("*"):
        # Timeout check
        if time.monotonic() - start > DISCOVERY_TIMEOUT:
            raise TimeoutError(
                f"File discovery timed out after {DISCOVERY_TIMEOUT}s. "
                f"Found {len(files)} file(s) so far. Reduce folder scope or increase SANITIZER_DISCOVERY_TIMEOUT."
            )
        if not p.is_file():
            continue
        if p.suffix.lower() not in ALL_EXTENSIONS:
            continue
        # Depth check
        try:
            depth = len(p.relative_to(folder).parts) - 1  # -1 for the file itself
        except ValueError:
            continue
        if depth > MAX_DEPTH:
            logger.debug(
                "Skipping %s: depth %d exceeds limit %d", p.name, depth, MAX_DEPTH
            )
            continue
        files.append(p)
        if len(files) > MAX_FILES:
            raise ValueError(
                f"Found more than {MAX_FILES} data files. Reduce folder scope or increase SANITIZER_MAX_FILES."
            )

    files.sort()
    if not files:
        raise ValueError(
            f"No data files found in {folder} (searched for .xlsx, .xls, .csv, .tsv)"
        )
    logger.info("Discovered %d data file(s) in %s", len(files), folder)
    return files


# Keep old name as alias for backward compatibility
discover_excel_files = discover_data_files


def load_csv_file(path: Path) -> dict[str, pl.DataFrame]:
    """Load a single CSV/TSV file as a table. Table name = filename stem."""
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pl.read_csv(path, separator=separator, infer_schema_length=10000)
    if df.height == 0:
        logger.info("Skipping empty CSV file: %s", path.name)
        return {}
    table_name = sanitize_table_name(path.stem)
    return {table_name: df}


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
            logger.info("Skipping empty sheet '%s' in %s", sheet, path.name)
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
) -> tuple[
    dict[str, pl.DataFrame], dict[str, pd.DataFrame], dict[str, str], dict[str, str]
]:
    """Load all Excel files from folder (recursively).

    Returns:
        (polars_dfs, pandas_dfs, table_sources, table_relative_dirs)
        - table_sources maps table_name -> absolute file path string
        - table_relative_dirs maps table_name -> posix-style relative subdirectory
          from the root folder (empty string for files in the root)
    """
    files = discover_data_files(folder)

    # Aggregate size check
    total_size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    if total_size_mb > MAX_TOTAL_MB:
        raise ValueError(
            f"Total file size ({total_size_mb:.0f} MB) exceeds limit ({MAX_TOTAL_MB} MB). "
            f"Reduce data or increase SANITIZER_MAX_TOTAL_MB."
        )

    polars_dfs: dict[str, pl.DataFrame] = {}
    table_sources: dict[str, str] = {}
    table_relative_dirs: dict[str, str] = {}
    load_errors: dict[str, str] = {}

    for file_path in files:
        # File size check
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > FILE_SIZE_REJECT_MB:
            msg = f"Skipping {file_path.name}: {size_mb:.0f} MB exceeds {FILE_SIZE_REJECT_MB} MB limit"
            logger.warning(msg)
            load_errors[str(file_path)] = msg
            continue
        if size_mb > FILE_SIZE_WARN_MB:
            logger.warning(
                "Large file: %s (%.0f MB) — loading may be slow",
                file_path.name,
                size_mb,
            )

        # Compute relative subdirectory (posix-style for consistency)
        rel_dir = file_path.parent.relative_to(folder).as_posix()
        if rel_dir == ".":
            rel_dir = ""

        try:
            if file_path.suffix.lower() in CSV_EXTENSIONS:
                tables = load_csv_file(file_path)
            else:
                tables = load_excel_file(file_path)
        except Exception as e:
            msg = f"Failed to load {file_path.name}: {e}"
            logger.warning(msg)
            load_errors[str(file_path)] = msg
            continue

        for name, df in tables.items():
            # Sanitize column names and deduplicate
            renamed = {}
            seen_names: set[str] = set()
            for col in df.columns:
                safe = sanitize_column_name(col)
                # Deduplicate: append suffix if another column already maps to this name
                base = safe
                counter = 1
                while safe in seen_names:
                    safe = f"{base}_{counter}"
                    counter += 1
                seen_names.add(safe)
                if safe != col:
                    renamed[col] = safe
            if renamed:
                df = df.rename(renamed)
                logger.info("Sanitized column names in %s: %s", name, renamed)

            # Disambiguate duplicate table names
            if name in polars_dfs:
                original = name
                suffix = 2
                while name in polars_dfs:
                    name = f"{original}_{suffix}"
                    suffix += 1
                logger.warning(
                    "Duplicate table name '%s', renamed to '%s'", original, name
                )
            polars_dfs[name] = df
            table_sources[name] = str(file_path)
            table_relative_dirs[name] = rel_dir

    if load_errors:
        logger.warning(
            "%d file(s) failed to load: %s",
            len(load_errors),
            ", ".join(load_errors.keys()),
        )

    if not polars_dfs:
        raise ValueError(f"No tables could be loaded from {folder}")

    pandas_dfs = {}
    for name, df in polars_dfs.items():
        pdf = df.to_pandas()
        # SDV requires datetime64[ns]; polars .to_pandas() produces [ms]
        for col in pdf.columns:
            if pd.api.types.is_datetime64_any_dtype(pdf[col]):
                pdf[col] = pdf[col].astype("datetime64[ns]")
        pandas_dfs[name] = pdf

    total_rows = sum(df.height for df in polars_dfs.values())
    logger.info(
        "Loaded %d table(s) with %d total rows from %s",
        len(polars_dfs),
        total_rows,
        folder,
    )
    return polars_dfs, pandas_dfs, table_sources, table_relative_dirs
