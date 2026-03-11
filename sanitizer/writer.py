from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sanitizer.models import AnalysisResult

logger = logging.getLogger(__name__)

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
    output_folder = output_folder.resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    rel_dirs = table_relative_dirs or {}
    paths: list[Path] = []
    for table_name, df in synthetic_data.items():
        safe_name = _safe_filename(table_name)
        rel_dir = rel_dirs.get(table_name, "")
        # Prevent path traversal
        if ".." in rel_dir:
            logger.warning("Skipping table %s: relative dir contains '..'", table_name)
            continue
        if rel_dir:
            target_dir = (output_folder / rel_dir).resolve()
            try:
                target_dir.relative_to(output_folder)
            except ValueError:
                logger.warning(
                    "Skipping table %s: resolved path escapes output folder", table_name
                )
                continue
            # Reject symlinks that could redirect writes outside the output folder
            if target_dir.exists() and target_dir.is_symlink():
                logger.warning(
                    "Skipping table %s: target directory is a symlink", table_name
                )
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = output_folder
        path = target_dir / f"{safe_name}.xlsx"
        df.to_excel(path, index=False, engine="openpyxl")
        paths.append(path)
        logger.info("Wrote %s (%d rows, %d cols)", path.name, len(df), len(df.columns))
    logger.info("Wrote %d file(s) to %s", len(paths), output_folder)
    return paths


def write_csv_files(
    synthetic_data: dict[str, pd.DataFrame],
    output_folder: Path,
    table_relative_dirs: dict[str, str] | None = None,
) -> list[Path]:
    """Write synthetic tables to .csv files."""
    output_folder = output_folder.resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    rel_dirs = table_relative_dirs or {}
    paths: list[Path] = []
    for table_name, df in synthetic_data.items():
        safe_name = _safe_filename(table_name)
        rel_dir = rel_dirs.get(table_name, "")
        if ".." in rel_dir:
            logger.warning("Skipping table %s: relative dir contains '..'", table_name)
            continue
        if rel_dir:
            target_dir = (output_folder / rel_dir).resolve()
            try:
                target_dir.relative_to(output_folder)
            except ValueError:
                logger.warning(
                    "Skipping table %s: resolved path escapes output folder", table_name
                )
                continue
            if target_dir.exists() and target_dir.is_symlink():
                logger.warning(
                    "Skipping table %s: target directory is a symlink", table_name
                )
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = output_folder
        path = target_dir / f"{safe_name}.csv"
        df.to_csv(path, index=False)
        paths.append(path)
        logger.info("Wrote %s (%d rows, %d cols)", path.name, len(df), len(df.columns))
    logger.info("Wrote %d CSV file(s) to %s", len(paths), output_folder)
    return paths


def dataframe_to_excel_buffer(df: pd.DataFrame) -> io.BytesIO:
    """Write a single DataFrame to an in-memory Excel buffer."""
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf


def build_manifest(
    synthetic_data: dict[str, pd.DataFrame],
    analysis: AnalysisResult | None = None,
    scale: float | None = None,
    seed: int | None = None,
) -> dict:
    """Build a synthesis manifest documenting what was generated."""
    tables_info = {}
    for table_name, df in synthetic_data.items():
        info: dict = {"rows": len(df), "columns": len(df.columns)}
        if analysis and table_name in analysis.tables:
            info["original_rows"] = analysis.tables[table_name].row_count
        tables_info[table_name] = info

    manifest: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "tables": tables_info,
        "total_synthetic_rows": sum(len(df) for df in synthetic_data.values()),
    }
    if scale is not None:
        manifest["scale_factor"] = scale
    if seed is not None:
        manifest["seed"] = seed
    if analysis:
        manifest["relationships"] = [
            {
                "parent": f"{r.parent_table}.{r.parent_column}",
                "child": f"{r.child_table}.{r.child_column}",
            }
            for r in analysis.relationships
        ]
        manifest["date_constraints"] = [
            {"table": dc.table_name, "low": dc.low_column, "high": dc.high_column}
            for dc in analysis.date_constraints
        ]
        manifest["dimension_groups"] = [
            {"table": dg.table_name, "columns": dg.column_names}
            for dg in analysis.dimension_groups
        ]
    return manifest


def create_zip_buffer(
    synthetic_data: dict[str, pd.DataFrame],
    table_relative_dirs: dict[str, str] | None = None,
    manifest: dict | None = None,
) -> io.BytesIO:
    """Create a ZIP archive of synthetic tables, preserving relative subdirectory structure."""
    rel_dirs = table_relative_dirs or {}
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for table_name, df in synthetic_data.items():
            safe_name = _safe_filename(table_name)
            rel_dir = rel_dirs.get(table_name, "")
            # Prevent path traversal in zip entries
            if ".." in rel_dir:
                logger.warning(
                    "Skipping table %s in zip: relative dir contains '..'", table_name
                )
                continue
            zip_path = f"{rel_dir}/{safe_name}.xlsx" if rel_dir else f"{safe_name}.xlsx"
            excel_buf = dataframe_to_excel_buffer(df)
            zf.writestr(zip_path, excel_buf.read())
        if manifest:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    zip_buf.seek(0)
    return zip_buf
