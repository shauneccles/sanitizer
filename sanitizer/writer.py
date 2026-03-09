from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd


def write_excel_files(
    synthetic_data: dict[str, pd.DataFrame],
    output_folder: Path,
) -> list[Path]:
    output_folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for table_name, df in synthetic_data.items():
        path = output_folder / f"{table_name}.xlsx"
        df.to_excel(path, index=False, engine="openpyxl")
        paths.append(path)
    return paths


def write_single_excel_buffer(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf


def create_zip_buffer(synthetic_data: dict[str, pd.DataFrame]) -> io.BytesIO:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for table_name, df in synthetic_data.items():
            excel_buf = write_single_excel_buffer(df)
            zf.writestr(f"{table_name}.xlsx", excel_buf.read())
    zip_buf.seek(0)
    return zip_buf
