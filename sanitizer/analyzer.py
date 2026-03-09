from __future__ import annotations

import polars as pl
import pandas as pd

from sanitizer.models import (
    AnalysisResult,
    ColumnMeta,
    ColumnRole,
    DateConstraintPair,
    DimensionGroup,
    Relationship,
    TableMeta,
)

# ── Thresholds ──────────────────────────────────────────────────────────────
PK_UNIQUENESS_THRESHOLD = 0.95
DIMENSION_CARDINALITY_THRESHOLD = 0.05
FK_OVERLAP_THRESHOLD = 0.80
PK_SUFFIX_PATTERNS = ("_id", "_key", "_nr", "_no", "_pk", "_code", "id")


# ── Primary key detection ───────────────────────────────────────────────────

def detect_primary_keys(table_name: str, df: pl.DataFrame) -> list[tuple[str, float]]:
    """Return candidate PK columns as (col_name, uniqueness_ratio) sorted best-first."""
    candidates: list[tuple[str, float]] = []
    n = df.height
    if n == 0:
        return candidates

    for col in df.columns:
        col_lower = col.lower()
        has_pk_suffix = any(col_lower.endswith(s) for s in PK_SUFFIX_PATTERNS)
        null_count = df.select(pl.col(col).is_null().sum()).item()
        if null_count > 0:
            continue
        n_unique = df.select(pl.col(col).n_unique()).item()
        ratio = n_unique / n
        if ratio >= PK_UNIQUENESS_THRESHOLD and has_pk_suffix:
            candidates.append((col, ratio))

    # If no suffix match found, fall back to any column with 100% uniqueness
    if not candidates:
        for col in df.columns:
            null_count = df.select(pl.col(col).is_null().sum()).item()
            if null_count > 0:
                continue
            n_unique = df.select(pl.col(col).n_unique()).item()
            if n_unique == n:
                candidates.append((col, 1.0))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


# ── Foreign key detection ───────────────────────────────────────────────────

def detect_foreign_keys(
    tables: dict[str, pl.DataFrame],
    primary_keys: dict[str, str],
) -> list[Relationship]:
    relationships: list[Relationship] = []
    pk_values_cache: dict[str, set] = {}

    for parent_table, pk_col in primary_keys.items():
        pk_values_cache[parent_table] = set(
            tables[parent_table].select(pl.col(pk_col)).to_series().to_list()
        )

    for child_table, child_df in tables.items():
        for col in child_df.columns:
            for parent_table, pk_col in primary_keys.items():
                if parent_table == child_table:
                    continue
                # Check column name match
                if col != pk_col and col.lower() != f"{parent_table.lower()}_{pk_col.lower()}" and col.lower() != f"{parent_table.lower()}_id":
                    continue
                # Skip if this column is the child's own PK
                if primary_keys.get(child_table) == col:
                    continue

                child_values = set(
                    child_df.select(pl.col(col)).drop_nulls().to_series().to_list()
                )
                if not child_values:
                    continue
                parent_values = pk_values_cache[parent_table]
                overlap = len(child_values & parent_values) / len(child_values)
                if overlap >= FK_OVERLAP_THRESHOLD:
                    relationships.append(Relationship(
                        parent_table=parent_table,
                        parent_column=pk_col,
                        child_table=child_table,
                        child_column=col,
                        overlap_ratio=round(overlap, 3),
                    ))
    return relationships


# ── Dimension detection ─────────────────────────────────────────────────────

def detect_dimensions(
    table_name: str,
    df: pl.DataFrame,
    exclude_cols: set[str] | None = None,
) -> list[str]:
    exclude = exclude_cols or set()
    dims: list[str] = []
    n = df.height
    if n == 0:
        return dims

    # Adaptive threshold: stricter for large tables, more lenient for small ones
    if n < 50:
        threshold = 0.50
    elif n < 200:
        threshold = 0.15
    else:
        threshold = DIMENSION_CARDINALITY_THRESHOLD

    for col in df.columns:
        if col in exclude:
            continue
        dtype = df.schema[col]
        if dtype not in (pl.Utf8, pl.String, pl.Categorical):
            continue
        n_unique = df.select(pl.col(col).n_unique()).item()
        ratio = n_unique / n
        if ratio < threshold:
            dims.append(col)
    return dims


def detect_dimension_groups(
    table_name: str,
    df: pl.DataFrame,
    dimension_cols: list[str],
) -> list[DimensionGroup]:
    groups: list[DimensionGroup] = []
    if len(dimension_cols) < 2:
        return groups

    # Check pairs for functional dependencies
    already_grouped: set[str] = set()
    for i, col_a in enumerate(dimension_cols):
        if col_a in already_grouped:
            continue
        group_cols = [col_a]
        for col_b in dimension_cols[i + 1:]:
            if col_b in already_grouped:
                continue
            n_a = df.select(pl.col(col_a).n_unique()).item()
            n_b = df.select(pl.col(col_b).n_unique()).item()
            n_ab = df.select(pl.struct(col_a, col_b).n_unique()).item()
            # Functional dependency: A -> B or B -> A
            if n_ab == n_a or n_ab == n_b:
                group_cols.append(col_b)

        if len(group_cols) > 1:
            combo_count = df.select(pl.struct(*group_cols).n_unique()).item()
            groups.append(DimensionGroup(
                table_name=table_name,
                column_names=group_cols,
                combination_count=combo_count,
            ))
            already_grouped.update(group_cols)

    return groups


# ── Date constraint detection ───────────────────────────────────────────────

def _is_date_col(df: pl.DataFrame, col: str) -> bool:
    dtype = df.schema[col]
    return dtype in (pl.Date, pl.Datetime, pl.Duration) or (
        "date" in col.lower() or "time" in col.lower()
    )


def detect_date_constraints(table_name: str, df: pl.DataFrame) -> list[DateConstraintPair]:
    date_cols = [c for c in df.columns if _is_date_col(df, c)]
    constraints: list[DateConstraintPair] = []

    for i, col_a in enumerate(date_cols):
        for col_b in date_cols[i + 1:]:
            # Check if col_a <= col_b for all non-null rows
            mask = df.filter(pl.col(col_a).is_not_null() & pl.col(col_b).is_not_null())
            if mask.height == 0:
                continue
            try:
                violations = mask.filter(pl.col(col_a) > pl.col(col_b)).height
            except Exception:
                continue
            if violations <= mask.height * 0.02:  # Allow 2% violation tolerance
                constraints.append(DateConstraintPair(
                    table_name=table_name,
                    low_column=col_a,
                    high_column=col_b,
                    strict=False,
                    violation_count=violations,
                ))
    return constraints


# ── Text column detection ──────────────────────────────────────────────────

def detect_text_columns(
    table_name: str,
    df: pl.DataFrame,
    exclude_cols: set[str] | None = None,
) -> list[str]:
    exclude = exclude_cols or set()
    text_cols: list[str] = []
    n = df.height
    if n == 0:
        return text_cols

    for col in df.columns:
        if col in exclude:
            continue
        dtype = df.schema[col]
        if dtype not in (pl.Utf8, pl.String):
            continue
        n_unique = df.select(pl.col(col).n_unique()).item()
        ratio = n_unique / n
        if ratio >= DIMENSION_CARDINALITY_THRESHOLD:
            # Check average string length
            avg_len = df.select(pl.col(col).str.len_chars().mean()).item()
            if avg_len is not None and avg_len > 3:
                text_cols.append(col)
    return text_cols


# ── Column classification ──────────────────────────────────────────────────

def classify_column(
    col_name: str,
    df: pl.DataFrame,
    pk_col: str | None,
    fk_cols: set[str],
    dimension_cols: set[str],
    text_cols: set[str],
) -> ColumnMeta:
    n = df.height
    n_unique = df.select(pl.col(col_name).n_unique()).item() if n > 0 else 0
    ratio = n_unique / n if n > 0 else 0.0
    sample = df.select(pl.col(col_name)).head(5).to_series().to_list()

    dtype = df.schema[col_name]

    if col_name == pk_col:
        return ColumnMeta(
            name=col_name, sdtype="id", role=ColumnRole.PRIMARY_KEY,
            is_primary_key=True, uniqueness_ratio=ratio, sample_values=sample,
        )
    if col_name in fk_cols:
        return ColumnMeta(
            name=col_name, sdtype="id", role=ColumnRole.FOREIGN_KEY,
            uniqueness_ratio=ratio, sample_values=sample,
        )
    if col_name in dimension_cols:
        return ColumnMeta(
            name=col_name, sdtype="categorical", role=ColumnRole.DIMENSION,
            uniqueness_ratio=ratio, sample_values=sample,
        )
    if dtype in (pl.Date, pl.Datetime):
        dt_fmt = "%Y-%m-%d" if dtype == pl.Date else "%Y-%m-%d %H:%M:%S"
        return ColumnMeta(
            name=col_name, sdtype="datetime", role=ColumnRole.DATE,
            uniqueness_ratio=ratio, datetime_format=dt_fmt,
            sample_values=[str(v) for v in sample],
        )
    if col_name in text_cols:
        return ColumnMeta(
            name=col_name, sdtype="unknown", role=ColumnRole.TEXT,
            uniqueness_ratio=ratio, sample_values=sample,
        )
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64):
        return ColumnMeta(
            name=col_name, sdtype="numerical", role=ColumnRole.MEASURE,
            uniqueness_ratio=ratio, sample_values=sample,
        )
    if dtype == pl.Boolean:
        return ColumnMeta(
            name=col_name, sdtype="boolean", role=ColumnRole.OTHER,
            uniqueness_ratio=ratio, sample_values=sample,
        )
    return ColumnMeta(
        name=col_name, sdtype="categorical", role=ColumnRole.OTHER,
        uniqueness_ratio=ratio, sample_values=sample,
    )


# ── Main analysis ──────────────────────────────────────────────────────────

def analyze(
    polars_dfs: dict[str, pl.DataFrame],
    pandas_dfs: dict[str, pd.DataFrame],
    table_sources: dict[str, str],
) -> AnalysisResult:
    result = AnalysisResult()

    # Phase 1: Detect primary keys for each table
    primary_keys: dict[str, str] = {}
    for table_name, df in polars_dfs.items():
        candidates = detect_primary_keys(table_name, df)
        pk_col = candidates[0][0] if candidates else None
        primary_keys[table_name] = pk_col  # type: ignore

    # Phase 2: Detect foreign keys across tables
    valid_pks = {t: pk for t, pk in primary_keys.items() if pk is not None}
    relationships = detect_foreign_keys(polars_dfs, valid_pks)
    result.relationships = relationships

    # Build FK lookup per table
    fk_by_table: dict[str, set[str]] = {}
    for rel in relationships:
        fk_by_table.setdefault(rel.child_table, set()).add(rel.child_column)

    # Phase 3: Per-table analysis
    for table_name, df in polars_dfs.items():
        pk_col = primary_keys.get(table_name)
        fk_cols = fk_by_table.get(table_name, set())
        key_cols = fk_cols | ({pk_col} if pk_col else set())

        dimensions = detect_dimensions(table_name, df, exclude_cols=key_cols)
        dim_groups = detect_dimension_groups(table_name, df, dimensions)
        date_pairs = detect_date_constraints(table_name, df)
        text_cols_list = detect_text_columns(table_name, df, exclude_cols=key_cols | set(dimensions))

        result.dimension_groups.extend(dim_groups)
        result.date_constraints.extend(date_pairs)

        columns: dict[str, ColumnMeta] = {}
        for col in df.columns:
            meta = classify_column(
                col, df, pk_col, fk_cols, set(dimensions), set(text_cols_list),
            )
            # Fill FK target info
            if meta.role == ColumnRole.FOREIGN_KEY:
                for rel in relationships:
                    if rel.child_table == table_name and rel.child_column == col:
                        meta.foreign_key_target = f"{rel.parent_table}.{rel.parent_column}"
                        break
            columns[col] = meta

        result.tables[table_name] = TableMeta(
            name=table_name,
            file_path=table_sources.get(table_name, ""),
            row_count=df.height,
            columns=columns,
            primary_key=pk_col,
        )

    return result
