from __future__ import annotations

import logging
import re
from collections import defaultdict

import polars as pl

from sanitizer.models import (
    AnalysisResult,
    ColumnMeta,
    ColumnRole,
    DateConstraintPair,
    DimensionGroup,
    Relationship,
    SensitivityLevel,
    TableMeta,
)

logger = logging.getLogger(__name__)

# ── Default thresholds ─────────────────────────────────────────────────────
DEFAULT_PK_UNIQUENESS_THRESHOLD = 0.95
DEFAULT_DIMENSION_CARDINALITY_THRESHOLD = 0.05
DEFAULT_FK_OVERLAP_THRESHOLD = 0.80
PK_SUFFIX_PATTERNS = ("_id", "_key", "_nr", "_no", "_pk", "_code", "id")


# ── Primary key detection ───────────────────────────────────────────────────


def detect_primary_keys(
    table_name: str,
    df: pl.DataFrame,
    pk_threshold: float = DEFAULT_PK_UNIQUENESS_THRESHOLD,
) -> list[tuple[str, float]]:
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
        if ratio >= pk_threshold and has_pk_suffix:
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
    if candidates:
        logger.info(
            "PK candidates for '%s': %s",
            table_name,
            ", ".join(f"{c[0]} ({c[1]:.1%})" for c in candidates),
        )
    else:
        logger.debug("No PK candidates found for '%s'", table_name)
    return candidates


# ── Foreign key detection ───────────────────────────────────────────────────


def detect_foreign_keys(
    tables: dict[str, pl.DataFrame],
    primary_keys: dict[str, str],
    fk_threshold: float = DEFAULT_FK_OVERLAP_THRESHOLD,
) -> list[Relationship]:
    relationships: list[Relationship] = []
    # Cache both raw and string-cast PK values for cross-type comparison
    pk_values_cache: dict[str, set] = {}
    pk_str_cache: dict[str, set] = {}

    for parent_table, pk_col in primary_keys.items():
        raw_vals = tables[parent_table].select(pl.col(pk_col)).to_series().to_list()
        pk_values_cache[parent_table] = set(raw_vals)
        pk_str_cache[parent_table] = {str(v) for v in raw_vals}

    for child_table, child_df in tables.items():
        for col in child_df.columns:
            for parent_table, pk_col in primary_keys.items():
                if parent_table == child_table:
                    continue
                # Check column name match
                if (
                    col != pk_col
                    and col.lower() != f"{parent_table.lower()}_{pk_col.lower()}"
                    and col.lower() != f"{parent_table.lower()}_id"
                ):
                    continue
                # Skip if this column is the child's own PK
                if primary_keys.get(child_table) == col:
                    continue

                raw_child = (
                    child_df.select(pl.col(col)).drop_nulls().to_series().to_list()
                )
                child_values = set(raw_child)
                if not child_values:
                    continue
                parent_values = pk_values_cache[parent_table]
                overlap = len(child_values & parent_values) / len(child_values)

                # Fall back to string comparison if types differ
                if overlap < fk_threshold:
                    child_str = {str(v) for v in raw_child}
                    parent_str = pk_str_cache[parent_table]
                    str_overlap = len(child_str & parent_str) / len(child_str)
                    if str_overlap > overlap:
                        overlap = str_overlap

                if overlap >= fk_threshold:
                    relationships.append(
                        Relationship(
                            parent_table=parent_table,
                            parent_column=pk_col,
                            child_table=child_table,
                            child_column=col,
                            overlap_ratio=round(overlap, 3),
                        )
                    )
    logger.info("Detected %d foreign key relationship(s)", len(relationships))
    for r in relationships:
        logger.info(
            "  FK: %s.%s -> %s.%s (overlap=%.1f%%)",
            r.child_table,
            r.child_column,
            r.parent_table,
            r.parent_column,
            r.overlap_ratio * 100,
        )
    return relationships


# ── Dimension detection ─────────────────────────────────────────────────────


def detect_dimensions(
    table_name: str,
    df: pl.DataFrame,
    exclude_cols: set[str] | None = None,
    dim_threshold: float = DEFAULT_DIMENSION_CARDINALITY_THRESHOLD,
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
        threshold = dim_threshold

    for col in df.columns:
        if col in exclude:
            continue
        dtype = df.schema[col]
        if dtype not in (pl.String, pl.Categorical):
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

    # Union-Find to handle transitive functional dependencies (A->B, B->C => {A,B,C})
    uf_parent: dict[str, str] = {col: col for col in dimension_cols}

    def _find(x: str) -> str:
        while uf_parent[x] != x:
            uf_parent[x] = uf_parent[uf_parent[x]]  # path compression
            x = uf_parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            uf_parent[ra] = rb

    # Check all pairs for functional dependencies; union those that have one
    for i, col_a in enumerate(dimension_cols):
        for col_b in dimension_cols[i + 1 :]:
            n_a = df.select(pl.col(col_a).n_unique()).item()
            n_b = df.select(pl.col(col_b).n_unique()).item()
            n_ab = df.select(pl.struct(col_a, col_b).n_unique()).item()
            if n_ab in (n_a, n_b):
                _union(col_a, col_b)

    # Collect clusters from union-find roots
    clusters: dict[str, list[str]] = defaultdict(list)
    for col in dimension_cols:
        clusters[_find(col)].append(col)

    for group_cols in clusters.values():
        if len(group_cols) > 1:
            combo_count = df.select(pl.struct(*group_cols).n_unique()).item()
            groups.append(
                DimensionGroup(
                    table_name=table_name,
                    column_names=group_cols,
                    combination_count=combo_count,
                )
            )

    return groups


# ── Date constraint detection ───────────────────────────────────────────────

# Match "date" or "time" only as whole word segments (e.g. "start_date" but not "mandate")
_DATE_NAME_RE = re.compile(
    r"(?:^|_)(?:date|time|timestamp|datetime)(?:_|$)", re.IGNORECASE
)


def _is_date_col(df: pl.DataFrame, col: str) -> bool:
    dtype = df.schema[col]
    if dtype in (pl.Date, pl.Datetime):
        return True
    # Only use name heuristic for string columns that might contain date strings
    return bool(dtype in (pl.String, pl.Categorical) and _DATE_NAME_RE.search(col))


def detect_date_constraints(
    table_name: str, df: pl.DataFrame
) -> list[DateConstraintPair]:
    date_cols = [c for c in df.columns if _is_date_col(df, c)]
    constraints: list[DateConstraintPair] = []

    for i, col_a in enumerate(date_cols):
        for col_b in date_cols[i + 1 :]:
            # Check if col_a <= col_b for all non-null rows
            mask = df.filter(pl.col(col_a).is_not_null() & pl.col(col_b).is_not_null())
            if mask.height == 0:
                continue
            try:
                violations = mask.filter(pl.col(col_a) > pl.col(col_b)).height
            except Exception as e:
                logger.warning(
                    "Date constraint check failed for %s.(%s, %s): %s",
                    table_name,
                    col_a,
                    col_b,
                    e,
                )
                continue
            if violations <= mask.height * 0.02:  # Allow 2% violation tolerance
                constraints.append(
                    DateConstraintPair(
                        table_name=table_name,
                        low_column=col_a,
                        high_column=col_b,
                        strict=False,
                        violation_count=violations,
                    )
                )
    return constraints


# ── Text column detection ──────────────────────────────────────────────────


def detect_text_columns(
    table_name: str,
    df: pl.DataFrame,
    exclude_cols: set[str] | None = None,
    dim_threshold: float = DEFAULT_DIMENSION_CARDINALITY_THRESHOLD,
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
        if dtype not in (pl.String,):
            continue
        n_unique = df.select(pl.col(col).n_unique()).item()
        ratio = n_unique / n
        if ratio >= dim_threshold:
            # Check average string length
            avg_len = df.select(pl.col(col).str.len_chars().mean()).item()
            if avg_len is not None and avg_len > 3:
                text_cols.append(col)
    return text_cols


# ── PII detection ──────────────────────────────────────────────────────────

# Patterns that indicate high-sensitivity PII
_HIGH_PII_PATTERNS = [
    "ssn",
    "social_security",
    "national_id",
    "tax_id",
    "tin",
    "credit_card",
    "card_number",
    "card_num",
    "cc_number",
    "passport",
    "driver_license",
    "drivers_license",
    "license_number",
    "bank_account",
    "account_number",
    "iban",
    "routing_number",
]

# Patterns that indicate moderate PII — matched as segments only (not substrings)
# to avoid false positives like "table_name", "file_name", etc.
_LOW_PII_SEGMENT_PATTERNS = [
    "email",
    "phone",
    "mobile",
    "cell",
    "surname",
    "address",
    "street",
    "zip",
    "zipcode",
    "dob",
    "birthday",
]

# Compound patterns matched as substrings (specific enough to avoid false positives)
_LOW_PII_SUBSTRING_PATTERNS = [
    "e_mail",
    "email_address",
    "phone_number",
    "telephone",
    "first_name",
    "last_name",
    "full_name",
    "person_name",
    "customer_name",
    "employee_name",
    "contact_name",
    "user_name",
    "street_address",
    "home_address",
    "zip_code",
    "postal_code",
    "date_of_birth",
    "birth_date",
    "ip_address",
    "ip_addr",
]

# Exact column names that indicate PII (matched as full column name only)
_LOW_PII_EXACT = {"name", "email"}


def detect_sensitivity(col_name: str) -> SensitivityLevel:
    """Detect likely PII sensitivity from column name patterns."""
    col_lower = col_name.lower().replace(" ", "_")
    segments = set(col_lower.split("_"))
    # Check high-sensitivity patterns (substring + segment)
    for pattern in _HIGH_PII_PATTERNS:
        if pattern in col_lower or pattern in segments:
            return SensitivityLevel.HIGH
    # Check exact column name matches
    if col_lower in _LOW_PII_EXACT:
        return SensitivityLevel.LOW
    # Check low-sensitivity compound patterns (substring match — safe because they're specific)
    for pattern in _LOW_PII_SUBSTRING_PATTERNS:
        if pattern in col_lower:
            return SensitivityLevel.LOW
    # Check low-sensitivity single-word patterns (segment match only)
    for pattern in _LOW_PII_SEGMENT_PATTERNS:
        if pattern in segments:
            return SensitivityLevel.LOW
    return SensitivityLevel.NONE


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
    sensitivity = detect_sensitivity(col_name)

    dtype = df.schema[col_name]

    if col_name == pk_col:
        return ColumnMeta(
            name=col_name,
            sdtype="id",
            role=ColumnRole.PRIMARY_KEY,
            is_primary_key=True,
            uniqueness_ratio=ratio,
            sample_values=sample,
            sensitivity=sensitivity,
        )
    if col_name in fk_cols:
        return ColumnMeta(
            name=col_name,
            sdtype="id",
            role=ColumnRole.FOREIGN_KEY,
            uniqueness_ratio=ratio,
            sample_values=sample,
            sensitivity=sensitivity,
        )
    if col_name in dimension_cols:
        return ColumnMeta(
            name=col_name,
            sdtype="categorical",
            role=ColumnRole.DIMENSION,
            uniqueness_ratio=ratio,
            sample_values=sample,
            sensitivity=sensitivity,
        )
    if dtype in (pl.Date, pl.Datetime):
        dt_fmt = "%Y-%m-%d" if dtype == pl.Date else "%Y-%m-%d %H:%M:%S"
        return ColumnMeta(
            name=col_name,
            sdtype="datetime",
            role=ColumnRole.DATE,
            uniqueness_ratio=ratio,
            datetime_format=dt_fmt,
            sample_values=[str(v) for v in sample],
            sensitivity=sensitivity,
        )
    if col_name in text_cols:
        return ColumnMeta(
            name=col_name,
            sdtype="unknown",
            role=ColumnRole.TEXT,
            uniqueness_ratio=ratio,
            sample_values=sample,
            sensitivity=sensitivity,
        )
    if dtype in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ):
        return ColumnMeta(
            name=col_name,
            sdtype="numerical",
            role=ColumnRole.MEASURE,
            uniqueness_ratio=ratio,
            sample_values=sample,
            sensitivity=sensitivity,
        )
    if dtype == pl.Boolean:
        return ColumnMeta(
            name=col_name,
            sdtype="boolean",
            role=ColumnRole.OTHER,
            uniqueness_ratio=ratio,
            sample_values=sample,
            sensitivity=sensitivity,
        )
    return ColumnMeta(
        name=col_name,
        sdtype="categorical",
        role=ColumnRole.OTHER,
        uniqueness_ratio=ratio,
        sample_values=sample,
        sensitivity=sensitivity,
    )


# ── Main analysis ──────────────────────────────────────────────────────────


def analyze(
    polars_dfs: dict[str, pl.DataFrame],
    table_sources: dict[str, str],
    table_relative_dirs: dict[str, str] | None = None,
    pk_threshold: float | None = None,
    fk_threshold: float | None = None,
    dim_threshold: float | None = None,
) -> AnalysisResult:
    # Resolve thresholds: use provided values or module defaults
    pk_thresh = (
        pk_threshold if pk_threshold is not None else DEFAULT_PK_UNIQUENESS_THRESHOLD
    )
    fk_thresh = (
        fk_threshold if fk_threshold is not None else DEFAULT_FK_OVERLAP_THRESHOLD
    )
    dim_thresh = (
        dim_threshold
        if dim_threshold is not None
        else DEFAULT_DIMENSION_CARDINALITY_THRESHOLD
    )

    logger.info(
        "Starting analysis of %d table(s) (thresholds: PK=%.0f%%, FK=%.0f%%, dim=%.0f%%)",
        len(polars_dfs),
        pk_thresh * 100,
        fk_thresh * 100,
        dim_thresh * 100,
    )
    result = AnalysisResult()

    # Phase 1: Detect primary keys for each table (only store when found)
    primary_keys: dict[str, str] = {}
    for table_name, df in polars_dfs.items():
        candidates = detect_primary_keys(table_name, df, pk_threshold=pk_thresh)
        if candidates:
            primary_keys[table_name] = candidates[0][0]

    # Phase 2: Detect foreign keys across tables
    relationships = detect_foreign_keys(
        polars_dfs, primary_keys, fk_threshold=fk_thresh
    )
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

        dimensions = detect_dimensions(
            table_name, df, exclude_cols=key_cols, dim_threshold=dim_thresh
        )
        dim_groups = detect_dimension_groups(table_name, df, dimensions)
        date_pairs = detect_date_constraints(table_name, df)
        text_cols_list = detect_text_columns(
            table_name,
            df,
            exclude_cols=key_cols | set(dimensions),
            dim_threshold=dim_thresh,
        )

        result.dimension_groups.extend(dim_groups)
        result.date_constraints.extend(date_pairs)

        columns: dict[str, ColumnMeta] = {}
        for col in df.columns:
            meta = classify_column(
                col,
                df,
                pk_col,
                fk_cols,
                set(dimensions),
                set(text_cols_list),
            )
            # Fill FK target info
            if meta.role == ColumnRole.FOREIGN_KEY:
                for rel in relationships:
                    if rel.child_table == table_name and rel.child_column == col:
                        meta.foreign_key_target = (
                            f"{rel.parent_table}.{rel.parent_column}"
                        )
                        break
            columns[col] = meta

        rel_dirs = table_relative_dirs or {}
        result.tables[table_name] = TableMeta(
            name=table_name,
            file_path=table_sources.get(table_name, ""),
            row_count=df.height,
            columns=columns,
            primary_key=pk_col,
            relative_dir=rel_dirs.get(table_name, ""),
        )

    logger.info(
        "Analysis complete: %d table(s), %d relationship(s), %d date constraint(s), %d dimension group(s)",
        len(result.tables),
        len(result.relationships),
        len(result.date_constraints),
        len(result.dimension_groups),
    )
    return result
