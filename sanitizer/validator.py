"""Post-synthesis validation: check PK uniqueness, FK integrity, date constraints, and quality."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sanitizer.models import AnalysisResult, ColumnRole

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Aggregated results from all validation checks."""

    pk_checks: dict[str, PKCheck] = field(default_factory=dict)
    fk_checks: dict[str, FKCheck] = field(default_factory=dict)
    date_checks: dict[str, DateCheck] = field(default_factory=dict)
    row_counts: dict[str, RowCountCheck] = field(default_factory=dict)
    column_stats: dict[str, dict[str, ColumnStats]] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return (
            all(c.passed for c in self.pk_checks.values())
            and all(c.passed for c in self.fk_checks.values())
            and all(c.passed for c in self.date_checks.values())
        )

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        for table, check in self.pk_checks.items():
            status = "PASS" if check.passed else "FAIL"
            lines.append(
                f"[{status}] PK uniqueness: {table}.{check.column} ({check.duplicate_count} duplicates)"
            )
        for key, check in self.fk_checks.items():
            status = "PASS" if check.passed else "FAIL"
            lines.append(
                f"[{status}] FK integrity: {key} ({check.orphan_count} orphaned)"
            )
        for key, check in self.date_checks.items():
            status = "PASS" if check.passed else "FAIL"
            lines.append(
                f"[{status}] Date constraint: {key} ({check.violation_count} violations)"
            )
        return lines


@dataclass
class PKCheck:
    column: str
    total_rows: int
    duplicate_count: int

    @property
    def passed(self) -> bool:
        return self.duplicate_count == 0


@dataclass
class FKCheck:
    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    total_fk_values: int
    orphan_count: int

    @property
    def passed(self) -> bool:
        return self.orphan_count == 0


@dataclass
class DateCheck:
    table_name: str
    low_column: str
    high_column: str
    total_rows: int
    violation_count: int

    @property
    def passed(self) -> bool:
        return self.violation_count == 0


@dataclass
class RowCountCheck:
    original_rows: int
    synthetic_rows: int
    expected_rows: int
    scale: float


@dataclass
class ColumnStats:
    """Basic descriptive stats for a single column (original vs synthetic)."""

    column_name: str
    orig_mean: float | None = None
    synth_mean: float | None = None
    orig_std: float | None = None
    synth_std: float | None = None
    orig_min: float | None = None
    synth_min: float | None = None
    orig_max: float | None = None
    synth_max: float | None = None
    orig_null_ratio: float | None = None
    synth_null_ratio: float | None = None


def validate_primary_keys(
    analysis: AnalysisResult,
    synthetic_data: dict[str, pd.DataFrame],
) -> dict[str, PKCheck]:
    """Check PK uniqueness for every table that has a primary key."""
    results: dict[str, PKCheck] = {}
    for table_name, table_meta in analysis.tables.items():
        pk = table_meta.primary_key
        if not pk or table_name not in synthetic_data:
            continue
        df = synthetic_data[table_name]
        if pk not in df.columns:
            continue
        total = len(df)
        n_unique = df[pk].nunique()
        dup_count = total - n_unique
        results[table_name] = PKCheck(
            column=pk, total_rows=total, duplicate_count=dup_count
        )
        if dup_count > 0:
            logger.warning(
                "PK validation FAILED: %s.%s has %d duplicates",
                table_name,
                pk,
                dup_count,
            )
        else:
            logger.info(
                "PK validation passed: %s.%s (%d unique rows)", table_name, pk, total
            )
    return results


def validate_foreign_keys(
    analysis: AnalysisResult,
    synthetic_data: dict[str, pd.DataFrame],
) -> dict[str, FKCheck]:
    """Check referential integrity for every relationship."""
    results: dict[str, FKCheck] = {}
    for rel in analysis.relationships:
        key = f"{rel.child_table}.{rel.child_column}->{rel.parent_table}.{rel.parent_column}"
        if (
            rel.parent_table not in synthetic_data
            or rel.child_table not in synthetic_data
        ):
            continue
        parent_df = synthetic_data[rel.parent_table]
        child_df = synthetic_data[rel.child_table]
        if (
            rel.parent_column not in parent_df.columns
            or rel.child_column not in child_df.columns
        ):
            continue

        parent_ids = set(parent_df[rel.parent_column].dropna().values)
        child_fk = child_df[rel.child_column].dropna()
        orphans = child_fk[~child_fk.isin(parent_ids)]
        orphan_count = len(orphans)

        results[key] = FKCheck(
            child_table=rel.child_table,
            child_column=rel.child_column,
            parent_table=rel.parent_table,
            parent_column=rel.parent_column,
            total_fk_values=len(child_fk),
            orphan_count=orphan_count,
        )
        if orphan_count > 0:
            logger.warning(
                "FK validation FAILED: %s has %d orphaned values", key, orphan_count
            )
        else:
            logger.info("FK validation passed: %s (%d values)", key, len(child_fk))
    return results


def validate_date_constraints(
    analysis: AnalysisResult,
    synthetic_data: dict[str, pd.DataFrame],
) -> dict[str, DateCheck]:
    """Check that date ordering constraints are satisfied."""
    results: dict[str, DateCheck] = {}
    for pair in analysis.date_constraints:
        key = f"{pair.table_name}.{pair.low_column}<={pair.high_column}"
        if pair.table_name not in synthetic_data:
            continue
        df = synthetic_data[pair.table_name]
        if pair.low_column not in df.columns or pair.high_column not in df.columns:
            continue

        mask = df[pair.low_column].notna() & df[pair.high_column].notna()
        comparable = df[mask]
        violations = (comparable[pair.low_column] > comparable[pair.high_column]).sum()

        results[key] = DateCheck(
            table_name=pair.table_name,
            low_column=pair.low_column,
            high_column=pair.high_column,
            total_rows=len(comparable),
            violation_count=int(violations),
        )
        if violations > 0:
            logger.warning(
                "Date constraint FAILED: %s has %d violations", key, violations
            )
        else:
            logger.info(
                "Date constraint passed: %s (%d rows checked)", key, len(comparable)
            )
    return results


def validate_row_counts(
    analysis: AnalysisResult,
    synthetic_data: dict[str, pd.DataFrame],
    scale: float,
) -> dict[str, RowCountCheck]:
    """Check row counts match expected scale factor."""
    results: dict[str, RowCountCheck] = {}
    for table_name, table_meta in analysis.tables.items():
        if table_name not in synthetic_data:
            continue
        expected = max(1, int(table_meta.row_count * scale))
        actual = len(synthetic_data[table_name])
        results[table_name] = RowCountCheck(
            original_rows=table_meta.row_count,
            synthetic_rows=actual,
            expected_rows=expected,
            scale=scale,
        )
    return results


def compute_column_stats(
    analysis: AnalysisResult,
    original_data: dict[str, pd.DataFrame],
    synthetic_data: dict[str, pd.DataFrame],
) -> dict[str, dict[str, ColumnStats]]:
    """Compute basic descriptive stats per column for original vs synthetic."""
    all_stats: dict[str, dict[str, ColumnStats]] = {}
    for table_name, table_meta in analysis.tables.items():
        if table_name not in original_data or table_name not in synthetic_data:
            continue
        orig_df = original_data[table_name]
        synth_df = synthetic_data[table_name]
        table_stats: dict[str, ColumnStats] = {}

        for col_name, col_meta in table_meta.columns.items():
            if col_meta.role == ColumnRole.TEXT:
                continue
            if col_name not in orig_df.columns or col_name not in synth_df.columns:
                continue

            stats = ColumnStats(column_name=col_name)

            # Null ratios
            stats.orig_null_ratio = orig_df[col_name].isna().mean()
            stats.synth_null_ratio = synth_df[col_name].isna().mean()

            # Numerical stats
            if col_meta.sdtype == "numerical" or col_meta.role == ColumnRole.MEASURE:
                orig_num = pd.to_numeric(orig_df[col_name], errors="coerce")
                synth_num = pd.to_numeric(synth_df[col_name], errors="coerce")
                stats.orig_mean = _safe_float(orig_num.mean())
                stats.synth_mean = _safe_float(synth_num.mean())
                stats.orig_std = _safe_float(orig_num.std())
                stats.synth_std = _safe_float(synth_num.std())
                stats.orig_min = _safe_float(orig_num.min())
                stats.synth_min = _safe_float(synth_num.min())
                stats.orig_max = _safe_float(orig_num.max())
                stats.synth_max = _safe_float(synth_num.max())

            table_stats[col_name] = stats
        all_stats[table_name] = table_stats
    return all_stats


def validate_all(
    analysis: AnalysisResult,
    synthetic_data: dict[str, pd.DataFrame],
    original_data: dict[str, pd.DataFrame],
    scale: float = 1.0,
) -> ValidationResult:
    """Run all validation checks and return a combined result."""
    logger.info("Running post-synthesis validation...")
    result = ValidationResult(
        pk_checks=validate_primary_keys(analysis, synthetic_data),
        fk_checks=validate_foreign_keys(analysis, synthetic_data),
        date_checks=validate_date_constraints(analysis, synthetic_data),
        row_counts=validate_row_counts(analysis, synthetic_data, scale),
        column_stats=compute_column_stats(analysis, original_data, synthetic_data),
    )
    if result.all_passed:
        logger.info("All validation checks passed")
    else:
        logger.warning("Some validation checks failed — see details above")
    return result


def _safe_float(val) -> float | None:
    """Convert to float, returning None for non-finite values."""
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
