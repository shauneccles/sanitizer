"""JSON serialization for analysis configs — save and reload metadata across sessions."""

from __future__ import annotations

import json
import logging

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

CONFIG_VERSION = "1.0"


def _redact_sample(value: object) -> str:
    """Redact a sample value to prevent PII leakage in config exports."""
    s = str(value)
    if not s or s in ("None", "nan", ""):
        return s
    return "[REDACTED]"


def analysis_to_json(
    analysis: AnalysisResult,
    settings: dict | None = None,
) -> str:
    """Serialize an AnalysisResult (and optional settings) to a JSON string.

    Sample values are redacted to prevent PII leakage. Absolute file paths
    are stripped — only the filename is stored.
    """
    data: dict = {
        "version": CONFIG_VERSION,
        "tables": {},
        "relationships": [],
        "date_constraints": [],
        "dimension_groups": [],
    }

    from pathlib import PurePosixPath, PureWindowsPath

    for name, tm in analysis.tables.items():
        # Strip absolute paths — store only the filename
        try:
            safe_path = (
                PureWindowsPath(tm.file_path).name or PurePosixPath(tm.file_path).name
            )
        except Exception:
            safe_path = "unknown"

        data["tables"][name] = {
            "name": tm.name,
            "file_path": safe_path,
            "row_count": tm.row_count,
            "primary_key": tm.primary_key,
            "relative_dir": tm.relative_dir,
            "columns": {
                cn: {
                    "name": cm.name,
                    "sdtype": cm.sdtype,
                    "role": cm.role.value,
                    "is_primary_key": cm.is_primary_key,
                    "foreign_key_target": cm.foreign_key_target,
                    "uniqueness_ratio": cm.uniqueness_ratio,
                    "datetime_format": cm.datetime_format,
                    "faker_override": cm.faker_override,
                    "sample_values": [_redact_sample(v) for v in cm.sample_values],
                    "sensitivity": cm.sensitivity.value
                    if hasattr(cm, "sensitivity")
                    else "none",
                }
                for cn, cm in tm.columns.items()
            },
        }

    for rel in analysis.relationships:
        data["relationships"].append(
            {
                "parent_table": rel.parent_table,
                "parent_column": rel.parent_column,
                "child_table": rel.child_table,
                "child_column": rel.child_column,
                "overlap_ratio": rel.overlap_ratio,
            }
        )

    for dc in analysis.date_constraints:
        data["date_constraints"].append(
            {
                "table_name": dc.table_name,
                "low_column": dc.low_column,
                "high_column": dc.high_column,
                "strict": dc.strict,
                "violation_count": dc.violation_count,
            }
        )

    for dg in analysis.dimension_groups:
        data["dimension_groups"].append(
            {
                "table_name": dg.table_name,
                "column_names": dg.column_names,
                "combination_count": dg.combination_count,
            }
        )

    if settings:
        data["settings"] = settings

    return json.dumps(data, indent=2)


def analysis_from_json(json_str: str) -> tuple[AnalysisResult, dict]:
    """Deserialize an AnalysisResult (and settings) from a JSON string.

    Returns:
        (analysis, settings) where settings is a dict with optional
        'scale' and 'seed' keys.
    """
    data = json.loads(json_str)

    tables: dict[str, TableMeta] = {}
    for name, t in data.get("tables", {}).items():
        columns: dict[str, ColumnMeta] = {}
        for cn, c in t.get("columns", {}).items():
            try:
                role = ColumnRole(c["role"])
            except (ValueError, KeyError):
                logger.warning(
                    "Table %s, column %s: invalid role %r — falling back to OTHER",
                    name,
                    cn,
                    c.get("role"),
                )
                role = ColumnRole.OTHER
            try:
                sensitivity = SensitivityLevel(c.get("sensitivity", "none"))
            except (ValueError, KeyError):
                sensitivity = SensitivityLevel.NONE
            raw_ratio = c.get("uniqueness_ratio", 0.0)
            try:
                uniqueness_ratio = max(0.0, min(1.0, float(raw_ratio)))
            except (TypeError, ValueError):
                uniqueness_ratio = 0.0
            columns[cn] = ColumnMeta(
                name=str(c.get("name", cn)),
                sdtype=str(c.get("sdtype", "unknown")),
                role=role,
                is_primary_key=bool(c.get("is_primary_key", False)),
                foreign_key_target=c.get("foreign_key_target"),
                uniqueness_ratio=uniqueness_ratio,
                datetime_format=c.get("datetime_format"),
                faker_override=c.get("faker_override"),
                sample_values=c.get("sample_values", []),
                sensitivity=sensitivity,
            )
        try:
            row_count = max(0, int(t["row_count"]))
        except (TypeError, ValueError, KeyError):
            logger.warning("Table %s: invalid row_count, defaulting to 0", name)
            row_count = 0
        tables[name] = TableMeta(
            name=str(t.get("name", name)),
            file_path=str(t.get("file_path", "")),
            row_count=row_count,
            columns=columns,
            primary_key=t.get("primary_key"),
            relative_dir=str(t.get("relative_dir", "")),
        )

    relationships = []
    for r in data.get("relationships", []):
        try:
            relationships.append(
                Relationship(
                    parent_table=str(r["parent_table"]),
                    parent_column=str(r["parent_column"]),
                    child_table=str(r["child_table"]),
                    child_column=str(r["child_column"]),
                    overlap_ratio=float(r.get("overlap_ratio", 0.0)),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping invalid relationship entry: %s", exc)

    date_constraints = []
    for d in data.get("date_constraints", []):
        try:
            date_constraints.append(
                DateConstraintPair(
                    table_name=str(d["table_name"]),
                    low_column=str(d["low_column"]),
                    high_column=str(d["high_column"]),
                    strict=bool(d.get("strict", False)),
                    violation_count=int(d.get("violation_count", 0)),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping invalid date constraint entry: %s", exc)

    dimension_groups = []
    for g in data.get("dimension_groups", []):
        try:
            dimension_groups.append(
                DimensionGroup(
                    table_name=str(g["table_name"]),
                    column_names=[str(c) for c in g.get("column_names", [])],
                    combination_count=int(g.get("combination_count", 0)),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping invalid dimension group entry: %s", exc)

    analysis = AnalysisResult(
        tables=tables,
        relationships=relationships,
        date_constraints=date_constraints,
        dimension_groups=dimension_groups,
    )
    settings = data.get("settings", {})
    return analysis, settings
