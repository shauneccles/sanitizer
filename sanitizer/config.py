"""JSON serialization for analysis configs — save and reload metadata across sessions."""
from __future__ import annotations

import json

from sanitizer.models import (
    AnalysisResult,
    ColumnMeta,
    ColumnRole,
    DateConstraintPair,
    DimensionGroup,
    Relationship,
    TableMeta,
)

CONFIG_VERSION = "1.0"


def analysis_to_json(
    analysis: AnalysisResult,
    settings: dict | None = None,
) -> str:
    """Serialize an AnalysisResult (and optional settings) to a JSON string."""
    data: dict = {
        "version": CONFIG_VERSION,
        "tables": {},
        "relationships": [],
        "date_constraints": [],
        "dimension_groups": [],
    }

    for name, tm in analysis.tables.items():
        data["tables"][name] = {
            "name": tm.name,
            "file_path": tm.file_path,
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
                    "sample_values": [str(v) for v in cm.sample_values],
                }
                for cn, cm in tm.columns.items()
            },
        }

    for rel in analysis.relationships:
        data["relationships"].append({
            "parent_table": rel.parent_table,
            "parent_column": rel.parent_column,
            "child_table": rel.child_table,
            "child_column": rel.child_column,
            "overlap_ratio": rel.overlap_ratio,
        })

    for dc in analysis.date_constraints:
        data["date_constraints"].append({
            "table_name": dc.table_name,
            "low_column": dc.low_column,
            "high_column": dc.high_column,
            "strict": dc.strict,
            "violation_count": dc.violation_count,
        })

    for dg in analysis.dimension_groups:
        data["dimension_groups"].append({
            "table_name": dg.table_name,
            "column_names": dg.column_names,
            "combination_count": dg.combination_count,
        })

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
            columns[cn] = ColumnMeta(
                name=c["name"],
                sdtype=c["sdtype"],
                role=ColumnRole(c["role"]),
                is_primary_key=c.get("is_primary_key", False),
                foreign_key_target=c.get("foreign_key_target"),
                uniqueness_ratio=c.get("uniqueness_ratio", 0.0),
                datetime_format=c.get("datetime_format"),
                faker_override=c.get("faker_override"),
                sample_values=c.get("sample_values", []),
            )
        tables[name] = TableMeta(
            name=t["name"],
            file_path=t.get("file_path", ""),
            row_count=t["row_count"],
            columns=columns,
            primary_key=t.get("primary_key"),
            relative_dir=t.get("relative_dir", ""),
        )

    relationships = [
        Relationship(**r) for r in data.get("relationships", [])
    ]
    date_constraints = [
        DateConstraintPair(**d) for d in data.get("date_constraints", [])
    ]
    dimension_groups = [
        DimensionGroup(**g) for g in data.get("dimension_groups", [])
    ]

    analysis = AnalysisResult(
        tables=tables,
        relationships=relationships,
        date_constraints=date_constraints,
        dimension_groups=dimension_groups,
    )
    settings = data.get("settings", {})
    return analysis, settings
