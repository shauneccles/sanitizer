from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ColumnRole(str, Enum):
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    DIMENSION = "dimension"
    MEASURE = "measure"
    TEXT = "text"
    DATE = "date"
    OTHER = "other"


@dataclass
class ColumnMeta:
    name: str
    sdtype: str  # SDV sdtype: "id", "numerical", "datetime", "categorical", "boolean", "unknown"
    role: ColumnRole
    is_primary_key: bool = False
    foreign_key_target: str | None = None  # "parent_table.column" if FK
    uniqueness_ratio: float = 0.0
    datetime_format: str | None = None  # e.g. "%Y-%m-%d" for SDV
    faker_override: str | None = None  # explicit Faker provider key, e.g. "email", "name"
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class DateConstraintPair:
    table_name: str
    low_column: str
    high_column: str
    strict: bool = False  # False => low <= high; True => low < high
    violation_count: int = 0


@dataclass
class Relationship:
    parent_table: str
    parent_column: str
    child_table: str
    child_column: str
    overlap_ratio: float = 0.0


@dataclass
class DimensionGroup:
    table_name: str
    column_names: list[str] = field(default_factory=list)
    combination_count: int = 0


@dataclass
class TableMeta:
    name: str
    file_path: str
    row_count: int
    columns: dict[str, ColumnMeta] = field(default_factory=dict)
    primary_key: str | None = None
    relative_dir: str = ""  # subdirectory relative to the root input folder (posix style)


@dataclass
class AnalysisResult:
    tables: dict[str, TableMeta] = field(default_factory=dict)
    relationships: list[Relationship] = field(default_factory=list)
    date_constraints: list[DateConstraintPair] = field(default_factory=list)
    dimension_groups: list[DimensionGroup] = field(default_factory=list)
