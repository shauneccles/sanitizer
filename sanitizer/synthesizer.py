from __future__ import annotations

import logging
from typing import Callable, Optional

import pandas as pd
from faker import Faker
from sdv.cag import FixedCombinations, Inequality
from sdv.metadata import Metadata
from sdv.multi_table import HMASynthesizer
from sdv.single_table import GaussianCopulaSynthesizer

from sanitizer.models import AnalysisResult, ColumnRole

logger = logging.getLogger(__name__)
fake = Faker()

# Column-name patterns mapped to Faker providers
_FAKER_DISPATCH: list[tuple[list[str], Callable[[], str]]] = [
    (["name", "full_name", "fullname", "customer_name", "employee_name"], fake.name),
    (["first_name", "firstname", "fname"], fake.first_name),
    (["last_name", "lastname", "lname", "surname"], fake.last_name),
    (["email", "email_address"], fake.email),
    (["phone", "phone_number", "telephone", "tel"], fake.phone_number),
    (["address", "street", "street_address"], fake.address),
    (["city"], fake.city),
    (["company", "company_name", "org", "organization"], fake.company),
    (["url", "website", "homepage"], fake.url),
    (
        ["description", "comment", "comments", "note", "notes", "remarks", "detail", "details", "summary", "body", "text", "content", "message"],
        lambda: fake.sentence(nb_words=10),
    ),
]


def _faker_for_column(col_name: str) -> Callable[[], str]:
    col_lower = col_name.lower()
    for patterns, generator in _FAKER_DISPATCH:
        if col_lower in patterns or any(p in col_lower for p in patterns):
            return generator
    return lambda: fake.sentence(nb_words=6)


def _prepare_clean_data(
    pandas_dfs: dict[str, pd.DataFrame],
    text_columns_by_table: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    clean_data = {}
    for table_name, df in pandas_dfs.items():
        drop_cols = text_columns_by_table.get(table_name, [])
        clean_data[table_name] = df.drop(columns=drop_cols, errors="ignore")
    return clean_data


def build_sdv_metadata(
    analysis: AnalysisResult,
    clean_data: dict[str, pd.DataFrame],
) -> Metadata:
    # Use the new Metadata class (not deprecated MultiTableMetadata)
    # infer_keys=None so we supply our own PKs/FKs
    metadata = Metadata.detect_from_dataframes(data=clean_data, infer_keys=None)

    # Set primary keys
    for table_name, table_meta in analysis.tables.items():
        if table_meta.primary_key:
            try:
                metadata.set_primary_key(
                    column_name=table_meta.primary_key,
                    table_name=table_name,
                )
            except Exception as e:
                logger.warning("Failed to set PK %s.%s: %s", table_name, table_meta.primary_key, e)

        # Update column sdtypes for non-text columns that are in clean_data
        for col_name, col_meta in table_meta.columns.items():
            if col_meta.role == ColumnRole.TEXT:
                continue  # dropped from clean_data
            if col_name not in clean_data.get(table_name, pd.DataFrame()).columns:
                continue
            try:
                kwargs = {"sdtype": col_meta.sdtype}
                if col_meta.sdtype == "datetime" and col_meta.datetime_format:
                    kwargs["datetime_format"] = col_meta.datetime_format
                metadata.update_column(
                    column_name=col_name,
                    table_name=table_name,
                    **kwargs,
                )
            except Exception as e:
                logger.warning("Failed to update column %s.%s: %s", table_name, col_name, e)

    # Add relationships
    for rel in analysis.relationships:
        try:
            metadata.add_relationship(
                parent_table_name=rel.parent_table,
                child_table_name=rel.child_table,
                parent_primary_key=rel.parent_column,
                child_foreign_key=rel.child_column,
            )
        except Exception as e:
            logger.warning("Failed to add relationship %s->%s: %s", rel.parent_table, rel.child_table, e)

    return metadata


def build_constraints(analysis: AnalysisResult) -> list:
    constraints = []
    for pair in analysis.date_constraints:
        constraints.append(Inequality(
            low_column_name=pair.low_column,
            high_column_name=pair.high_column,
            strict_boundaries=pair.strict,
            table_name=pair.table_name,
        ))
    for group in analysis.dimension_groups:
        constraints.append(FixedCombinations(
            column_names=group.column_names,
            table_name=group.table_name,
        ))
    return constraints


def _get_text_columns_by_table(analysis: AnalysisResult) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for table_name, table_meta in analysis.tables.items():
        text_cols = [
            col_name for col_name, col_meta in table_meta.columns.items()
            if col_meta.role == ColumnRole.TEXT
        ]
        if text_cols:
            result[table_name] = text_cols
    return result


def postprocess_text_columns(
    analysis: AnalysisResult,
    synthetic_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    for table_name, table_meta in analysis.tables.items():
        if table_name not in synthetic_data:
            continue
        df = synthetic_data[table_name]
        for col_name, col_meta in table_meta.columns.items():
            if col_meta.role != ColumnRole.TEXT:
                continue
            generator = _faker_for_column(col_name)
            df[col_name] = [generator() for _ in range(len(df))]
        synthetic_data[table_name] = df
    return synthetic_data


def synthesize(
    analysis: AnalysisResult,
    pandas_dfs: dict[str, pd.DataFrame],
    scale: float = 1.0,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict[str, pd.DataFrame]:
    def _progress(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    _progress(0.05, "Identifying text columns...")
    text_columns_by_table = _get_text_columns_by_table(analysis)

    _progress(0.08, "Preparing data...")
    clean_data = _prepare_clean_data(pandas_dfs, text_columns_by_table)

    _progress(0.10, "Building SDV metadata...")
    metadata = build_sdv_metadata(analysis, clean_data)

    _progress(0.15, "Building constraints...")
    constraints = build_constraints(analysis)

    _progress(0.20, "Validating metadata...")
    try:
        metadata.validate()
    except Exception as e:
        logger.warning("Metadata validation warning: %s", e)

    has_relationships = len(analysis.relationships) > 0
    n_tables = len(analysis.tables)

    if has_relationships and n_tables <= 10:
        _progress(0.25, "Fitting multi-table synthesizer (HMA)...")
        try:
            synthetic_data = _synthesize_multi_table(
                metadata, clean_data, constraints, scale, _progress,
            )
        except Exception as e:
            logger.warning("HMA synthesis failed (%s), falling back to per-table synthesis", e)
            _progress(0.30, "Falling back to per-table synthesis...")
            synthetic_data = _synthesize_single_tables(
                metadata, clean_data, constraints, scale, _progress,
            )
    else:
        _progress(0.25, "Fitting per-table synthesizers...")
        synthetic_data = _synthesize_single_tables(
            metadata, clean_data, constraints, scale, _progress,
        )

    _progress(0.88, "Stitching foreign key references...")
    synthetic_data = _stitch_foreign_keys(analysis, synthetic_data)

    _progress(0.92, "Post-processing text columns with Faker...")
    synthetic_data = postprocess_text_columns(analysis, synthetic_data)

    _progress(1.0, "Done!")
    return synthetic_data


def _stitch_foreign_keys(
    analysis: AnalysisResult,
    synthetic_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Replace FK columns in child tables with valid parent PK values."""
    import numpy as np

    for rel in analysis.relationships:
        parent_table = rel.parent_table
        child_table = rel.child_table
        parent_col = rel.parent_column
        child_col = rel.child_column

        if parent_table not in synthetic_data or child_table not in synthetic_data:
            continue
        if child_col not in synthetic_data[child_table].columns:
            continue
        if parent_col not in synthetic_data[parent_table].columns:
            continue

        parent_ids = synthetic_data[parent_table][parent_col].dropna().values
        if len(parent_ids) == 0:
            continue

        n_child = len(synthetic_data[child_table])
        synthetic_data[child_table][child_col] = np.random.choice(
            parent_ids, size=n_child, replace=True,
        )

    return synthetic_data


def _synthesize_multi_table(
    metadata: Metadata,
    clean_data: dict[str, pd.DataFrame],
    constraints: list,
    scale: float,
    _progress: Callable[[float, str], None],
) -> dict[str, pd.DataFrame]:
    synth = HMASynthesizer(metadata=metadata, verbose=True)
    if constraints:
        try:
            synth.add_constraints(constraints)
        except Exception as e:
            logger.warning("Failed to add constraints: %s", e)
    _progress(0.35, "Fitting HMA synthesizer (this may take a while)...")
    synth.fit(clean_data)
    _progress(0.75, "Sampling synthetic data...")
    return synth.sample(scale=scale)


def _synthesize_single_tables(
    metadata: Metadata,
    clean_data: dict[str, pd.DataFrame],
    constraints: list,
    scale: float,
    _progress: Callable[[float, str], None],
) -> dict[str, pd.DataFrame]:
    table_names = list(clean_data.keys())
    synthetic_data: dict[str, pd.DataFrame] = {}

    for i, table_name in enumerate(table_names):
        pct = 0.30 + 0.55 * (i / max(len(table_names), 1))
        _progress(pct, f"Synthesizing {table_name}...")

        table_meta = metadata.get_table_metadata(table_name)
        df = clean_data[table_name]
        n_rows = max(1, int(len(df) * scale))

        synth = GaussianCopulaSynthesizer(metadata=table_meta)
        # Add table-specific constraints
        table_constraints = [
            c for c in constraints
            if getattr(c, "_table_name", None) == table_name
        ]
        if table_constraints:
            try:
                synth.add_constraints(table_constraints)
            except Exception as e:
                logger.warning("Failed to add constraints for %s: %s", table_name, e)

        synth.fit(df)
        synthetic_data[table_name] = synth.sample(num_rows=n_rows)

    return synthetic_data
