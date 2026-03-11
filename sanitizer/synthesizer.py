from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from faker import Faker
from sdv.cag import FixedCombinations, Inequality
from sdv.metadata import Metadata
from sdv.multi_table import HMASynthesizer
from sdv.single_table import GaussianCopulaSynthesizer

from sanitizer.models import AnalysisResult, ColumnRole

logger = logging.getLogger(__name__)

# Default Faker instance for preview/UI usage (non-synthesis contexts).
# Synthesis creates its own local instance per call for thread safety.
_default_fake = Faker()

# Column-name patterns mapped to Faker provider method names.
_FAKER_DISPATCH: list[tuple[list[str], str]] = [
    (["name", "full_name", "fullname", "customer_name", "employee_name"], "name"),
    (["first_name", "firstname", "fname"], "first_name"),
    (["last_name", "lastname", "lname", "surname"], "last_name"),
    (["email", "email_address"], "email"),
    (["phone", "phone_number", "telephone", "tel"], "phone_number"),
    (["address", "street", "street_address"], "address"),
    (["city"], "city"),
    (["company", "company_name", "org", "organization"], "company"),
    (["url", "website", "homepage"], "url"),
    (
        [
            "description",
            "comment",
            "comments",
            "note",
            "notes",
            "remarks",
            "detail",
            "details",
            "summary",
            "body",
            "text",
            "content",
            "message",
        ],
        "sentence_10",
    ),
]

# Named Faker generator method names for the override dropdown
_FAKER_TYPE_METHODS: dict[str, str] = {
    "name": "name",
    "first_name": "first_name",
    "last_name": "last_name",
    "email": "email",
    "phone": "phone_number",
    "address": "address",
    "city": "city",
    "company": "company",
    "url": "url",
    "sentence": "sentence_10",
    "paragraph": "paragraph",
    "uuid": "uuid4",
    "date": "date",
    "country": "country",
    "job": "job",
    "zipcode": "zipcode",
}


def _resolve_faker_method(fake: Faker, method_name: str) -> Callable[[], str]:
    """Resolve a method name to a callable on the given Faker instance."""
    if method_name == "sentence_10":
        return lambda: fake.sentence(nb_words=10)
    return getattr(fake, method_name)


# Public FAKER_TYPES for the UI — uses the default instance (safe for preview)
FAKER_TYPES: dict[str, Callable[[], str]] = {
    name: _resolve_faker_method(_default_fake, method)
    for name, method in _FAKER_TYPE_METHODS.items()
}

# Supported Faker locales
FAKER_LOCALES: list[str] = [
    "en_US",
    "en_GB",
    "en_AU",
    "en_CA",
    "en_NZ",
    "fr_FR",
    "de_DE",
    "de_AT",
    "de_CH",
    "es_ES",
    "es_MX",
    "it_IT",
    "pt_BR",
    "pt_PT",
    "nl_NL",
    "nl_BE",
    "pl_PL",
    "sv_SE",
    "da_DK",
    "nb_NO",
    "fi_FI",
    "ja_JP",
    "ko_KR",
    "zh_CN",
    "zh_TW",
    "ru_RU",
    "uk_UA",
    "cs_CZ",
    "ro_RO",
    "hu_HU",
    "ar_SA",
    "he_IL",
    "hi_IN",
    "th_TH",
    "vi_VN",
]

# Options list for the UI dropdown
FAKER_TYPE_OPTIONS: list[str] = ["auto", *sorted(FAKER_TYPES.keys())]


def _faker_for_column(
    col_name: str,
    override: str | None = None,
    fake: Faker | None = None,
) -> Callable[[], str]:
    f = fake or _default_fake

    # Check explicit override first
    if override and override != "auto" and override in _FAKER_TYPE_METHODS:
        return _resolve_faker_method(f, _FAKER_TYPE_METHODS[override])

    col_lower = col_name.lower()
    # Split on underscores/spaces to get word segments for matching
    segments = set(col_lower.replace(" ", "_").split("_"))

    # Exact full-name match first, then segment-level match
    for patterns, method_name in _FAKER_DISPATCH:
        if col_lower in patterns:
            return _resolve_faker_method(f, method_name)
        if segments & set(patterns):
            return _resolve_faker_method(f, method_name)
    return lambda: f.sentence(nb_words=6)


def preview_sample(
    table_name: str,
    analysis: AnalysisResult,
    pandas_dfs: dict[str, pd.DataFrame],
    n_rows: int = 10,
    redact_preview: bool = False,
) -> pd.DataFrame:
    """Generate a fast heuristic preview of synthetic data (no SDV fitting).

    Produces structurally plausible rows in <50ms by sampling from source
    data ranges and using Faker for text columns.

    When *redact_preview* is True, dimension/FK/fallback columns use synthetic
    placeholders instead of sampling from real data — useful when the preview
    may be shared or screen-recorded.
    """
    table_meta = analysis.tables[table_name]
    source_df = pandas_dfs.get(table_name, pd.DataFrame())
    result: dict[str, list] = {}

    for col_name, col_meta in table_meta.columns.items():
        role = col_meta.role
        sdtype = col_meta.sdtype
        source_col = source_df[col_name] if col_name in source_df.columns else None

        if role == ColumnRole.PRIMARY_KEY or col_meta.is_primary_key:
            # Sequential IDs
            if source_col is not None and pd.api.types.is_string_dtype(source_col):
                result[col_name] = [_default_fake.uuid4() for _ in range(n_rows)]
            else:
                result[col_name] = list(range(1, n_rows + 1))

        elif role == ColumnRole.FOREIGN_KEY and col_meta.foreign_key_target:
            if redact_preview:
                # Synthetic FK values instead of real parent data
                result[col_name] = list(range(1, n_rows + 1))
            else:
                # Sample from parent PK
                parts = col_meta.foreign_key_target.split(".", 1)
                if len(parts) == 2:
                    parent_table, parent_col = parts
                    parent_df = pandas_dfs.get(parent_table)
                    if parent_df is not None and parent_col in parent_df.columns:
                        parent_vals = parent_df[parent_col].dropna().values
                        if len(parent_vals) > 0:
                            result[col_name] = list(
                                np.random.choice(parent_vals, size=n_rows, replace=True)
                            )
                            continue
                # Fallback: sample from own column
                if source_col is not None and not source_col.dropna().empty:
                    result[col_name] = list(
                        source_col.dropna().sample(n=n_rows, replace=True).values
                    )
                else:
                    result[col_name] = [None] * n_rows

        elif role == ColumnRole.TEXT:
            generator = _faker_for_column(col_name, col_meta.faker_override)
            result[col_name] = [generator() for _ in range(n_rows)]

        elif sdtype == "numerical" or role == ColumnRole.MEASURE:
            if source_col is not None and not source_col.dropna().empty:
                col_min = float(source_col.min())
                col_max = float(source_col.max())
                if pd.api.types.is_integer_dtype(source_col):
                    result[col_name] = list(
                        np.random.randint(
                            int(col_min),
                            max(int(col_max) + 1, int(col_min) + 1),
                            size=n_rows,
                        )
                    )
                else:
                    result[col_name] = list(
                        np.round(np.random.uniform(col_min, col_max, size=n_rows), 2)
                    )
            else:
                result[col_name] = list(np.random.uniform(0, 100, size=n_rows))

        elif sdtype == "datetime" or role == ColumnRole.DATE:
            if source_col is not None and not source_col.dropna().empty:
                try:
                    dates = pd.to_datetime(source_col.dropna())
                    start = dates.min()
                    end = dates.max()
                    delta = (end - start).total_seconds()
                    if delta <= 0:
                        delta = 86400  # 1 day
                    random_offsets = np.random.uniform(0, delta, size=n_rows)
                    result[col_name] = [
                        start + pd.Timedelta(seconds=s) for s in random_offsets
                    ]
                except Exception:
                    result[col_name] = [
                        _default_fake.date_time() for _ in range(n_rows)
                    ]
            else:
                result[col_name] = [_default_fake.date_time() for _ in range(n_rows)]

        elif sdtype == "boolean":
            result[col_name] = list(np.random.choice([True, False], size=n_rows))

        elif sdtype == "categorical" or role == ColumnRole.DIMENSION:
            if redact_preview:
                n_cats = (
                    source_col.nunique()
                    if source_col is not None and not source_col.dropna().empty
                    else 5
                )
                synth_cats = [f"Category_{i + 1}" for i in range(n_cats)]
                result[col_name] = list(
                    np.random.choice(synth_cats, size=n_rows, replace=True)
                )
            else:
                if source_col is not None and not source_col.dropna().empty:
                    unique_vals = source_col.dropna().unique()
                    result[col_name] = list(
                        np.random.choice(unique_vals, size=n_rows, replace=True)
                    )
                else:
                    result[col_name] = [f"cat_{i}" for i in range(n_rows)]

        else:
            # Fallback: sample from source or generate placeholders
            if redact_preview:
                result[col_name] = [f"value_{i + 1}" for i in range(n_rows)]
            elif source_col is not None and not source_col.dropna().empty:
                result[col_name] = list(
                    source_col.dropna().sample(n=n_rows, replace=True).values
                )
            else:
                result[col_name] = [None] * n_rows

    return pd.DataFrame(result)


def _prepare_clean_data(
    pandas_dfs: dict[str, pd.DataFrame],
    text_columns_by_table: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    clean_data = {}
    for table_name, df in pandas_dfs.items():
        drop_cols = text_columns_by_table.get(table_name, [])
        clean_data[table_name] = df.drop(columns=drop_cols, errors="ignore").copy()
    return clean_data


def _align_fk_types(
    clean_data: dict[str, pd.DataFrame],
    analysis: AnalysisResult,
    warn: Callable[[str], None],
) -> dict[str, pd.DataFrame]:
    """Cast FK columns to match their parent PK column dtype so SDV accepts the relationship."""
    for rel in analysis.relationships:
        parent, child = rel.parent_table, rel.child_table
        pk_col, fk_col = rel.parent_column, rel.child_column

        if parent not in clean_data or child not in clean_data:
            continue
        if (
            pk_col not in clean_data[parent].columns
            or fk_col not in clean_data[child].columns
        ):
            continue

        pk_dtype = clean_data[parent][pk_col].dtype
        fk_dtype = clean_data[child][fk_col].dtype

        if pk_dtype != fk_dtype:
            try:
                clean_data[child][fk_col] = clean_data[child][fk_col].astype(pk_dtype)
                warn(
                    f"Cast {child}.{fk_col} from {fk_dtype} to {pk_dtype} to match parent PK type"
                )
            except (ValueError, TypeError) as e:
                warn(
                    f"Cannot align FK type {child}.{fk_col} ({fk_dtype}) with PK {parent}.{pk_col} ({pk_dtype}): {e}"
                )

    return clean_data


def _fix_date_constraint_violations(
    clean_data: dict[str, pd.DataFrame],
    analysis: AnalysisResult,
    warn: Callable[[str], None],
) -> dict[str, pd.DataFrame]:
    """Swap low/high values on rows that violate date inequality constraints.

    SDV requires zero violations. The analyzer allows 2% tolerance for
    detection, so we fix the small number of violating rows here.
    """
    for pair in analysis.date_constraints:
        table_name = pair.table_name
        if table_name not in clean_data:
            continue
        df = clean_data[table_name]
        low, high = pair.low_column, pair.high_column
        if low not in df.columns or high not in df.columns:
            continue

        mask = df[low].notna() & df[high].notna() & (df[low] > df[high])
        n_violations = int(mask.sum())
        if n_violations > 0:
            # Swap the two columns on violating rows so low <= high
            low_vals = df.loc[mask, low].copy()
            df.loc[mask, low] = df.loc[mask, high]
            df.loc[mask, high] = low_vals
            clean_data[table_name] = df
            warn(
                f"Fixed {n_violations} date constraint violation(s) in "
                f"{table_name}: swapped {low} / {high} on violating rows"
            )

    return clean_data


def build_sdv_metadata(
    analysis: AnalysisResult,
    clean_data: dict[str, pd.DataFrame],
    warn: Callable[[str], None] | None = None,
) -> Metadata:
    def _warn(msg: str):
        logger.warning(msg)
        if warn:
            warn(msg)

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
                _warn(f"Failed to set PK {table_name}.{table_meta.primary_key}: {e}")

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
                _warn(f"Failed to update column {table_name}.{col_name}: {e}")

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
            _warn(
                f"Failed to add relationship {rel.parent_table}->{rel.child_table}: {e}"
            )

    return metadata


def build_constraints(
    analysis: AnalysisResult,
) -> tuple[list, dict[str, list]]:
    """Build SDV constraints.

    Returns:
        (all_constraints, constraints_by_table) — flat list for multi-table
        synthesis and per-table grouping for single-table synthesis.
    """
    all_constraints: list = []
    by_table: dict[str, list] = {}
    for pair in analysis.date_constraints:
        c = Inequality(
            low_column_name=pair.low_column,
            high_column_name=pair.high_column,
            strict_boundaries=pair.strict,
            table_name=pair.table_name,
        )
        all_constraints.append(c)
        by_table.setdefault(pair.table_name, []).append(c)
    for group in analysis.dimension_groups:
        c = FixedCombinations(
            column_names=group.column_names,
            table_name=group.table_name,
        )
        all_constraints.append(c)
        by_table.setdefault(group.table_name, []).append(c)
    return all_constraints, by_table


def _get_text_columns_by_table(analysis: AnalysisResult) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for table_name, table_meta in analysis.tables.items():
        text_cols = [
            col_name
            for col_name, col_meta in table_meta.columns.items()
            if col_meta.role == ColumnRole.TEXT
        ]
        if text_cols:
            result[table_name] = text_cols
    return result


def postprocess_text_columns(
    analysis: AnalysisResult,
    synthetic_data: dict[str, pd.DataFrame],
    fake: Faker | None = None,
) -> dict[str, pd.DataFrame]:
    for table_name, table_meta in analysis.tables.items():
        if table_name not in synthetic_data:
            continue
        df = synthetic_data[table_name]
        for col_name, col_meta in table_meta.columns.items():
            if col_meta.role != ColumnRole.TEXT:
                continue
            generator = _faker_for_column(col_name, col_meta.faker_override, fake=fake)
            df[col_name] = pd.Series(
                [generator() for _ in range(len(df))],
                index=df.index,
            )
        synthetic_data[table_name] = df
    return synthetic_data


def postprocess_dimension_columns(
    analysis: AnalysisResult,
    synthetic_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Remap dimension column values to anonymized labels, preserving cardinality."""
    for table_name, table_meta in analysis.tables.items():
        if table_name not in synthetic_data:
            continue
        df = synthetic_data[table_name]
        for col_name, col_meta in table_meta.columns.items():
            if col_meta.role != ColumnRole.DIMENSION:
                continue
            if col_name not in df.columns:
                continue
            unique_vals = df[col_name].dropna().unique()
            label_map = {
                val: f"{col_name}_{i + 1}" for i, val in enumerate(unique_vals)
            }
            df[col_name] = df[col_name].map(label_map)
        synthetic_data[table_name] = df
    return synthetic_data


# Synthesis method choices
SYNTHESIS_METHODS = ["auto", "hma", "per_table"]


def synthesize(
    analysis: AnalysisResult,
    pandas_dfs: dict[str, pd.DataFrame],
    scale: float = 1.0,
    seed: int | None = None,
    locale: str | None = None,
    method: str = "auto",
    anonymize_dimensions: bool = False,
    progress_callback: Callable[[float, str], None] | None = None,
    warning_callback: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    def _progress(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    def _warn(msg: str):
        logger.warning(msg)
        if warning_callback:
            warning_callback(msg)

    # Create a local Faker instance for thread safety — each synthesis call
    # gets its own instance so concurrent Streamlit sessions don't interfere.
    if locale and locale != "en_US":
        local_fake = Faker(locale)
        logger.info("Faker locale set to %s", locale)
    else:
        local_fake = Faker()

    # Set seeds for reproducibility (local to this instance)
    if seed is not None:
        local_fake.seed_instance(seed)
        Faker.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    _progress(0.05, "Identifying text columns...")
    text_columns_by_table = _get_text_columns_by_table(analysis)

    _progress(0.08, "Preparing data...")
    clean_data = _prepare_clean_data(pandas_dfs, text_columns_by_table)

    _progress(0.09, "Aligning FK column types...")
    clean_data = _align_fk_types(clean_data, analysis, _warn)

    _progress(0.10, "Fixing date constraint violations...")
    clean_data = _fix_date_constraint_violations(clean_data, analysis, _warn)

    _progress(0.12, "Building SDV metadata...")
    metadata = build_sdv_metadata(analysis, clean_data, _warn)

    _progress(0.15, "Building constraints...")
    all_constraints, constraints_by_table = build_constraints(analysis)

    _progress(0.20, "Validating metadata...")
    try:
        metadata.validate()
    except Exception as e:
        _warn(f"Metadata validation warning: {e}")

    has_relationships = len(analysis.relationships) > 0
    n_tables = len(analysis.tables)
    synthesis_start = time.time()

    logger.info(
        "Starting synthesis: %d table(s), %d relationship(s), scale=%.2f, seed=%s, method=%s",
        n_tables,
        len(analysis.relationships),
        scale,
        seed,
        method,
    )

    # Memory estimation warning + hard limit
    max_output_rows = int(os.environ.get("SANITIZER_MAX_OUTPUT_ROWS", "5000000"))
    estimated_rows = sum(
        max(1, int(len(pandas_dfs.get(t, pd.DataFrame())) * scale))
        for t in analysis.tables
    )
    if estimated_rows > max_output_rows:
        raise ValueError(
            f"Estimated output ({estimated_rows:,} rows) exceeds safety limit ({max_output_rows:,}). "
            f"Reduce scale factor or increase SANITIZER_MAX_OUTPUT_ROWS."
        )
    if estimated_rows > 1_000_000:
        _warn(
            f"Large synthesis: ~{estimated_rows:,} total rows estimated. This may use significant memory."
        )

    use_hma = False
    if method == "hma":
        use_hma = True
    elif method == "per_table":
        use_hma = False
    else:  # auto
        use_hma = has_relationships and n_tables <= 10

    if use_hma:
        _progress(0.25, "Fitting multi-table synthesizer (HMA)...")
        try:
            synthetic_data = _synthesize_multi_table(
                metadata,
                clean_data,
                all_constraints,
                scale,
                _progress,
                _warn,
            )
        except Exception as e:
            logger.exception(
                "HMA synthesis failed, falling back to per-table synthesis"
            )
            _warn(f"HMA synthesis failed ({e}), falling back to per-table synthesis")
            _progress(0.30, "Falling back to per-table synthesis...")
            synthetic_data = _synthesize_single_tables(
                metadata,
                clean_data,
                constraints_by_table,
                scale,
                _progress,
                _warn,
            )
    else:
        _progress(0.25, "Fitting per-table synthesizers...")
        synthetic_data = _synthesize_single_tables(
            metadata,
            clean_data,
            constraints_by_table,
            scale,
            _progress,
            _warn,
        )

    _progress(0.88, "Stitching foreign key references...")
    synthetic_data = _stitch_foreign_keys(analysis, synthetic_data, pandas_dfs)

    _progress(0.92, "Post-processing text columns with Faker...")
    synthetic_data = postprocess_text_columns(analysis, synthetic_data, fake=local_fake)

    if anonymize_dimensions:
        _progress(0.94, "Anonymizing dimension columns...")
        synthetic_data = postprocess_dimension_columns(analysis, synthetic_data)

    # Warn about small tables where synthetic data may closely resemble originals
    for tname, tmeta in analysis.tables.items():
        if tmeta.row_count <= 3 and tname in synthetic_data:
            _warn(
                f"Table '{tname}' has only {tmeta.row_count} row(s) — "
                f"synthetic output may closely resemble original data."
            )

    elapsed = time.time() - synthesis_start
    total_rows = sum(len(df) for df in synthetic_data.values())
    logger.info(
        "Synthesis complete in %.1fs: %d table(s), %d total synthetic rows",
        elapsed,
        len(synthetic_data),
        total_rows,
    )
    _progress(1.0, "Done!")
    return synthetic_data


def _stitch_foreign_keys(
    analysis: AnalysisResult,
    synthetic_data: dict[str, pd.DataFrame],
    original_data: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Replace FK columns in child tables with valid parent PK values,
    preserving the original cardinality distribution and null ratio."""
    if original_data is None:
        original_data = {}

    for rel in analysis.relationships:
        parent_table = rel.parent_table
        child_table = rel.child_table
        parent_col = rel.parent_column
        child_col = rel.child_column

        if parent_table not in synthetic_data or child_table not in synthetic_data:
            logger.debug(
                "Skipping FK stitch %s.%s -> %s.%s: table not in synthetic data",
                child_table,
                child_col,
                parent_table,
                parent_col,
            )
            continue
        if child_col not in synthetic_data[child_table].columns:
            continue
        if parent_col not in synthetic_data[parent_table].columns:
            continue

        parent_ids = synthetic_data[parent_table][parent_col].dropna().values
        if len(parent_ids) == 0:
            logger.warning(
                "FK stitch %s.%s -> %s.%s: parent has no non-null PKs, skipping",
                child_table,
                child_col,
                parent_table,
                parent_col,
            )
            continue

        n_child = len(synthetic_data[child_table])

        # Preserve cardinality distribution from original data instead of uniform sampling
        distribution = None
        if (
            child_table in original_data
            and child_col in original_data[child_table].columns
        ):
            orig_fk = original_data[child_table][child_col].dropna()
            if len(orig_fk) > 0:
                freq = orig_fk.value_counts(normalize=True)
                # Map original FK frequencies to synthetic parent IDs
                unique_parents = pd.Series(parent_ids).unique()
                n_orig_parents = len(freq)
                n_synth_parents = len(unique_parents)
                if n_orig_parents > 0 and n_synth_parents > 0:
                    # Distribute the original frequency pattern across synthetic parent IDs.
                    # If parent counts differ, redistribute proportionally.
                    sorted_freq = freq.sort_values(ascending=False).values
                    if n_synth_parents <= n_orig_parents:
                        # Fewer or equal synthetic parents: aggregate top frequencies
                        weights = np.zeros(n_synth_parents)
                        for i, f in enumerate(sorted_freq):
                            weights[i % n_synth_parents] += f
                    else:
                        # More synthetic parents: spread frequencies across them
                        weights = np.zeros(n_synth_parents)
                        for i in range(n_synth_parents):
                            weights[i] = sorted_freq[i % n_orig_parents]
                    weights = weights / weights.sum()  # normalize
                    distribution = weights
                    parent_ids = unique_parents

        if distribution is not None:
            new_values = np.random.choice(
                parent_ids, size=n_child, replace=True, p=distribution
            )
        else:
            new_values = np.random.choice(parent_ids, size=n_child, replace=True)

        # Preserve null ratio from original data for nullable FKs
        null_ratio = 0.0
        if (
            child_table in original_data
            and child_col in original_data[child_table].columns
        ):
            orig_col = original_data[child_table][child_col]
            if len(orig_col) > 0:
                null_ratio = orig_col.isna().sum() / len(orig_col)

        child_index = synthetic_data[child_table].index
        if null_ratio > 0:
            null_mask = np.random.random(size=n_child) < null_ratio
            synthetic_data[child_table][child_col] = pd.Series(
                new_values, index=child_index
            ).where(~null_mask, other=np.nan)
        else:
            synthetic_data[child_table][child_col] = pd.Series(
                new_values,
                index=child_index,
            )

        logger.info(
            "FK stitch %s.%s -> %s.%s: %d child rows, %d unique parents, null_ratio=%.2f, distribution=%s",
            child_table,
            child_col,
            parent_table,
            parent_col,
            n_child,
            len(np.unique(parent_ids)),
            null_ratio,
            "preserved" if distribution is not None else "uniform",
        )

    return synthetic_data


def _synthesize_multi_table(
    metadata: Metadata,
    clean_data: dict[str, pd.DataFrame],
    constraints: list,
    scale: float,
    _progress: Callable[[float, str], None],
    _warn: Callable[[str], None],
) -> dict[str, pd.DataFrame]:
    synth = HMASynthesizer(metadata=metadata, verbose=True)
    if constraints:
        try:
            synth.add_constraints(constraints)
        except Exception as e:
            _warn(f"Failed to add constraints: {e}")
    _progress(0.35, "Fitting HMA synthesizer (this may take a while)...")
    synth.fit(clean_data)
    _progress(0.75, "Sampling synthetic data...")
    return synth.sample(scale=scale)


def _synthesize_single_tables(
    metadata: Metadata,
    clean_data: dict[str, pd.DataFrame],
    constraints_by_table: dict[str, list],
    scale: float,
    _progress: Callable[[float, str], None],
    _warn: Callable[[str], None],
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
        table_constraints = constraints_by_table.get(table_name, [])
        if table_constraints:
            try:
                synth.add_constraints(table_constraints)
            except Exception as e:
                _warn(f"Failed to add constraints for {table_name}: {e}")

        synth.fit(df)
        synthetic_data[table_name] = synth.sample(num_rows=n_rows)

    return synthetic_data
