from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from sanitizer.models import (
    AnalysisResult,
    ColumnMeta,
    ColumnRole,
    DateConstraintPair,
    DimensionGroup,
    Relationship,
)

SDTYPE_OPTIONS = ["id", "numerical", "datetime", "categorical", "boolean", "unknown"]
ROLE_OPTIONS = [r.value for r in ColumnRole]

# Directories that should never be used as input/output paths
_BLOCKED_PATHS = {
    "/", "C:\\", "C:\\Windows", "C:\\Windows\\System32",
    "/etc", "/usr", "/bin", "/sbin", "/var", "/tmp",
}


def _validate_folder_path(path_str: str) -> Path | None:
    """Validate a user-provided folder path. Returns resolved Path or None with st.error."""
    if not path_str or not path_str.strip():
        return None
    try:
        p = Path(path_str).resolve()
    except (OSError, ValueError):
        st.error(f"Invalid path: {path_str}")
        return None

    # Block path traversal and system directories
    p_str = str(p).replace("\\", "/")
    if any(p_str.upper() == blocked.upper().replace("\\", "/") for blocked in _BLOCKED_PATHS):
        st.error(f"Cannot use system directory: {p}")
        return None

    return p


def get_folder_from_args() -> str:
    """Extract folder path from CLI args (after '--' in streamlit run)."""
    # Only consider args after '--' separator, which streamlit uses
    args = sys.argv[1:]
    try:
        sep_index = args.index("--")
        user_args = args[sep_index + 1:]
    except ValueError:
        # No '--' separator; take last arg if it doesn't look like a flag
        user_args = args

    if user_args:
        candidate = user_args[-1]
        if not candidate.startswith("-"):
            return candidate
    return ""


def init_session_state(
    analysis: AnalysisResult,
    pandas_dfs: dict[str, pd.DataFrame],
):
    st.session_state.analysis = analysis
    st.session_state.pandas_dfs = pandas_dfs
    st.session_state.phase = "review"
    st.session_state.synthetic_data = None
    st.session_state.scale = 1.0


def render_folder_selector() -> str | None:
    default = get_folder_from_args()
    folder = st.text_input("Folder path containing Excel files", value=default)
    return folder if folder.strip() else None


def render_table_overview(analysis: AnalysisResult):
    st.subheader("Tables Overview")
    if not analysis.tables:
        st.info("No tables loaded.")
        return
    cols = st.columns(min(len(analysis.tables), 4))
    for i, (table_name, table_meta) in enumerate(analysis.tables.items()):
        col = cols[i % len(cols)]
        fk_count = sum(
            1 for c in table_meta.columns.values()
            if c.role == ColumnRole.FOREIGN_KEY
        )
        dim_count = sum(
            1 for c in table_meta.columns.values()
            if c.role == ColumnRole.DIMENSION
        )
        with col:
            st.metric(table_name, f"{table_meta.row_count} rows")
            st.caption(
                f"PK: {table_meta.primary_key or 'None'} | "
                f"FKs: {fk_count} | Dims: {dim_count} | "
                f"Cols: {len(table_meta.columns)}"
            )


def render_table_detail(table_name: str, analysis: AnalysisResult, df: pd.DataFrame):
    table_meta = analysis.tables[table_name]

    # Build editable dataframe from column metadata
    rows = []
    for col_name, col_meta in table_meta.columns.items():
        rows.append({
            "Column": col_name,
            "SDType": col_meta.sdtype,
            "Role": col_meta.role.value,
            "Primary Key": col_meta.is_primary_key,
            "FK Target": col_meta.foreign_key_target or "",
            "Uniqueness": f"{col_meta.uniqueness_ratio:.1%}",
            "Samples": str(col_meta.sample_values[:3]),
        })

    edit_df = pd.DataFrame(rows)
    edited = st.data_editor(
        edit_df,
        key=f"columns_{table_name}",
        column_config={
            "Column": st.column_config.TextColumn(disabled=True),
            "SDType": st.column_config.SelectboxColumn(options=SDTYPE_OPTIONS),
            "Role": st.column_config.SelectboxColumn(options=ROLE_OPTIONS),
            "Primary Key": st.column_config.CheckboxColumn(),
            "FK Target": st.column_config.TextColumn(),
            "Uniqueness": st.column_config.TextColumn(disabled=True),
            "Samples": st.column_config.TextColumn(disabled=True),
        },
        hide_index=True,
        width="stretch",
    )

    # Sync edits back to analysis
    _sync_column_edits(table_name, analysis, edited)


def _sync_column_edits(table_name: str, analysis: AnalysisResult, edited: pd.DataFrame):
    table_meta = analysis.tables[table_name]
    new_pk = None

    for _, row in edited.iterrows():
        col_name = row["Column"]
        if col_name not in table_meta.columns:
            continue
        col_meta = table_meta.columns[col_name]

        # Validate sdtype
        sdtype = row["SDType"]
        if sdtype not in SDTYPE_OPTIONS:
            st.warning(f"Invalid SDType '{sdtype}' for column '{col_name}', keeping '{col_meta.sdtype}'")
            continue

        # Validate role
        try:
            role = ColumnRole(row["Role"])
        except ValueError:
            st.warning(f"Invalid Role '{row['Role']}' for column '{col_name}', keeping '{col_meta.role.value}'")
            continue

        col_meta.sdtype = sdtype
        col_meta.role = role
        col_meta.is_primary_key = bool(row["Primary Key"])
        col_meta.foreign_key_target = row["FK Target"] if row["FK Target"] else None

        if col_meta.is_primary_key:
            new_pk = col_name

    if new_pk:
        # Ensure only one PK
        for col_meta in table_meta.columns.values():
            if col_meta.name != new_pk:
                col_meta.is_primary_key = False
        table_meta.primary_key = new_pk


def render_data_preview(table_name: str, df: pd.DataFrame):
    st.caption("Data preview (first 10 rows)")
    st.dataframe(df.head(10), width="stretch", hide_index=True)


def render_relationships_editor(analysis: AnalysisResult):
    st.subheader("Relationships")
    table_names = list(analysis.tables.keys())

    if analysis.relationships:
        for i, rel in enumerate(analysis.relationships):
            cols = st.columns([3, 2, 3, 2, 1, 1])
            cols[0].text(f"{rel.parent_table}")
            cols[1].text(f".{rel.parent_column}")
            cols[2].text(f"{rel.child_table}")
            cols[3].text(f".{rel.child_column}")
            cols[4].text(f"{rel.overlap_ratio:.0%}")
            if cols[5].button("X", key=f"del_rel_{i}"):
                analysis.relationships.pop(i)
                st.rerun()
    else:
        st.caption("No relationships detected.")

    with st.expander("Add relationship"):
        c1, c2, c3, c4 = st.columns(4)
        parent_table = c1.selectbox("Parent table", table_names, key="add_rel_pt")
        parent_cols = list(analysis.tables[parent_table].columns.keys()) if parent_table else []
        parent_col = c2.selectbox("Parent column", parent_cols, key="add_rel_pc")
        child_table = c3.selectbox("Child table", table_names, key="add_rel_ct")
        child_cols = list(analysis.tables[child_table].columns.keys()) if child_table else []
        child_col = c4.selectbox("Child column", child_cols, key="add_rel_cc")

        if st.button("Add", key="add_rel_btn"):
            analysis.relationships.append(Relationship(
                parent_table=parent_table,
                parent_column=parent_col,
                child_table=child_table,
                child_column=child_col,
                overlap_ratio=0.0,
            ))
            st.rerun()


def render_date_constraints_editor(analysis: AnalysisResult):
    st.subheader("Date Constraints")

    if analysis.date_constraints:
        for i, pair in enumerate(analysis.date_constraints):
            cols = st.columns([3, 3, 3, 2, 1])
            cols[0].text(pair.table_name)
            cols[1].text(f"{pair.low_column} <=")
            cols[2].text(pair.high_column)
            cols[3].text(f"{pair.violation_count} violations")
            if cols[4].button("X", key=f"del_dc_{i}"):
                analysis.date_constraints.pop(i)
                st.rerun()
    else:
        st.caption("No date constraints detected.")

    with st.expander("Add date constraint"):
        table_names = list(analysis.tables.keys())
        table = st.selectbox("Table", table_names, key="add_dc_t")
        date_cols = [
            c for c, m in analysis.tables[table].columns.items()
            if m.sdtype == "datetime" or m.role == ColumnRole.DATE
        ] if table else []
        c1, c2 = st.columns(2)
        low = c1.selectbox("Low column (earlier)", date_cols, key="add_dc_low")
        high = c2.selectbox("High column (later)", date_cols, key="add_dc_high")
        if st.button("Add", key="add_dc_btn") and low and high and low != high:
            analysis.date_constraints.append(DateConstraintPair(
                table_name=table,
                low_column=low,
                high_column=high,
            ))
            st.rerun()


def render_dimension_groups_editor(analysis: AnalysisResult):
    st.subheader("Dimension Groups (Fixed Combinations)")

    if analysis.dimension_groups:
        for i, group in enumerate(analysis.dimension_groups):
            cols = st.columns([3, 5, 2, 1])
            cols[0].text(group.table_name)
            cols[1].text(", ".join(group.column_names))
            cols[2].text(f"{group.combination_count} combos")
            if cols[3].button("X", key=f"del_dg_{i}"):
                analysis.dimension_groups.pop(i)
                st.rerun()
    else:
        st.caption("No dimension groups detected.")

    with st.expander("Add dimension group"):
        table_names = list(analysis.tables.keys())
        table = st.selectbox("Table", table_names, key="add_dg_t")
        dim_cols = [
            c for c, m in analysis.tables[table].columns.items()
            if m.role == ColumnRole.DIMENSION
        ] if table else []
        selected = st.multiselect("Columns", dim_cols, key="add_dg_cols")
        if st.button("Add", key="add_dg_btn") and len(selected) >= 2:
            analysis.dimension_groups.append(DimensionGroup(
                table_name=table,
                column_names=selected,
                combination_count=0,
            ))
            st.rerun()


def render_synthesis_controls():
    st.subheader("Generate Synthetic Data")
    st.session_state.scale = st.slider(
        "Scale factor (multiplier for row counts)",
        min_value=0.1,
        max_value=10.0,
        value=st.session_state.get("scale", 1.0),
        step=0.1,
    )
    st.session_state.seed = st.number_input(
        "Random seed (leave 0 for non-deterministic)",
        min_value=0,
        max_value=2**31 - 1,
        value=st.session_state.get("seed", 0),
        step=1,
    )
    if st.button("Generate Synthetic Data", type="primary"):
        st.session_state.trigger_synthesis = True


def render_output_section(synthetic_data: dict[str, pd.DataFrame], analysis: AnalysisResult):
    from sanitizer.writer import create_zip_buffer, dataframe_to_excel_buffer

    # Extract relative dirs from analysis for preserving folder structure
    table_relative_dirs = {
        name: meta.relative_dir for name, meta in analysis.tables.items()
    }
    has_subdirs = any(d for d in table_relative_dirs.values())

    st.subheader("Synthetic Data Output")

    for table_name, df in synthetic_data.items():
        rel_dir = table_relative_dirs.get(table_name, "")
        label = f"{rel_dir}/{table_name}" if rel_dir else table_name
        with st.expander(f"{label} ({len(df)} rows)", expanded=False):
            st.dataframe(df.head(20), width="stretch", hide_index=True)
            buf = dataframe_to_excel_buffer(df)
            st.download_button(
                f"Download {table_name}.xlsx",
                data=buf,
                file_name=f"{table_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_{table_name}",
            )

    st.divider()
    zip_buf = create_zip_buffer(synthetic_data, table_relative_dirs)
    st.download_button(
        "Download All (ZIP)",
        data=zip_buf,
        file_name="synthetic_data.zip",
        mime="application/zip",
    )
    if has_subdirs:
        st.caption("ZIP preserves the original subdirectory structure.")

    st.divider()
    output_path_str = st.text_input("Save to folder", value="")
    if output_path_str and st.button("Save to folder"):
        from sanitizer.writer import write_excel_files

        output_path = _validate_folder_path(output_path_str)
        if output_path is not None:
            paths = write_excel_files(synthetic_data, output_path, table_relative_dirs)
            st.success(f"Saved {len(paths)} files to {output_path}")
            if has_subdirs:
                st.caption("Subdirectory structure from the original folder has been preserved.")
