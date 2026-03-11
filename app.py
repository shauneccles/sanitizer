from __future__ import annotations

import logging

import streamlit as st

from sanitizer.loader import load_all
from sanitizer.analyzer import analyze
from sanitizer.synthesizer import synthesize
from sanitizer.ui import (
    _validate_folder_path,
    init_session_state,
    render_config_download,
    render_config_upload,
    render_folder_selector,
    render_table_overview,
    render_table_detail,
    render_data_preview,
    render_synthetic_preview,
    render_relationships_editor,
    render_date_constraints_editor,
    render_dimension_groups_editor,
    render_synthesis_controls,
    render_output_section,
)

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")


def main():
    st.set_page_config(
        page_title="Sanitizer",
        layout="wide",
    )

    st.title("Sanitizer")
    st.caption("Generate fully synthetic versions of your Excel data")

    if "phase" not in st.session_state:
        st.session_state.phase = "load"

    # ── LOAD ────────────────────────────────────────────────────────────────
    if st.session_state.phase == "load":
        folder_path = render_folder_selector()

        st.divider()
        saved_analysis, saved_settings = render_config_upload()

        if folder_path and st.button("Load & Analyze", type="primary"):
            folder = _validate_folder_path(folder_path)
            if folder is None:
                return
            if not folder.is_dir():
                st.error(f"Not a directory: {folder}")
                return

            with st.spinner("Loading Excel files (searching subdirectories)..."):
                try:
                    polars_dfs, pandas_dfs, table_sources, table_relative_dirs = load_all(folder)
                except Exception as e:
                    st.error(f"Failed to load: {e}")
                    return

            loaded_tables = set(polars_dfs.keys())
            st.success(f"Loaded {len(loaded_tables)} table(s)")

            if saved_analysis is not None:
                config_tables = set(saved_analysis.tables.keys())
                matched = loaded_tables & config_tables
                missing_from_config = loaded_tables - config_tables

                if matched:
                    st.info(f"Applying saved config for {len(matched)} table(s).")

                if missing_from_config:
                    # Analyze tables that aren't covered by the config
                    st.info(
                        f"Analyzing {len(missing_from_config)} new table(s) "
                        f"not in config: {', '.join(sorted(missing_from_config))}"
                    )
                    fresh = analyze(polars_dfs, table_sources, table_relative_dirs)
                    for t in missing_from_config:
                        saved_analysis.tables[t] = fresh.tables[t]
                    # Keep fresh relationships/constraints for new tables too
                    existing_rels = {
                        (r.parent_table, r.child_table, r.parent_column, r.child_column)
                        for r in saved_analysis.relationships
                    }
                    for r in fresh.relationships:
                        key = (r.parent_table, r.child_table, r.parent_column, r.child_column)
                        if key not in existing_rels:
                            saved_analysis.relationships.append(r)
                    for dc in fresh.date_constraints:
                        if dc.table_name in missing_from_config:
                            saved_analysis.date_constraints.append(dc)
                    for dg in fresh.dimension_groups:
                        if dg.table_name in missing_from_config:
                            saved_analysis.dimension_groups.append(dg)

                # Remove config entries for tables that no longer exist in the data
                extra = config_tables - loaded_tables
                if extra:
                    st.warning(
                        f"Config tables not found in data (skipped): "
                        f"{', '.join(sorted(extra))}"
                    )
                    for t in extra:
                        del saved_analysis.tables[t]
                    saved_analysis.relationships = [
                        r for r in saved_analysis.relationships
                        if r.parent_table in loaded_tables and r.child_table in loaded_tables
                    ]
                    saved_analysis.date_constraints = [
                        dc for dc in saved_analysis.date_constraints
                        if dc.table_name in loaded_tables
                    ]
                    saved_analysis.dimension_groups = [
                        dg for dg in saved_analysis.dimension_groups
                        if dg.table_name in loaded_tables
                    ]

                # Update relative dirs from the actual loaded data
                for t in saved_analysis.tables:
                    saved_analysis.tables[t].relative_dir = table_relative_dirs.get(t, "")

                analysis = saved_analysis

                # Apply saved settings
                if saved_settings:
                    st.session_state.scale = saved_settings.get("scale", 1.0)
                    st.session_state.seed = saved_settings.get("seed", 0)
            else:
                with st.spinner("Analyzing metadata..."):
                    analysis = analyze(polars_dfs, table_sources, table_relative_dirs)

            init_session_state(analysis, pandas_dfs)
            st.session_state.phase = "review"
            st.rerun()
        return

    # ── REVIEW ──────────────────────────────────────────────────────────────
    if st.session_state.phase == "review":
        analysis = st.session_state.analysis
        pandas_dfs = st.session_state.pandas_dfs

        render_table_overview(analysis)

        st.divider()

        for table_name, table_meta in analysis.tables.items():
            rel = table_meta.relative_dir
            label = f"Table: {rel}/{table_name}" if rel else f"Table: {table_name}"
            with st.expander(label, expanded=False):
                render_table_detail(table_name, analysis, pandas_dfs[table_name])
                render_data_preview(table_name, pandas_dfs[table_name])
                render_synthetic_preview(table_name, analysis, pandas_dfs)

        st.divider()

        render_relationships_editor(analysis)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            render_date_constraints_editor(analysis)
        with col2:
            render_dimension_groups_editor(analysis)

        st.divider()

        left, right = st.columns([2, 1])
        with left:
            render_synthesis_controls()
        with right:
            st.subheader("Config")
            render_config_download(analysis)

        if st.session_state.get("trigger_synthesis"):
            st.session_state.trigger_synthesis = False
            st.session_state.phase = "synthesize"
            st.rerun()
        return

    # ── SYNTHESIZE ──────────────────────────────────────────────────────────
    if st.session_state.phase == "synthesize":
        analysis = st.session_state.analysis
        pandas_dfs = st.session_state.pandas_dfs
        scale = st.session_state.get("scale", 1.0)
        seed_val = st.session_state.get("seed", 0)
        seed = seed_val if seed_val != 0 else None

        progress_bar = st.progress(0.0, text="Starting synthesis...")
        warning_container = st.container()

        def progress_callback(pct: float, msg: str):
            progress_bar.progress(min(pct, 1.0), text=msg)

        def warning_callback(msg: str):
            warning_container.warning(msg)

        try:
            synthetic_data = synthesize(
                analysis=analysis,
                pandas_dfs=pandas_dfs,
                scale=scale,
                seed=seed,
                progress_callback=progress_callback,
                warning_callback=warning_callback,
            )
            st.session_state.synthetic_data = synthetic_data
            st.session_state.phase = "done"
            st.rerun()
        except Exception as e:
            st.error(f"Synthesis failed: {e}")
            with st.expander("Show traceback (debug)"):
                import traceback
                st.code(traceback.format_exc())
            if st.button("Back to Review"):
                st.session_state.phase = "review"
                st.rerun()
        return

    # ── DONE ────────────────────────────────────────────────────────────────
    if st.session_state.phase == "done":
        st.success("Synthetic data generated successfully!")
        render_output_section(st.session_state.synthetic_data, st.session_state.analysis)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            render_config_download(st.session_state.analysis, key_suffix="_done")
        with col2:
            if st.button("Back to Review"):
                st.session_state.phase = "review"
                st.session_state.synthetic_data = None
                st.rerun()


if __name__ == "__main__":
    main()
