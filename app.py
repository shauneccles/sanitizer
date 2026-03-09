from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from sanitizer.loader import load_all
from sanitizer.analyzer import analyze
from sanitizer.synthesizer import synthesize
from sanitizer.ui import (
    _validate_folder_path,
    get_folder_from_args,
    init_session_state,
    render_folder_selector,
    render_table_overview,
    render_table_detail,
    render_data_preview,
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

            st.success(f"Loaded {len(polars_dfs)} table(s)")

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

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            render_relationships_editor(analysis)
        with col2:
            render_date_constraints_editor(analysis)

        render_dimension_groups_editor(analysis)

        st.divider()
        render_synthesis_controls()

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
        if st.button("Back to Review"):
            st.session_state.phase = "review"
            st.session_state.synthetic_data = None
            st.rerun()


if __name__ == "__main__":
    main()
