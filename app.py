from __future__ import annotations

from pathlib import Path

import streamlit as st

from sanitizer.loader import load_all
from sanitizer.analyzer import analyze
from sanitizer.synthesizer import synthesize
from sanitizer.ui import (
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
            folder = Path(folder_path)
            if not folder.exists():
                st.error(f"Folder not found: {folder_path}")
                return

            with st.spinner("Loading Excel files..."):
                try:
                    polars_dfs, pandas_dfs, table_sources = load_all(folder)
                except Exception as e:
                    st.error(f"Failed to load: {e}")
                    return

            st.success(f"Loaded {len(polars_dfs)} table(s)")

            with st.spinner("Analyzing metadata..."):
                analysis = analyze(polars_dfs, pandas_dfs, table_sources)

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

        for table_name in analysis.tables:
            with st.expander(f"Table: {table_name}", expanded=False):
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

        progress_bar = st.progress(0.0, text="Starting synthesis...")

        def progress_callback(pct: float, msg: str):
            progress_bar.progress(min(pct, 1.0), text=msg)

        try:
            synthetic_data = synthesize(
                analysis=analysis,
                pandas_dfs=pandas_dfs,
                scale=scale,
                progress_callback=progress_callback,
            )
            st.session_state.synthetic_data = synthetic_data
            st.session_state.phase = "done"
            st.rerun()
        except Exception as e:
            st.error(f"Synthesis failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            if st.button("Back to Review"):
                st.session_state.phase = "review"
                st.rerun()
        return

    # ── DONE ────────────────────────────────────────────────────────────────
    if st.session_state.phase == "done":
        st.success("Synthetic data generated successfully!")
        render_output_section(st.session_state.synthetic_data)

        st.divider()
        if st.button("Back to Review"):
            st.session_state.phase = "review"
            st.session_state.synthetic_data = None
            st.rerun()


if __name__ == "__main__":
    main()
