from __future__ import annotations

import re
import subprocess
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
    SensitivityLevel,
)
from sanitizer.synthesizer import FAKER_LOCALES, FAKER_TYPE_OPTIONS, SYNTHESIS_METHODS

SDTYPE_OPTIONS = ["id", "numerical", "datetime", "categorical", "boolean", "unknown"]
ROLE_OPTIONS = [r.value for r in ColumnRole]

# Phase display names for the step indicator
_PHASES = {
    "load": ("1", "Load Data"),
    "review": ("2", "Review Metadata"),
    "synthesize": ("3", "Synthesize"),
    "done": ("4", "Results"),
}

# Tooltip help strings
_HELP = {
    "role": {
        "primary_key": "Unique identifier for each row in this table",
        "foreign_key": "Reference to a primary key in another table",
        "dimension": "Low-cardinality categorical column (e.g., region, status)",
        "measure": "Numerical value (e.g., amount, quantity, price)",
        "text": "High-cardinality text replaced with Faker-generated values",
        "date": "Date or datetime column",
        "other": "Uncategorized column",
    },
    "sdtype": {
        "id": "Unique identifier (primary/foreign key)",
        "numerical": "Continuous or discrete number",
        "datetime": "Date or timestamp",
        "categorical": "Finite set of categories",
        "boolean": "True/False value",
        "unknown": "Unclassified — SDV will attempt to infer",
    },
    "scale": "Multiplier for row count. 1.0 = same size as source data. 2.0 = double.",
    "seed": "Set a non-zero value for reproducible output. Same seed + same config = identical data.",
    "faker": "Auto detects the text generator from column name (e.g., 'email' -> email provider). Override if incorrect.",
    "fk_target": "Select the parent_table.column this foreign key references.",
}

# Directory prefixes that should never be used as input/output paths.
# Checked case-insensitively with forward-slash normalisation.
_BLOCKED_PREFIXES = [
    "/",
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/var",
    "/tmp",  # nosec B108
    "/proc",
    "/sys",
    "/dev",
    "C:/Windows",
    "C:/Program Files",
    "C:/Program Files (x86)",
]

# Mermaid identifiers must be alphanumeric/underscores
_MERMAID_SAFE_RE = re.compile(r"[^a-zA-Z0-9_]")


def render_phase_indicator():
    """Show a step indicator at the top of the page."""
    current = st.session_state.get("phase", "load")
    phase_list = list(_PHASES.items())
    cols = st.columns(len(phase_list))
    for i, (phase_key, (num, label)) in enumerate(phase_list):
        with cols[i]:
            if phase_key == current:
                st.markdown(f"**Step {num}: {label}**")
            else:
                st.caption(f"Step {num}: {label}")


def _validate_folder_path(path_str: str) -> Path | None:
    """Validate a user-provided folder path. Returns resolved Path or None with st.error."""
    if not path_str or not path_str.strip():
        return None
    try:
        p = Path(path_str).resolve()
    except (OSError, ValueError):
        st.error(f"Invalid path: {path_str}")
        return None

    # Reject symbolic links to prevent symlink-based traversal
    if p.is_symlink():
        st.error("Symbolic links are not allowed as input paths.")
        return None

    # Block system directories by prefix (not exact match)
    p_normalised = str(p).replace("\\", "/").rstrip("/")

    # Block filesystem / drive roots (e.g. "/", "C:", "D:")
    import re as _re

    if p_normalised == "/" or _re.fullmatch(r"[A-Za-z]:", p_normalised):
        st.error("Cannot use a filesystem root as a data directory.")
        return None

    for prefix in _BLOCKED_PREFIXES:
        pfx = prefix.rstrip("/")
        if pfx == "/":
            continue  # already handled above
        elif p_normalised.upper() == pfx.upper() or p_normalised.upper().startswith(
            pfx.upper() + "/"
        ):
            st.error(f"Cannot use system directory: {p}")
            return None

    return p


def get_folder_from_args() -> str:
    """Extract folder path from CLI args (after '--' in streamlit run)."""
    args = sys.argv[1:]
    try:
        sep_index = args.index("--")
        user_args = args[sep_index + 1 :]
    except ValueError:
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


def _open_folder_dialog() -> str:
    """Open a native OS folder picker dialog via tkinter subprocess."""
    code = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root = tk.Tk()\n"
        "root.withdraw()\n"
        "root.wm_attributes('-topmost', 1)\n"
        "folder = filedialog.askdirectory(title='Select folder containing Excel files')\n"
        "root.destroy()\n"
        "print(folder)\n"
    )
    try:
        result = subprocess.run(  # nosec B603
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def render_folder_selector() -> str | None:
    default = get_folder_from_args()

    # If a browse result is pending, apply it before the widget renders
    if "browse_result" in st.session_state and st.session_state.browse_result:
        st.session_state.folder_path = st.session_state.browse_result
        del st.session_state.browse_result

    if "folder_path" not in st.session_state:
        st.session_state.folder_path = default

    col1, col2 = st.columns([5, 1])
    with col1:
        folder = st.text_input(
            "Folder path containing Excel files",
            key="folder_path",
        )
    with col2:
        st.markdown("<div style='margin-top:1.65em'></div>", unsafe_allow_html=True)
        if st.button("Browse"):
            selected = _open_folder_dialog()
            if selected:
                st.session_state.browse_result = selected
                st.rerun()

    return folder if folder and folder.strip() else None


def render_analysis_thresholds():
    """Render threshold sliders for analysis auto-detection."""
    with st.expander("Advanced Settings", expanded=False):
        st.caption("These thresholds control auto-detection during analysis.")
        st.session_state.pk_threshold = st.slider(
            "PK Uniqueness Threshold",
            min_value=0.80,
            max_value=1.00,
            value=st.session_state.get("pk_threshold", 0.95),
            step=0.01,
            format="%.2f",
            help="Minimum uniqueness ratio to consider a column as a primary key candidate.",
        )
        st.session_state.fk_threshold = st.slider(
            "FK Overlap Threshold",
            min_value=0.50,
            max_value=1.00,
            value=st.session_state.get("fk_threshold", 0.80),
            step=0.01,
            format="%.2f",
            help="Minimum value overlap ratio to detect a foreign key relationship.",
        )
        st.session_state.dim_threshold = st.slider(
            "Dimension Cardinality Threshold",
            min_value=0.01,
            max_value=0.20,
            value=st.session_state.get("dim_threshold", 0.05),
            step=0.01,
            format="%.2f",
            help="Maximum cardinality ratio (unique/total) to classify a column as a dimension.",
        )


def render_config_upload():
    """Render a JSON config upload widget. Returns (analysis, settings) or (None, None)."""
    from sanitizer.config import analysis_from_json

    uploaded = st.file_uploader(
        "Load saved config (optional)",
        type=["json"],
        help="Upload a previously saved .json config to skip analysis and reuse metadata.",
    )
    if uploaded is not None:
        try:
            json_str = uploaded.read().decode("utf-8")
            analysis, settings = analysis_from_json(json_str)
            return analysis, settings
        except Exception as e:
            st.error(f"Failed to parse config: {e}")
    return None, None


def render_config_download(analysis: AnalysisResult, key_suffix: str = ""):
    """Render a JSON config download button."""
    from sanitizer.config import analysis_to_json

    settings = {
        "scale": st.session_state.get("scale", 1.0),
        "seed": st.session_state.get("seed", 0),
        "faker_locale": st.session_state.get("faker_locale", "en_US"),
        "synthesis_method": st.session_state.get("synthesis_method", "auto"),
        "pk_threshold": st.session_state.get("pk_threshold", 0.95),
        "fk_threshold": st.session_state.get("fk_threshold", 0.80),
        "dim_threshold": st.session_state.get("dim_threshold", 0.05),
    }
    json_str = analysis_to_json(analysis, settings)
    st.download_button(
        "Save Config (.json)",
        data=json_str,
        file_name="sanitizer_config.json",
        mime="application/json",
        key=f"dl_config{key_suffix}",
    )
    st.caption(
        "Reload this config later to skip analysis and reuse your metadata settings."
    )


# ── Table overview ───────────────────────────────────────────────────────────


def render_table_overview(analysis: AnalysisResult):
    st.subheader("Tables Overview")
    if not analysis.tables:
        st.info("No tables loaded.")
        return
    cols = st.columns(min(len(analysis.tables), 4))
    for i, (table_name, table_meta) in enumerate(analysis.tables.items()):
        col = cols[i % len(cols)]
        fk_count = sum(
            1 for c in table_meta.columns.values() if c.role == ColumnRole.FOREIGN_KEY
        )
        dim_count = sum(
            1 for c in table_meta.columns.values() if c.role == ColumnRole.DIMENSION
        )
        with col:
            st.metric(table_name, f"{table_meta.row_count} rows")
            st.caption(
                f"PK: {table_meta.primary_key or 'None'} | "
                f"FKs: {fk_count} | Dims: {dim_count} | "
                f"Cols: {len(table_meta.columns)}"
            )


# ── Table detail / column editor ─────────────────────────────────────────────


def _build_fk_target_options(analysis: AnalysisResult, exclude_table: str) -> list[str]:
    """Build list of valid FK targets like 'table.pk_column' from other tables."""
    options = [""]  # empty = no FK
    for name, meta in analysis.tables.items():
        if name == exclude_table:
            continue
        if meta.primary_key:
            options.append(f"{name}.{meta.primary_key}")
    return options


def render_table_detail(table_name: str, analysis: AnalysisResult, df: pd.DataFrame):
    table_meta = analysis.tables[table_name]

    # Data profiling summary
    with st.expander("Data Profile", expanded=False):
        prof_cols = st.columns(4)
        prof_cols[0].metric("Rows", f"{len(df):,}")
        prof_cols[1].metric("Columns", len(df.columns))
        null_pct = df.isna().mean().mean()
        prof_cols[2].metric("Avg Null %", f"{null_pct:.1%}")
        pk = table_meta.primary_key or "-"
        prof_cols[3].metric("Primary Key", pk)

    # PII sensitivity warnings
    pii_cols = [
        (cn, cm)
        for cn, cm in table_meta.columns.items()
        if cm.sensitivity != SensitivityLevel.NONE
    ]
    if pii_cols:
        high_pii = [
            cn for cn, cm in pii_cols if cm.sensitivity == SensitivityLevel.HIGH
        ]
        low_pii = [cn for cn, cm in pii_cols if cm.sensitivity == SensitivityLevel.LOW]
        if high_pii:
            st.warning(
                f"High-sensitivity PII detected: **{', '.join(high_pii)}** — "
                f"ensure Faker override is set for these columns."
            )
        if low_pii:
            st.info(
                f"Possible PII columns: **{', '.join(low_pii)}** — verify Faker override is appropriate."
            )

    fk_options = _build_fk_target_options(analysis, table_name)

    rows = []
    for col_name, col_meta in table_meta.columns.items():
        fk_val = col_meta.foreign_key_target or ""
        # Ensure current value is in options (may be a custom target)
        if fk_val and fk_val not in fk_options:
            fk_options.append(fk_val)
        rows.append(
            {
                "Column": col_name,
                "SDType": col_meta.sdtype,
                "Role": col_meta.role.value,
                "Primary Key": col_meta.is_primary_key,
                "FK Target": fk_val,
                "Faker": col_meta.faker_override or "auto",
                "Uniqueness": f"{col_meta.uniqueness_ratio:.1%}",
                "Samples": str(col_meta.sample_values[:3]),
            }
        )

    edit_df = pd.DataFrame(rows)
    edited = st.data_editor(
        edit_df,
        key=f"columns_{table_name}",
        column_config={
            "Column": st.column_config.TextColumn(disabled=True),
            "SDType": st.column_config.SelectboxColumn(options=SDTYPE_OPTIONS),
            "Role": st.column_config.SelectboxColumn(options=ROLE_OPTIONS),
            "Primary Key": st.column_config.CheckboxColumn(),
            "FK Target": st.column_config.SelectboxColumn(options=fk_options),
            "Faker": st.column_config.SelectboxColumn(
                options=FAKER_TYPE_OPTIONS,
                help="Faker provider for text columns (only used when Role=text)",
            ),
            "Uniqueness": st.column_config.TextColumn(disabled=True),
            "Samples": st.column_config.TextColumn(disabled=True),
        },
        hide_index=True,
        width="stretch",
    )

    _sync_column_edits(table_name, analysis, edited)


def _sync_column_edits(table_name: str, analysis: AnalysisResult, edited: pd.DataFrame):
    table_meta = analysis.tables[table_name]
    new_pk = None

    for _, row in edited.iterrows():
        col_name = row["Column"]
        if col_name not in table_meta.columns:
            continue
        col_meta = table_meta.columns[col_name]

        sdtype = row["SDType"]
        if sdtype not in SDTYPE_OPTIONS:
            st.warning(
                f"Invalid SDType '{sdtype}' for column '{col_name}', keeping '{col_meta.sdtype}'"
            )
            continue

        try:
            role = ColumnRole(row["Role"])
        except ValueError:
            st.warning(
                f"Invalid Role '{row['Role']}' for column '{col_name}', keeping '{col_meta.role.value}'"
            )
            continue

        col_meta.sdtype = sdtype
        col_meta.role = role
        col_meta.is_primary_key = bool(row["Primary Key"])
        col_meta.foreign_key_target = row["FK Target"] if row["FK Target"] else None
        faker_val = row.get("Faker", "auto")
        col_meta.faker_override = (
            faker_val if faker_val and faker_val != "auto" else None
        )

        if col_meta.is_primary_key:
            new_pk = col_name

    if new_pk:
        for col_meta in table_meta.columns.values():
            if col_meta.name != new_pk:
                col_meta.is_primary_key = False
        table_meta.primary_key = new_pk
    else:
        # Check if multiple PKs were selected
        pk_cols = [c.name for c in table_meta.columns.values() if c.is_primary_key]
        if len(pk_cols) > 1:
            st.warning(
                f"Multiple primary keys selected in {table_name}: {', '.join(pk_cols)}. Only one PK per table is supported."
            )


def render_data_preview(table_name: str, df: pd.DataFrame):
    st.caption("Data preview (first 10 rows)")
    st.dataframe(df.head(10), width="stretch", hide_index=True)


def render_synthetic_preview(
    table_name: str,
    analysis: AnalysisResult,
    pandas_dfs: dict[str, pd.DataFrame],
):
    """Show a fast heuristic preview of what synthetic data would look like."""
    from sanitizer.synthesizer import preview_sample

    with st.expander("Synthetic Preview", expanded=False):
        redact = st.checkbox(
            "Redact real values in preview",
            value=st.session_state.get("redact_preview", False),
            key=f"redact_preview_{table_name}",
            help="Replace dimension/FK values with synthetic placeholders (useful for screen-sharing)",
        )
        st.caption(
            "Heuristic preview — structurally plausible, not statistically accurate"
        )
        try:
            preview_df = preview_sample(
                table_name,
                analysis,
                pandas_dfs,
                n_rows=10,
                redact_preview=redact,
            )
            st.dataframe(preview_df, width="stretch", hide_index=True)
        except Exception as e:
            st.warning(f"Preview unavailable: {e}")


# ── Relationship graph + editor ──────────────────────────────────────────────


def _mermaid_id(name: str) -> str:
    """Make a name safe for use as a mermaid identifier."""
    return _MERMAID_SAFE_RE.sub("_", name) or "table"


def _render_relationship_graph_mermaid(analysis: AnalysisResult):
    """Render a mermaid ER diagram of table relationships (fallback)."""
    table_names = set(analysis.tables.keys())
    if not table_names:
        return

    lines = ["erDiagram"]
    linked_tables: set[str] = set()

    for rel in analysis.relationships:
        pid = _mermaid_id(rel.parent_table)
        cid = _mermaid_id(rel.child_table)
        label = f"{rel.parent_column} to {rel.child_column}"
        lines.append(f'    {pid} ||--o{{ {cid} : "{label}"')
        linked_tables.add(rel.parent_table)
        linked_tables.add(rel.child_table)

    for t in table_names - linked_tables:
        tid = _mermaid_id(t)
        lines.append(f"    {tid}")

    diagram = "\n".join(lines)
    st.markdown(f"```mermaid\n{diagram}\n```")


# ── Column-level interactive graph ───────────────────────────────────────────

_TABLE_GAP_X = 300
_HEADER_HEIGHT = 40
_COL_NODE_HEIGHT = 36
_COL_NODE_GAP = 4
_COL_NODE_WIDTH = 180

_HEADER_STYLE = {
    "background": "#4A90D9",
    "color": "white",
    "fontWeight": "bold",
    "borderRadius": "6px 6px 0 0",
    "border": "2px solid #2C5F8A",
    "width": f"{_COL_NODE_WIDTH}px",
    "padding": "6px 12px",
    "fontSize": "13px",
    "textAlign": "center",
}


def _parse_column_node_id(node_id: str) -> tuple[str, str] | None:
    """Parse 'col::table::column' → (table, column), or None."""
    parts = node_id.split("::", 2)
    if len(parts) == 3 and parts[0] == "col":
        return parts[1], parts[2]
    return None


def _get_visible_columns(table_meta, show_all: bool) -> list[str]:
    if show_all:
        return list(table_meta.columns.keys())
    return [
        name
        for name, col in table_meta.columns.items()
        if col.is_primary_key
        or col.role
        in (ColumnRole.PRIMARY_KEY, ColumnRole.FOREIGN_KEY, ColumnRole.DIMENSION)
        or col.sdtype == "id"
    ]


def _column_bg_color(col_meta: ColumnMeta) -> str:
    if col_meta.is_primary_key:
        return "#FFF3CD"
    if col_meta.role == ColumnRole.FOREIGN_KEY:
        return "#D1ECF1"
    if col_meta.role == ColumnRole.DIMENSION:
        return "#D4EDDA"
    return "#F8F9FA"


def _render_relationship_graph_interactive(analysis: AnalysisResult):
    """Render an interactive column-level graph for drag-and-drop relationship building."""
    try:
        from streamlit_flow import streamlit_flow
        from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
        from streamlit_flow.state import StreamlitFlowState
    except ImportError:
        st.caption(
            "Install `streamlit-flow-component` for interactive graph. Falling back to diagram."
        )
        _render_relationship_graph_mermaid(analysis)
        return

    if not analysis.tables:
        return

    # Clean up old pending state from previous implementation
    if "_pending_rel" in st.session_state:
        del st.session_state["_pending_rel"]

    show_all = st.checkbox(
        "Show all columns",
        value=False,
        key="rel_graph_show_all",
        help="By default only key and dimension columns are shown",
    )

    # Table positions stored in session state so dragging headers persists
    if "_table_positions" not in st.session_state:
        st.session_state._table_positions = {}
    table_positions: dict[str, tuple[float, float]] = st.session_state._table_positions

    # Build nodes — header + columns per table, arranged horizontally
    nodes: list[StreamlitFlowNode] = []
    max_col_count = 0

    for i, (table_name, table_meta) in enumerate(analysis.tables.items()):
        visible_cols = _get_visible_columns(table_meta, show_all)
        max_col_count = max(max_col_count, len(visible_cols))

        # Use stored position or default grid layout
        default_x = i * _TABLE_GAP_X
        base_x, base_y = table_positions.get(table_name, (default_x, 0))

        # Header node — draggable handle for the whole table group
        nodes.append(
            StreamlitFlowNode(
                id=f"hdr::{table_name}",
                pos=(base_x, base_y),
                data={"content": f"**{table_name}** ({table_meta.row_count} rows)"},
                node_type="default",
                source_position="right",
                target_position="left",
                connectable=False,
                draggable=True,
                style=_HEADER_STYLE,
            )
        )

        # Column nodes stacked below header, pinned relative to it
        y = base_y + _HEADER_HEIGHT + _COL_NODE_GAP
        for col_name in visible_cols:
            col_meta = table_meta.columns[col_name]
            prefix = (
                "PK "
                if col_meta.is_primary_key
                else ("FK " if col_meta.role == ColumnRole.FOREIGN_KEY else "")
            )
            bg = _column_bg_color(col_meta)

            nodes.append(
                StreamlitFlowNode(
                    id=f"col::{table_name}::{col_name}",
                    pos=(base_x, y),
                    data={"content": f"{prefix}{col_name}"},
                    node_type="default",
                    source_position="right",
                    target_position="left",
                    connectable=True,
                    draggable=False,
                    style={
                        "background": bg,
                        "border": "1px solid #ccc",
                        "borderRadius": "0px",
                        "width": f"{_COL_NODE_WIDTH}px",
                        "padding": "4px 10px",
                        "fontSize": "12px",
                        "cursor": "crosshair",
                    },
                )
            )
            y += _COL_NODE_HEIGHT + _COL_NODE_GAP

    # Build edges from existing relationships (column-level)
    edges: list[StreamlitFlowEdge] = []
    for i, rel in enumerate(analysis.relationships):
        edges.append(
            StreamlitFlowEdge(
                id=f"rel_{i}",
                source=f"col::{rel.parent_table}::{rel.parent_column}",
                target=f"col::{rel.child_table}::{rel.child_column}",
                animated=True,
                label=f"{rel.parent_column} → {rel.child_column}",
                deletable=True,
            )
        )

    # Dynamic height based on tallest column stack
    content_h = (
        _HEADER_HEIGHT
        + _COL_NODE_GAP
        + max_col_count * (_COL_NODE_HEIGHT + _COL_NODE_GAP)
    )
    height = max(300, content_h + 80)

    curr_state = StreamlitFlowState(nodes=nodes, edges=edges)
    new_state = streamlit_flow(
        "rel_flow",
        curr_state,
        fit_view=True,
        height=height,
        allow_new_edges=True,
        animate_new_edges=True,
        enable_node_menu=False,
        enable_edge_menu=True,
        enable_pane_menu=False,
        show_minimap=False,
        pan_on_drag=True,
        allow_zoom=True,
        min_zoom=0.3,
        hide_watermark=True,
    )

    # Sync header positions — when a header is dragged, columns follow on next render
    header_moved = False
    for node in new_state.nodes:
        if node.id.startswith("hdr::"):
            tname = node.id.split("::", 1)[1]
            new_pos = (node.position["x"], node.position["y"])
            old_pos = table_positions.get(tname)
            if old_pos != new_pos:
                table_positions[tname] = new_pos
                header_moved = True
    if header_moved:
        st.session_state._table_positions = table_positions
        st.rerun()

    # Detect new edges — create relationships directly from column node IDs
    existing_edge_ids = {e.id for e in edges}
    changed = False
    for edge in new_state.edges:
        if edge.id in existing_edge_ids:
            continue
        source = _parse_column_node_id(edge.source)
        target = _parse_column_node_id(edge.target)
        if source is None or target is None:
            continue  # connected to header — ignore
        parent_table, parent_column = source
        child_table, child_column = target
        if parent_table == child_table:
            continue  # skip self-table edges
        already = any(
            r.parent_table == parent_table
            and r.parent_column == parent_column
            and r.child_table == child_table
            and r.child_column == child_column
            for r in analysis.relationships
        )
        if not already:
            analysis.relationships.append(
                Relationship(
                    parent_table=parent_table,
                    parent_column=parent_column,
                    child_table=child_table,
                    child_column=child_column,
                    overlap_ratio=0.0,
                )
            )
            changed = True

    # Detect removed edges
    new_edge_pairs = set()
    for edge in new_state.edges:
        src = _parse_column_node_id(edge.source)
        tgt = _parse_column_node_id(edge.target)
        if src and tgt:
            new_edge_pairs.add((*src, *tgt))
    before = len(analysis.relationships)
    analysis.relationships = [
        r
        for r in analysis.relationships
        if (r.parent_table, r.parent_column, r.child_table, r.child_column)
        in new_edge_pairs
    ]
    if len(analysis.relationships) != before:
        changed = True

    if changed:
        st.rerun()


def render_relationships_editor(analysis: AnalysisResult):
    st.subheader("Relationships")
    table_names = list(analysis.tables.keys())

    # Interactive column-level graph (with Mermaid fallback)
    _render_relationship_graph_interactive(analysis)

    # Current relationships list
    if analysis.relationships:
        for _i, rel in enumerate(analysis.relationships):
            c = st.columns([3, 1, 3, 1])
            c[0].markdown(f"**{rel.parent_table}** `.{rel.parent_column}`")
            c[1].markdown("&rarr;")
            c[2].markdown(f"**{rel.child_table}** `.{rel.child_column}`")
            rel_key = f"del_rel_{rel.parent_table}_{rel.parent_column}_{rel.child_table}_{rel.child_column}"
            if c[3].button("Remove", key=rel_key):
                analysis.relationships = [
                    r
                    for r in analysis.relationships
                    if not (
                        r.parent_table == rel.parent_table
                        and r.parent_column == rel.parent_column
                        and r.child_table == rel.child_table
                        and r.child_column == rel.child_column
                    )
                ]
                st.rerun()
    else:
        st.caption("No relationships detected.")

    # Manual linker (fallback / alternative)
    with st.expander("Link tables manually"):
        left, mid, right = st.columns([5, 1, 5])

        with left:
            st.caption("Parent (one side)")
            parent_table = st.selectbox(
                "Parent table",
                table_names,
                key="add_rel_pt",
                label_visibility="collapsed",
            )
            parent_cols = (
                list(analysis.tables[parent_table].columns.keys())
                if parent_table
                else []
            )
            parent_col = st.selectbox("Parent column", parent_cols, key="add_rel_pc")
            if parent_table:
                pk = analysis.tables[parent_table].primary_key
                st.caption(
                    f"PK: {pk or 'None'}  |  {analysis.tables[parent_table].row_count} rows"
                )

        with mid:
            st.markdown("")
            st.markdown("")
            st.markdown(
                "<h2 style='text-align:center'>&rarr;</h2>", unsafe_allow_html=True
            )

        with right:
            st.caption("Child (many side)")
            child_table = st.selectbox(
                "Child table",
                table_names,
                key="add_rel_ct",
                label_visibility="collapsed",
            )
            child_cols = (
                list(analysis.tables[child_table].columns.keys()) if child_table else []
            )
            child_col = st.selectbox("Child column", child_cols, key="add_rel_cc")
            if child_table:
                pk = analysis.tables[child_table].primary_key
                st.caption(
                    f"PK: {pk or 'None'}  |  {analysis.tables[child_table].row_count} rows"
                )

        if (
            st.button("Link", type="primary", key="add_rel_btn")
            and parent_table
            and child_table
            and parent_col
            and child_col
        ):
            if parent_table == child_table:
                st.warning("Cannot create a relationship within the same table.")
            elif any(
                r.parent_table == parent_table
                and r.parent_column == parent_col
                and r.child_table == child_table
                and r.child_column == child_col
                for r in analysis.relationships
            ):
                st.warning(
                    f"Relationship {parent_table}.{parent_col} → {child_table}.{child_col} already exists."
                )
            else:
                analysis.relationships.append(
                    Relationship(
                        parent_table=parent_table,
                        parent_column=parent_col,
                        child_table=child_table,
                        child_column=child_col,
                        overlap_ratio=0.0,
                    )
                )
                st.rerun()


# ── Date constraints editor ──────────────────────────────────────────────────


def render_date_constraints_editor(analysis: AnalysisResult):
    st.subheader("Date Constraints")

    if analysis.date_constraints:
        for _i, pair in enumerate(analysis.date_constraints):
            cols = st.columns([3, 3, 3, 2, 1])
            cols[0].text(pair.table_name)
            cols[1].text(f"{pair.low_column} <=")
            cols[2].text(pair.high_column)
            cols[3].text(f"{pair.violation_count} violations")
            dc_key = f"del_dc_{pair.table_name}_{pair.low_column}_{pair.high_column}"
            if cols[4].button("X", key=dc_key):
                analysis.date_constraints = [
                    p
                    for p in analysis.date_constraints
                    if not (
                        p.table_name == pair.table_name
                        and p.low_column == pair.low_column
                        and p.high_column == pair.high_column
                    )
                ]
                st.rerun()
    else:
        st.caption("No date constraints detected.")

    with st.expander("Add date constraint"):
        table_names = list(analysis.tables.keys())
        table = st.selectbox("Table", table_names, key="add_dc_t")
        date_cols = (
            [
                c
                for c, m in analysis.tables[table].columns.items()
                if m.sdtype == "datetime" or m.role == ColumnRole.DATE
            ]
            if table
            else []
        )
        c1, c2 = st.columns(2)
        low = c1.selectbox("Low column (earlier)", date_cols, key="add_dc_low")
        high = c2.selectbox("High column (later)", date_cols, key="add_dc_high")
        if st.button("Add", key="add_dc_btn") and low and high and low != high:
            analysis.date_constraints.append(
                DateConstraintPair(
                    table_name=table,
                    low_column=low,
                    high_column=high,
                )
            )
            st.rerun()


# ── Dimension groups editor ──────────────────────────────────────────────────


def render_dimension_groups_editor(analysis: AnalysisResult):
    st.subheader("Dimension Groups (Fixed Combinations)")

    if analysis.dimension_groups:
        for _i, group in enumerate(analysis.dimension_groups):
            cols = st.columns([3, 5, 2, 1])
            cols[0].text(group.table_name)
            cols[1].text(", ".join(group.column_names))
            cols[2].text(f"{group.combination_count} combos")
            dg_key = f"del_dg_{group.table_name}_{'_'.join(group.column_names)}"
            if cols[3].button("X", key=dg_key):
                analysis.dimension_groups = [
                    g
                    for g in analysis.dimension_groups
                    if not (
                        g.table_name == group.table_name
                        and g.column_names == group.column_names
                    )
                ]
                st.rerun()
    else:
        st.caption("No dimension groups detected.")

    with st.expander("Add dimension group"):
        table_names = list(analysis.tables.keys())
        table = st.selectbox("Table", table_names, key="add_dg_t")
        dim_cols = (
            [
                c
                for c, m in analysis.tables[table].columns.items()
                if m.role == ColumnRole.DIMENSION
            ]
            if table
            else []
        )
        selected = st.multiselect("Columns", dim_cols, key="add_dg_cols")
        if st.button("Add", key="add_dg_btn") and len(selected) >= 2:
            analysis.dimension_groups.append(
                DimensionGroup(
                    table_name=table,
                    column_names=selected,
                    combination_count=0,
                )
            )
            st.rerun()


# ── Synthesis controls ───────────────────────────────────────────────────────


def render_synthesis_controls():
    st.subheader("Generate Synthetic Data")
    st.session_state.scale = st.slider(
        "Scale factor",
        min_value=0.1,
        max_value=10.0,
        value=st.session_state.get("scale", 1.0),
        step=0.1,
        help=_HELP["scale"],
    )
    st.session_state.seed = st.number_input(
        "Random seed (0 = non-deterministic)",
        min_value=0,
        max_value=2**31 - 1,
        value=st.session_state.get("seed", 0),
        step=1,
        help=_HELP["seed"],
    )

    current_locale = st.session_state.get("faker_locale", "en_US")
    locale_idx = (
        FAKER_LOCALES.index(current_locale) if current_locale in FAKER_LOCALES else 0
    )
    st.session_state.faker_locale = st.selectbox(
        "Faker locale",
        options=FAKER_LOCALES,
        index=locale_idx,
        help="Locale for Faker-generated text columns (names, addresses, etc.)",
    )

    with st.expander("Advanced", expanded=False):
        method_labels = {
            "auto": "Auto (HMA if relationships exist, else per-table)",
            "hma": "Force HMA (multi-table, preserves cross-table relationships)",
            "per_table": "Force Per-Table (faster, independent synthesis per table)",
        }
        current_method = st.session_state.get("synthesis_method", "auto")
        method_idx = (
            SYNTHESIS_METHODS.index(current_method)
            if current_method in SYNTHESIS_METHODS
            else 0
        )
        st.session_state.synthesis_method = st.selectbox(
            "Synthesis method",
            options=SYNTHESIS_METHODS,
            index=method_idx,
            format_func=lambda x: method_labels.get(x, x),
            help="HMA preserves cross-table relationships but is slower. Per-table is faster but synthesizes each table independently.",
        )
        st.session_state.anonymize_dimensions = st.checkbox(
            "Anonymize dimension values",
            value=st.session_state.get("anonymize_dimensions", False),
            help="Remap categorical/dimension column values to anonymized labels (e.g., Region_1, Status_2). "
            "Preserves cardinality and distribution but removes original category names from the output.",
        )

    if st.button("Generate Synthetic Data", type="primary"):
        st.session_state.trigger_synthesis = True


# ── Validation report ────────────────────────────────────────────────────────


def render_validation_report():
    """Show validation results in the DONE phase."""
    from sanitizer.validator import ValidationResult

    validation: ValidationResult | None = st.session_state.get("validation_result")
    if validation is None:
        return

    st.subheader("Validation Report")

    if validation.all_passed:
        st.success("All validation checks passed")
    else:
        st.warning("Some validation checks failed — review details below")

    # PK uniqueness
    if validation.pk_checks:
        with st.expander("Primary Key Uniqueness", expanded=not validation.all_passed):
            for table, check in validation.pk_checks.items():
                if check.passed:
                    st.markdown(
                        f"  {table}.{check.column} — {check.total_rows} rows, all unique"
                    )
                else:
                    st.markdown(
                        f"  {table}.{check.column} — **{check.duplicate_count} duplicates** "
                        f"out of {check.total_rows} rows"
                    )

    # FK integrity
    if validation.fk_checks:
        with st.expander("Foreign Key Integrity", expanded=not validation.all_passed):
            for _key, check in validation.fk_checks.items():
                if check.passed:
                    st.markdown(
                        f"  {check.child_table}.{check.child_column} -> "
                        f"{check.parent_table}.{check.parent_column} — "
                        f"{check.total_fk_values} values, all valid"
                    )
                else:
                    st.markdown(
                        f"  {check.child_table}.{check.child_column} -> "
                        f"{check.parent_table}.{check.parent_column} — "
                        f"**{check.orphan_count} orphaned** out of {check.total_fk_values}"
                    )

    # Date constraints
    if validation.date_checks:
        with st.expander("Date Constraints", expanded=not validation.all_passed):
            for _key, check in validation.date_checks.items():
                if check.passed:
                    st.markdown(
                        f"  {check.table_name}: {check.low_column} <= {check.high_column} — "
                        f"{check.total_rows} rows checked, all valid"
                    )
                else:
                    st.markdown(
                        f"  {check.table_name}: {check.low_column} <= {check.high_column} — "
                        f"**{check.violation_count} violations** out of {check.total_rows} rows"
                    )

    # Row counts
    if validation.row_counts:
        with st.expander("Row Counts", expanded=False):
            rows_data = []
            for table, rc in validation.row_counts.items():
                rows_data.append(
                    {
                        "Table": table,
                        "Original": rc.original_rows,
                        "Synthetic": rc.synthetic_rows,
                        "Expected": rc.expected_rows,
                        "Scale": f"{rc.scale:.2f}x",
                    }
                )
            st.dataframe(pd.DataFrame(rows_data), hide_index=True)

    # Column stats comparison
    if validation.column_stats:
        with st.expander("Distribution Comparison (numerical columns)", expanded=False):
            for table_name, cols in validation.column_stats.items():
                stats_rows = []
                for col_name, s in cols.items():
                    if s.orig_mean is not None:
                        stats_rows.append(
                            {
                                "Column": col_name,
                                "Orig Mean": f"{s.orig_mean:.2f}"
                                if s.orig_mean is not None
                                else "-",
                                "Synth Mean": f"{s.synth_mean:.2f}"
                                if s.synth_mean is not None
                                else "-",
                                "Orig Std": f"{s.orig_std:.2f}"
                                if s.orig_std is not None
                                else "-",
                                "Synth Std": f"{s.synth_std:.2f}"
                                if s.synth_std is not None
                                else "-",
                                "Orig Null%": f"{s.orig_null_ratio:.1%}"
                                if s.orig_null_ratio is not None
                                else "-",
                                "Synth Null%": f"{s.synth_null_ratio:.1%}"
                                if s.synth_null_ratio is not None
                                else "-",
                            }
                        )
                if stats_rows:
                    st.caption(f"**{table_name}**")
                    st.dataframe(pd.DataFrame(stats_rows), hide_index=True)

    # Download validation as JSON
    import json

    report_data = {
        "all_passed": validation.all_passed,
        "pk_checks": {
            t: {"column": c.column, "duplicates": c.duplicate_count, "passed": c.passed}
            for t, c in validation.pk_checks.items()
        },
        "fk_checks": {
            k: {
                "orphans": c.orphan_count,
                "total": c.total_fk_values,
                "passed": c.passed,
            }
            for k, c in validation.fk_checks.items()
        },
        "date_checks": {
            k: {
                "violations": c.violation_count,
                "total": c.total_rows,
                "passed": c.passed,
            }
            for k, c in validation.date_checks.items()
        },
    }
    st.download_button(
        "Download Validation Report (JSON)",
        data=json.dumps(report_data, indent=2),
        file_name="validation_report.json",
        mime="application/json",
        key="dl_validation",
    )


# ── Output section ───────────────────────────────────────────────────────────


def render_output_section(
    synthetic_data: dict[str, pd.DataFrame], analysis: AnalysisResult
):
    from sanitizer.writer import (
        build_manifest,
        create_zip_buffer,
        dataframe_to_excel_buffer,
    )

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
    manifest = build_manifest(
        synthetic_data,
        analysis,
        scale=st.session_state.get("scale"),
        seed=st.session_state.get("seed"),
    )
    zip_buf = create_zip_buffer(synthetic_data, table_relative_dirs, manifest=manifest)
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
                st.caption(
                    "Subdirectory structure from the original folder has been preserved."
                )
