# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sanitizer is a Streamlit app that generates fully synthetic versions of Excel data while preserving relational metadata and data patterns. It uses SDV (Synthetic Data Vault) for synthesis and Faker for text columns.

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run the app
streamlit run app.py

# Run with a pre-filled folder path
streamlit run app.py test_data
```

There are no tests or linting configured.

## Architecture

The app follows a four-phase pipeline: **Load → Analyze → Synthesize → Write**.

### Core Modules (`sanitizer/`)

- **models.py** — Dataclasses defining the pipeline's data structures: `ColumnRole` enum, `ColumnMeta`, `TableMeta`, `AnalysisResult`, `Relationship`, `DateConstraintPair`, `DimensionGroup`
- **loader.py** — Discovers .xlsx/.xls files in a folder, reads sheets with fastexcel/polars, converts to pandas (required by SDV's datetime64[ns] expectation)
- **analyzer.py** — Auto-detects primary keys (95% uniqueness threshold), foreign keys (80% overlap threshold), dimensions (adaptive cardinality thresholds), date ordering constraints, and text columns
- **synthesizer.py** — Builds SDV metadata and constraints, chooses HMA (multi-table, ≤10 tables with relationships) or GaussianCopulaSynthesizer (per-table fallback), then post-processes text columns with Faker
- **writer.py** — Exports synthetic data as individual .xlsx files, a single multi-sheet Excel buffer, or a .zip archive
- **ui.py** — Streamlit UI managing four phases (LOAD, REVIEW, SYNTHESIZE, DONE) via `st.session_state`

### Entry Point

`app.py` — Sets Streamlit page config and delegates to `ui.py` phase functions. Session state drives the phase machine.

### Key Design Decisions

- Polars for fast loading, pandas for SDV compatibility — both are maintained in parallel
- Multi-table synthesis falls back to per-table if HMA fails
- Faker column mapping infers column purpose from name patterns (e.g., "email" → `fake.email()`)
- Analyzer thresholds are constants at module top (`PK_UNIQUENESS_THRESHOLD`, `FK_OVERLAP_THRESHOLD`, etc.)

## Dependencies

Python 3.14. Key libraries: streamlit, polars, fastexcel, pandas, sdv, faker, openpyxl. Build system: hatchling.
