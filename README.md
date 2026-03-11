# Sanitizer

Generate fully synthetic versions of Excel data while preserving relational metadata, statistical distributions, and structural patterns. Point it at a folder of `.xlsx` files and get back fake data that looks and behaves like the real thing — without any of the real values.

Built with [Streamlit](https://streamlit.io/), [SDV (Synthetic Data Vault)](https://sdv.dev/), [Polars](https://pola.rs/), and [Faker](https://faker.readthedocs.io/).

## Why

You need realistic test data, a demo dataset, or a safe copy of production data for development — but the original contains PII, financial records, or other sensitive information. Sanitizer automates this by:

1. Detecting the structure of your data (primary keys, foreign keys, dimensions, date orderings)
2. Fitting statistical models to learn distributions and correlations
3. Generating new rows that follow the same patterns with entirely synthetic values
4. Replacing text columns (names, emails, addresses) with plausible Faker-generated values

The output contains zero original records. Foreign key references between tables remain valid.

## Quick Start

```bash
# Install dependencies (requires Python 3.11+, uses uv)
uv sync

# Run the app
streamlit run app.py

# Run with a pre-filled folder path
streamlit run app.py test_data
```

Open the URL printed by Streamlit (typically `http://localhost:8501`).

## Usage Walkthrough

Sanitizer operates in four phases, driven by the UI:

### Phase 1 — Load

Enter the path to a folder containing `.xlsx` or `.xls` files and click **Load & Analyze**. Every Excel file in the folder (including subdirectories) is read — multi-sheet workbooks produce one table per non-empty sheet (named `filename_sheetname`). Single-sheet workbooks use the filename as the table name.

You can optionally upload a previously saved config JSON to restore analysis settings from a prior session.

### Phase 2 — Review

After loading, the analyzer has auto-detected metadata for every table. This phase lets you inspect and override everything before synthesis.

**Tables Overview** shows row counts, detected primary keys, foreign key counts, and dimension counts for each table at a glance.

**Per-table detail** (expand any table) shows an editable grid of column metadata:

| Field | Meaning |
|-------|---------|
| SDType | The SDV semantic type: `id`, `numerical`, `datetime`, `categorical`, `boolean`, or `unknown` |
| Role | How the column is treated: `primary_key`, `foreign_key`, `dimension`, `measure`, `text`, `date`, `other` |
| Primary Key | Check to mark as PK (only one per table) |
| FK Target | If a foreign key, the `parent_table.column` it references (dropdown of available PKs) |
| Faker | Override the Faker provider for text columns (e.g., `email`, `name`, `company`) |

You can change any value in the grid. Edits are validated — invalid SDType or Role values are rejected with a warning.

**Synthetic Preview** (inside each table expander) shows a fast heuristic preview of what synthetic data would look like — structurally plausible rows generated in milliseconds without SDV fitting.

**Relationships** shows an interactive column-level graph for drag-and-drop relationship building. Columns from different tables can be connected by dragging between them. Existing relationships appear as animated edges. You can also right-click edges to delete them, or use the manual linker as a fallback.

The graph shows key and dimension columns by default — toggle **Show all columns** to see every column. Nodes are color-coded: gold for primary keys, blue for foreign keys, green for dimensions. Table headers can be dragged to rearrange the layout.

**Date Constraints** shows detected column orderings (e.g., `order_date <= ship_date`). These are enforced during synthesis so the synthetic data maintains temporal logic.

**Dimension Groups** shows sets of low-cardinality columns with functional dependencies (e.g., `region` and `country` always appear in the same combinations). These are preserved as fixed combinations during synthesis.

**Config** — Download or upload the current analysis as a JSON config file to reuse settings across sessions.

**Synthesis Controls** at the bottom of the review page:

- **Scale factor** — Multiplier for row counts. `1.0` produces the same number of rows as the original. `2.0` doubles them.
- **Random seed** — Set a non-zero value for reproducible output. Leave at `0` for non-deterministic generation.

Click **Generate Synthetic Data** to proceed.

### Phase 3 — Synthesize

A progress bar tracks the synthesis pipeline. Any issues (failed constraints, metadata warnings, HMA fallback) are displayed as warnings in real time rather than silently logged.

If multi-table synthesis via HMA fails, the system automatically falls back to per-table Gaussian Copula synthesis.

### Phase 4 — Output

Once synthesis completes, you can:

- **Preview** the first 20 rows of each synthetic table
- **Download individual tables** as `.xlsx` files
- **Download all tables** as a single `.zip` archive
- **Save to a folder** on disk (the path is validated before writing; subdirectory structure from the original folder is preserved)

Click **Back to Review** to adjust settings and re-generate.

## How It Works

### Pipeline Architecture

```
Excel files on disk
        |
   [1. LOAD]        loader.py — Polars reads via fastexcel, converts to pandas for SDV
        |
   [2. ANALYZE]     analyzer.py — Auto-detects PKs, FKs, dimensions, date constraints, text columns
        |
   [3. SYNTHESIZE]  synthesizer.py — Builds SDV metadata, fits models, generates rows, Faker post-processes text
        |
   [4. WRITE]       writer.py — Exports to .xlsx files, in-memory buffers, or .zip
```

### Loading (`sanitizer/loader.py`)

- Discovers all `.xlsx` / `.xls` files in the given folder and subdirectories
- Reads each sheet using `fastexcel` + `polars` for speed
- Converts to pandas DataFrames (required by SDV, which expects `datetime64[ns]`)
- Sanitizes table names to remove filesystem-unsafe characters
- Disambiguates duplicate table names by appending `_2`, `_3`, etc.
- Tracks relative subdirectory paths so output preserves folder structure

### Analysis (`sanitizer/analyzer.py`)

The analyzer runs three phases over the Polars DataFrames:

**Primary Key Detection** — Looks for non-null columns with high uniqueness ratio (>= 95%) whose names end in common PK patterns (`_id`, `_key`, `_code`, etc.). Falls back to any 100%-unique column if no suffix match is found.

**Foreign Key Detection** — For each detected PK, scans all other tables for columns with matching names and >= 80% value overlap with the parent PK values. Column name matching checks:
- Exact name match (`customer_id` = `customer_id`)
- Table-prefixed match (`customers_customer_id`)
- Table-suffixed ID (`customers_id`)

**Per-Table Analysis:**

| Detection | Method |
|-----------|--------|
| Dimensions | String/categorical columns with cardinality ratio below an adaptive threshold (50% for <50 rows, 15% for <200, 5% otherwise) |
| Dimension groups | Pairs of dimensions with functional dependencies (the distinct-pair count equals one column's distinct count) |
| Date constraints | Pairs of date/datetime columns where `col_a <= col_b` holds for >= 98% of rows |
| Text columns | High-cardinality string columns with average length > 3 characters |

**Column Classification** maps each column to an SDV `sdtype` (`id`, `numerical`, `datetime`, `categorical`, `boolean`, `unknown`) and a `ColumnRole` (`primary_key`, `foreign_key`, `dimension`, `measure`, `text`, `date`, `other`).

### Synthesis (`sanitizer/synthesizer.py`)

1. **Text column removal** — Text columns are dropped from the data before fitting (they'd add noise to statistical models). They're restored with Faker values after synthesis.

2. **SDV metadata construction** — Builds an `sdv.metadata.Metadata` object from the analysis results: sets primary keys, column sdtypes, datetime formats, and parent-child relationships.

3. **Constraint construction** — Creates SDV `Inequality` constraints for date orderings and `FixedCombinations` constraints for dimension groups.

4. **Model fitting and sampling:**
   - If tables have relationships and there are <= 10 tables: uses `HMASynthesizer` (Hierarchical Modeling Approach) for multi-table synthesis that respects relational structure
   - Otherwise, or if HMA fails: falls back to per-table `GaussianCopulaSynthesizer`
   - Scale factor controls how many rows to generate relative to the original

5. **FK stitching** — After sampling, foreign key columns in child tables are replaced with values randomly drawn from the parent table's synthetic PK column. This ensures referential integrity even when using per-table synthesis.

6. **Text post-processing** — Faker generates replacement values for text columns. Column names are matched against known patterns to select appropriate generators:

   | Column name pattern | Faker generator |
   |---------------------|-----------------|
   | `name`, `full_name` | `fake.name()` |
   | `first_name`, `fname` | `fake.first_name()` |
   | `email`, `email_address` | `fake.email()` |
   | `phone`, `telephone` | `fake.phone_number()` |
   | `address`, `street` | `fake.address()` |
   | `city` | `fake.city()` |
   | `company`, `organization` | `fake.company()` |
   | `url`, `website` | `fake.url()` |
   | `description`, `comment`, `notes`, etc. | `fake.sentence()` |
   | *(unrecognized)* | `fake.sentence(nb_words=6)` |

   Matching uses word segments (split on `_`), not substring matching — so `username` won't incorrectly match the `name` pattern. Per-column overrides can be set in the UI's Faker dropdown.

7. **Heuristic preview** — `preview_sample()` generates structurally plausible rows in milliseconds without SDV fitting, used for real-time feedback in the column editor. It samples from source data ranges for numerical/datetime columns, uses Faker for text, and draws from existing values for categoricals.

### Config (`sanitizer/config.py`)

Serializes and deserializes `AnalysisResult` objects to/from JSON, enabling save/reload of analysis configurations across sessions. Includes all column metadata, relationships, date constraints, dimension groups, and synthesis settings (scale, seed).

### Writing (`sanitizer/writer.py`)

- `write_excel_files()` — Writes one `.xlsx` per table to a folder on disk, preserving subdirectory structure
- `dataframe_to_excel_buffer()` — Writes a single DataFrame to an in-memory `BytesIO` buffer
- `create_zip_buffer()` — Bundles all tables into a `.zip` archive in memory

All output filenames are sanitized to remove unsafe characters.

### UI (`sanitizer/ui.py`)

The Streamlit UI manages a four-phase state machine via `st.session_state`. Key features:

- Editable column metadata grids with dropdown selectors for SDType, Role, FK Target, and Faker provider
- Interactive column-level relationship graph via `streamlit-flow-component` — drag between columns to create relationships, right-click edges to delete
- Draggable table headers to rearrange the relationship graph layout
- Color-coded nodes: gold (PK), blue (FK), green (dimension)
- Fast heuristic synthetic data preview per table
- Inline date constraint and dimension group editors with add/remove controls
- Real-time progress bar and warning display during synthesis
- Config save/load for reusing analysis settings across sessions
- Download buttons for individual and bulk export
- Path validation for both input and output directories

## Project Structure

```
sanitizer/
    __init__.py         Package definition
    models.py           Dataclasses: ColumnMeta, TableMeta, Relationship, etc.
    loader.py           Excel file discovery and reading
    analyzer.py         Auto-detection of keys, relationships, dimensions
    synthesizer.py      SDV metadata, model fitting, sampling, Faker text, heuristic preview
    config.py           JSON serialization/deserialization for analysis configs
    writer.py           Excel/ZIP export
    ui.py               Streamlit UI components and state management
app.py                  Entry point — page config, phase routing
pyproject.toml          Dependencies and build config
test_data/              Sample Excel files (customers, orders, products)
```

## Configuration

### Analyzer Thresholds

These constants in `sanitizer/analyzer.py` control auto-detection sensitivity:

| Constant | Default | Effect |
|----------|---------|--------|
| `PK_UNIQUENESS_THRESHOLD` | 0.95 | Minimum uniqueness ratio for a column to be a PK candidate |
| `FK_OVERLAP_THRESHOLD` | 0.80 | Minimum value overlap for a column to be detected as a FK |
| `DIMENSION_CARDINALITY_THRESHOLD` | 0.05 | Maximum cardinality ratio for a column to be classified as a dimension (for tables with 200+ rows) |
| `PK_SUFFIX_PATTERNS` | `_id`, `_key`, `_nr`, `_no`, `_pk`, `_code`, `id` | Column name suffixes that indicate PK candidates |

### Synthesis Options

| Option | UI Control | Default |
|--------|-----------|---------|
| Scale factor | Slider (0.1x – 10x) | 1.0 |
| Random seed | Number input | 0 (non-deterministic) |

Setting a non-zero seed makes Faker, NumPy, and SDV produce identical output across runs.

### Faker Overrides

Per-column Faker provider overrides can be set in the column editor dropdown. Available providers:

`address`, `city`, `company`, `country`, `date`, `email`, `first_name`, `job`, `last_name`, `name`, `paragraph`, `phone`, `sentence`, `url`, `uuid`, `zipcode`

Set to `auto` (default) to use name-pattern matching.

## Dependencies

| Library | Purpose |
|---------|---------|
| streamlit | Web UI framework |
| polars | Fast Excel reading and data analysis |
| fastexcel | High-performance Excel sheet discovery |
| pandas | DataFrame format required by SDV |
| sdv | Synthetic data generation (Gaussian Copula, HMA) |
| faker | Realistic fake text values (names, emails, etc.) |
| openpyxl | Excel `.xlsx` writing |
| streamlit-flow-component | Interactive node graph for relationship building |

Requires Python 3.11 or newer. Install with:

```bash
uv sync
```

## Limitations

- **Input format** — Only `.xlsx` and `.xls` files. CSV, Parquet, and database connections are not supported.
- **Memory** — Data is held in both Polars and pandas representations simultaneously. Very large datasets (millions of rows across many tables) may require significant RAM.
- **HMA scaling** — Multi-table synthesis with HMA can be slow for datasets with many tables or complex relationship graphs. The app limits HMA to <= 10 tables and falls back to per-table synthesis otherwise.
- **Text column matching** — Faker provider selection is heuristic. Columns with unusual names may get generic sentence output instead of domain-specific values. Use the Faker dropdown override for these cases.
