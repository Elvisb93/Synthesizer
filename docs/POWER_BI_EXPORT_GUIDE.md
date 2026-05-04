# Power BI Export Guide

Synthesizer can export generated or enriched tabular data as versioned Power BI runs.

The export is intentionally file-based for the first integration stage. This works with local folders, OneDrive synced folders, and SharePoint synced document libraries without requiring Microsoft authentication inside Synthesizer.

## Output Layout

Each export creates a new run folder and appends a row to `index.csv`.

```text
.web_ui_exports/
  power_bi/
    index.csv
    runs/
      2026-05-01_161530_customer_contacts/
        data.csv
        schema.json
        metadata.json
```

Run folders are never overwritten. If two exports happen in the same second for the same dataset, Synthesizer adds a numeric suffix to the later run folder.

## Files

### data.csv

The generated or enriched rows in the current schema order. Values are flat tabular CSV values for simple Power BI import.

### schema.json

The schema file records:

- column name
- Synthesizer column type
- Power BI type hint
- prompt instruction
- duplicate policy
- constraints
- schema hash

### metadata.json

The metadata file records:

- run ID
- dataset name and slug
- creation timestamp
- row and column counts
- source mode: fresh generation or imported enrichment
- privacy export mode
- provider and model
- schema hash
- whether the schema changed from the previous run for the same dataset

### index.csv

The index file is an append-only run registry. It is the best starting point for Power BI audit/history reports because it lists every successful export and relative paths to each run artifact.

## Recommended SharePoint Workflow

1. Sync the target SharePoint document library with OneDrive.
2. In Synthesizer, open `Generate Sample Data`.
3. Generate or enrich the tabular rows.
4. Open `Power BI Export`.
5. Set `Destination Folder` to the synced SharePoint/OneDrive folder.
6. Click `Export Power BI Run`.
7. In Power BI, connect to the synced folder or to the SharePoint folder and use `index.csv` as the run registry.

This avoids direct SharePoint authentication in Synthesizer while still making the files available to team Power BI reports.

## Privacy Export Mode

`Restored imported values` is the default. It exports the current generated/enriched rows, including restored imported source columns.

`Masked imported values where available` replaces imported source columns with the masked import values when the run came from imported data. Generated free-text fields are exported as currently stored in the session.

## Schema Change Warning

Synthesizer computes a schema hash from ordered column names and types. If a new run for the same dataset has a different schema from the previous run, the UI warns that existing Power BI reports may need query or model updates.

## Current Limits

- CSV is easy to import but does not preserve rich typing by itself.
- Direct SharePoint web upload is not included yet.
- Power BI custom connector support is a future option once the file contract is proven.
- Parquet would be a useful later addition for larger datasets and stronger typing.
