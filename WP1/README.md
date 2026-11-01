# WP1: EFFIS Fire Data Cleaning & Standardization (`fires_master`)

Cleans and standardizes the raw EFFIS tabular wildfire data for Spain and Portugal, producing the master dataset (`fires_master.parquet`) used by downstream modeling work packages (WP2–WP5).

## Setup
    pip install pandas numpy pyarrow

Effis data table file goes into 'data/raw' (not committed; add 'data/' to gitignore or specify .csv and .parquet files in gitignore):
- `effis_data_table_29thSept2026.csv` *(Note: update `input_file` in `process_effis_data.py` if your downloaded file has a different name)*.

## Run, in order
    python process_effis_data.py
    python check_master_table.py
    python generate_stats.py  # prints summary metrics to review

## Choices made (each is easy to change)
- **Geography Filter:** Restricted to Spain (`ES`) and Portugal (`PT`).
- **Date Parsing:** Standardized using mixed format parsing (`pd.to_datetime(..., format='mixed')`) to handle inconsistent string formatting in raw export dates. Year and month are extracted from `FIREDATE` (start date).
- **Product Era Mapping (`product_era`):** Categorizes fire records into eras based on satellite/product version changes:
  - Era `0`: $\le 2018$
  - Era `1`: $2019$
  - Era `2`: $\ge 2020$
- **Size Classification (`size_class`):** Grouped into standard analytical tiers for modeling:
  - `< 30 ha` (Small fires)
  - `30 to 382 ha` (Medium fires)
  - `383 ha or more` (Large / extreme fires)
- **Duration:** Calculated as `duration_days = FINALDATE - FIREDATE`.
- **Feature Cleaning:** Removed redundant land cover percentage columns (`BROADLEA`, `CONIFER`, etc.) and administrative metadata (`LASTUPDATE`, `COMMUNE`, etc.) to keep the master table focused.
- **Column Renaming:** Standardized primary keys (`id` $\rightarrow$ `fire_id`, `COUNTRY` $\rightarrow$ `country`, `PROVINCE` $\rightarrow$ `nuts2`, `AREA_HA` $\rightarrow$ `area_ha`).

## Outputs (`data/processed/`)
| file | one row per | key columns |
|---|---|---|
| `fires_master.parquet` | individual wildfire event | `fire_id`, `country`, `nuts2`, `area_ha`, `firedate`, `year`, `month`, `product_era`, `size_class`, `duration_days` |
| `fire_stats.md` | Manually curated summary report (built from `generate_stats.py` output) | Overview of fire counts, size distributions, and temporal breakdowns across Spain and Portugal |