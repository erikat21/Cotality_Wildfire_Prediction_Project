# WP5: Spatio-Temporal Panel Generation & Validation

This work package (`WP5`) constructs the final cell-month panel required for statistical analysis and machine-learning model training in **WP6**. It integrates multi-source environmental, meteorological, topographical, and historical fire datasets across a standardized spatio-temporal grid for Spain and Portugal.

The panel uses the current **6,555 in-mask cells** from WP2, identified consistently by `cell_id`. The **Open Period** covers May 2016–December 2023 (92 months). The optional **Locked-inclusive Period** extends through August 2026 (124 months) and must be reserved for the agreed test workflow.

## Repository Structure

```text
WP5/
├── README.md
├── cell_month_panel.py      # Panel construction, joins, and imputation
├── validate_panel.py        # Independent structural and feature validation
└── outputs/
    ├── cell_month_panel.parquet          # Open Period: May 2016–Dec 2023
    ├── cell_month_panel.parquet.dvc      # DVC metadata, when tracked
    └── cell_month_panel_locked.parquet   # Optional locked-inclusive output
```

> `cell_month_panel_locked.parquet` is produced only when `--locked` is run. It includes **both** open and locked years, not only the held-out years. Its existence, dimensions, and validation have not yet been confirmed in the current review. Do not treat the locked-inclusive output as an ordinary training file.

## Setup

From the repository root, install the required packages if they are not already available:

```powershell
pip install pandas numpy scipy pyarrow
```

**Required processed inputs** (paths relative to repository root):

| Work package | Input | Purpose |
|---|---|---|
| WP2 | `wp2/outputs/cells.parquet` | Spatial IDs, mask, grid row/column and `nuts2` |
| WP2 | `wp2/outputs/labels.parquet` | Official monthly fire labels, including `y_S30_any` and `burn_frac_S30` |
| WP3 | `wp3/outputs/weather_cell_month.parquet` | Monthly `tmax`, `vpdmax`, `tdmean`, `ppt` |
| WP4 | `WP4/outputs/landcover_cell.parquet` | CORINE 2012 and 2018 land-cover shares |
| WP4 | `WP4/outputs/terrain_cell.parquet` | Static elevation, slope, and aspect features |
| WP4 | `WP4/outputs/veg_cell_month.parquet` | Monthly lag-one NDVI |

The earlier WP5 implementation used `wp2/outputs/fire_cell.parquet` and computed `max(burn_frac_cell)` for each cell-month. **The revised panel instead uses the official WP2 `labels.parquet` output**, which computes combined burned fractions after merging overlapping fire perimeters. Generate upstream data or retrieve DVC-tracked outputs before running WP5. The complete raw WP1–WP4 pipelines do not need to be rerun if the required processed files are already available.

## Run, in order

Run these commands from the **repository root**:

```powershell
# Build the open-period panel (May 2016–December 2023)
python -u WP5/cell_month_panel.py

# Validate the open-period panel
python -u WP5/validate_panel.py
```

Optional locked-inclusive build, **only when authorized for evaluation**:

```powershell
python -u WP5/cell_month_panel.py --locked
```

The locked-inclusive build writes `WP5/outputs/cell_month_panel_locked.parquet`, separately from the open-period panel. The current `validate_panel.py` verifies the **open** output by default; successful open-period validation does **not** validate the locked-inclusive file.

## Choices made

- **Spatio-Temporal Skeleton:** Creates one row for every combination of the 6,555 current WP2 in-mask cells and each selected month. The cell-month key is (`cell_id`, `year`, `month`).
- **Spatial Imputation (cKDTree; FIX P0-1):** For each month and each selected numeric predictor **independently**, fills missing values using the nearest cell with a valid observation in grid (`row`, `col`) space. The old function reused the missingness mask from the first incomplete feature across all features; the new function preserves every original nonmissing value.
- **Temporal Fallback (FIX P0-2):** Sorts observations chronologically by cell and applies **forward fill only** to selected weather and NDVI predictors if spatial filling leaves gaps. Backward filling (`bfill`) and the whole-panel median fallback were removed to avoid using later months or locked years to fill earlier observations. A missing predictor is not silently replaced with an arbitrary value if no eligible donor exists.
- **Missingness Indicators (FIX P0-3):** Preserves `ndvi_missing_original` and `aspect_missing_original` as binary columns before predictor imputation. This distinguishes original observations from estimated values; undefined terrain aspect is especially relevant to the four near-water coastal cells identified by WP4.
- **Land Cover Epoch Mapping:** Uses CORINE 2012 for 2016–2017 and CORINE 2018 for 2018 onward. This is the team's current retrospective epoch convention and must be reviewed against the intended forecasting date and publication availability.
- **Fire Aggregation (FIX P1):** Uses `y_S30_any` as `fire` and `burn_frac_S30` as `burn_frac` from WP2 `labels.parquet`. WP2 merges overlapping burned-area polygons within a cell-month before measuring the combined fraction. Cell-months absent from the sparse label table are assigned zero under the project's recorded-fire assumption. The S30 definition refers to fire events meeting the 30-hectare minimum-size criterion; other thresholds are available in WP2 but are not selected here.
- **Output Isolation (FIX P2):** Open and locked-inclusive modes write different filenames so the optional test-period run cannot overwrite the validated open-period panel.
- **Join Integrity:** Requires the expected upstream key columns and rejects duplicate input keys before joining. This protects against silent duplication of panel rows.

### Important modeling-time caveat

Removing `bfill` and whole-panel median fitting addresses **two specific leakage pathways**, but is **not a complete proof of leakage-free forecasting**. In particular, the current weather variables are **same-month aggregates**. They would not be available at the start of that month for a true month-ahead forecast. WP6 must define the prediction issue date and shift/lag any features that would not yet be observable. Spatial imputation using the same-month values also assumes those donor values are available at prediction time. Keep locked years out of training, hyperparameter selection, and preprocessing-statistic fitting.

## Outputs (`WP5/outputs/`)

| File | Period | Expected rows | Primary key |
|---|---|---:|---|
| `cell_month_panel.parquet` | Open: 2016-05–2023-12 | **603,060** | `cell_id`, `year`, `month` |
| `cell_month_panel_locked.parquet` | Open + Locked: 2016-05–2026-08 | **812,820** (expected, not yet tested here) | `cell_id`, `year`, `month` |

The validated **open-period** panel contains **30 columns**:

- **Keys/metadata:** `cell_id`, `year`, `month`, `nuts2`
- **Targets:** `fire`, `burn_frac`
- **Weather:** `tmax`, `vpdmax`, `tdmean`, `ppt`
- **Land cover:** `artificial_share`, `agriculture_share`, `forest_share`, `shrub_herbaceous_share`, `bare_sparse_share`, `wetland_share`, `water_share`, `dominant_class`, `share_sum`, `valid_fraction`, `n_valid_pixels`, `n_total_pixels`
- **Vegetation:** `ndvi_lag1`, `ndvi_missing_original`
- **Terrain:** `elev_mean`, `elev_std`, `slope_mean`, `slope_p90`, `aspect_south_share`, `aspect_missing_original`

`valid_fraction`, `n_valid_pixels`, `n_total_pixels`, and `share_sum` are diagnostics and should not automatically be treated as independent predictive features without examining their meaning. `dominant_class` is categorical and requires appropriate encoding if used by WP6.

## Results / Validation (Open Period)

The latest local run of the revised `cell_month_panel.py` and expanded `validate_panel.py` reported:

| Check | Observed result |
|---|---|
| In-mask WP2 cells | **6,555**; panel IDs exactly match current WP2 mask |
| Months | **92** (May 2016–December 2023) |
| Panel shape | **603,060 rows × 30 columns** |
| Coverage | Exactly **6,555 rows per month** and **92 months per cell** |
| Duplicate cell-month keys | **0** |
| Final missing values | **0** |
| Weather missing before imputation | **27,416** missing for each of `tmax`, `vpdmax`, `tdmean`, and `ppt` |
| NDVI originally missing | **2,199** cell-month observations |
| Terrain aspect originally missing | **368** cell-month observations (4 cells × 92 months) |
| Fire-positive cell-months | **4,402** of 603,060 (**~0.73%**) |
| Fire-negative cell-months | **598,658** (**~99.27%**) |
| Binary `fire` vs. WP2 `y_S30_any` | **0 disagreements** |
| Continuous `burn_frac` vs. WP2 `burn_frac_S30` | **0 disagreements; maximum absolute difference 0.0** |
| Feature checks | Numeric finiteness and bounds; land-cover shares sum to one; NDVI in [-1, 1]; valid pixel counts consistent; slope and aspect bounds; missing-indicator consistency — **all passed** |

Before switching to the official WP2 union-based burn fraction, **906** open-period cell-months differed from the prior `max(burn_frac_cell)` implementation, with a maximum absolute difference of about **0.3882**. The revised run eliminates these discrepancies while retaining the same **4,402** positive binary labels.

The standalone validation checks the panel's structure and values. It **cannot independently prove the absence of future-data leakage**, verify the scientific appropriateness of imputed values, or replace external validation of recorded-fire completeness. A zero-NaN output should not be interpreted as every feature being directly observed.

## Handoff to WP6 and DVC

Use the **open-period panel** for development according to the team's agreed chronological and spatial validation scheme. The **locked-inclusive panel** must not be used for routine model selection or hyperparameter tuning. Since only **~0.73%** of open-period observations are positive, WP6 should report metrics suitable for rare events (for example precision, recall, and PR-AUC) rather than accuracy alone.

WP5 processed outputs may be managed through DVC. If `cell_month_panel.parquet` has a `.dvc` pointer, remember that updating the local Parquet file **does not** update its DVC version automatically. Coordinate with the WP5 owner before running `dvc add` / `dvc push` and publishing new `.dvc` pointers. Commit the scripts and documentation separately from large Parquet binaries. DVC data objects must be pushed to the GCS remote before collaborators can retrieve a new version.

### Review status

The **Open Period** pipeline has passed the expanded local validation and exact comparison against WP2 S30 fire labels. The output-isolation logic has been added, but the `--locked` run has **not** been independently tested in this review. Forecast-time feature availability and the team's final target/threshold policy still require agreement before WP6 treats this as a finalized predictive dataset.
