# WP4 — Terrain, Land Cover, and Vegetation Features

## Overview

Work Package 4 (WP4) prepares terrain, land-cover, and vegetation predictors for the **European Wildfire and Property Intelligence Modeling** capstone project, with a focus on the Spain–Portugal study area. All processed features are aligned to the **6,225 in-mask grid cells** defined by WP2 and identified by `cell_id`.

WP4 produces three complementary datasets:

| Output | Granularity | Rows | Key |
|---|---|---:|---|
| `outputs/terrain_cell.parquet` | Static, one row per cell | 6,225 | `cell_id` |
| `outputs/landcover_cell.parquet` | Two land-cover epochs per cell | 12,450 | `cell_id`, `epoch` |
| `outputs/veg_cell_month.parquet` | Monthly lagged vegetation features per cell | 771,900 | `cell_id`, `year`, `month` |

These files have passed the structural and cross-dataset checks in `scripts/10_validate_wp4_integration.py` (zero failures and zero warnings). This is a **data-integrity check**, not a guarantee that the predictors are causally informative or that every edge pixel represents land perfectly.

## Data sources and methods

### Terrain — Copernicus DEM GLO-30

The terrain workflow downloads Copernicus GLO-30 elevation tiles, processes them in the study area, and aggregates terrain metrics onto WP2 cells. Features include mean and standard deviation of elevation, mean slope, a high-slope percentile, southern-facing aspect share, aspect concentration, and DEM coverage diagnostics. Terrain processing uses an equal-area projected working grid (EPSG:3035) for the WP2 alignment.

### Land cover — CORINE Land Cover (CLC), 2012 and 2018

CORINE categorical raster values are decoded using the corresponding value-attribute tables (VAT; e.g., `CODE_12` and `CODE_18`) rather than treating the GeoTIFF's integer pixel values as CLC codes directly. The workflow aggregates categories into fractional shares, including artificial surfaces, agriculture, forest, shrub/herbaceous cover, bare/sparse cover, wetlands, and water.

Each grid cell has one record for 2012 and one for 2018. Category shares sum to approximately 1 within each epoch (validated to ±0.001). **No mapping from the two CLC epochs to model months is imposed by WP4.** The downstream team must define a temporally defensible mapping before joining to monthly observations, taking care to avoid using future land-cover information.

### Vegetation — NASA MODIS MOD13A3.061 monthly NDVI

Monthly NDVI and `pixel_reliability` GeoTIFFs were obtained through NASA AppEEARS for source months **April 2016–July 2026** (124 months), in three download batches:

- April 2016–December 2019 (45 months)
- January 2020–December 2023 (48 months)
- January 2024–July 2026 (31 months)

NDVI is scaled using the GeoTIFF band scale (`0.0001` in the inspected files) and filtered to retain observations with pixel reliability codes **0 (good)** or **1 (marginal)**. Codes **2 (snow/ice)** and **3 (cloud)** are excluded. Valid pixels are assigned to WP2 grid cells by their pixel centers and summarized to mean NDVI, within-cell standard deviation, quality/valid-pixel fractions, and pixel-count diagnostics.

**Temporal convention:** `source_year` / `source_month` identify the MODIS observation month; `year` / `month` identify the **following** month in which the lag-one predictor `ndvi_lag1` is intended to be used. Thus an April 2022 observation appears as a May 2022 model feature. The target-date range is **May 2016–August 2026**. This one-month offset does not by itself guarantee leakage-free downstream modeling: forecast issuance dates and the temporal conventions for all other predictors and labels must also be checked.

## Repository structure

```text
WP4/
├── .gitignore
├── README.md
├── data/
│   ├── raw/                    # Large local inputs; not tracked by Git
│   │   ├── dem/
│   │   ├── corine/
│   │   └── modis/
│   │       ├── 2016_2019/
│   │       ├── 2020_2023/
│   │       └── 2024_2026/
│   └── reference/
│       └── ndvi_iberia_aoi.geojson
├── scripts/
│   ├── 01_download_dem.py
│   ├── 02_process_terrain.py
│   ├── 03_validate_terrain_maps.py
│   ├── 04_process_landcover.py
│   ├── 05_validate_landcover.py
│   ├── 06_prepare_ndvi_aoi.py
│   ├── 07_process_ndvi.py
│   ├── 08_diagnose_ndvi_quality.py
│   ├── 09_validate_ndvi_maps.py
│   └── 10_validate_wp4_integration.py
└── outputs/
    ├── terrain_cell.parquet
    ├── landcover_cell.parquet
    ├── veg_cell_month.parquet
    ├── terrain_validation_maps/
    ├── landcover_validation_maps/
    └── ndvi_validation_maps/
```

> The final Parquet files and most raw raster files are excluded from Git by `.gitignore`; downstream consumers need an **explicit data transfer** or a documented shared-storage path. Cloning this repository alone will not recreate those files.

## Processing scripts

| Script | Purpose |
|---|---|
| `01_download_dem.py` | Download DEM blocks for the study region |
| `02_process_terrain.py` | Aggregate elevation, slope, aspect, and coverage to cells |
| `03_validate_terrain_maps.py` | Generate terrain spatial-validation figures |
| `04_process_landcover.py` | Decode and aggregate CLC 2012/2018 classes |
| `05_validate_landcover.py` | Generate land-cover comparison/change figures |
| `06_prepare_ndvi_aoi.py` | Prepare AppEEARS area-of-interest geometry |
| `07_process_ndvi.py` | Process MODIS NDVI and pixel reliability into lag-one cell-month features |
| `08_diagnose_ndvi_quality.py` | Inspect quality flags behind completely missing NDVI cell-months |
| `09_validate_ndvi_maps.py` | Generate vegetation-seasonality, interannual-change, and missingness maps |
| `10_validate_wp4_integration.py` | Check schemas, coverage, dates, ranges, duplicate keys, and cell alignment |

## Reproducing the outputs

Run from the repository root in the project's Python environment (previous development used the `DSC180A` Conda environment). The processing scripts require the geospatial and data-analysis libraries used by the project, including `numpy`, `pandas`, `geopandas`, `rasterio`, `pyarrow`, and `matplotlib`; DEM download and land-cover scripts may have additional dependencies. See the scripts for source-specific inputs and configuration.

The WP2 grid (`wp2/outputs/cells.gpkg`, containing `cell_id` and `in_mask`) must exist before running WP4 aggregation. Large raw DEM, CORINE, and MODIS files must be placed under the expected directories first. The AppEEARS data are downloaded separately and are not reproduced automatically by merely running the scripts.

After raw inputs are prepared, run the relevant pipelines and then validate:

```powershell
conda activate DSC180A
python -u WP4/scripts/02_process_terrain.py
python -u WP4/scripts/03_validate_terrain_maps.py
python -u WP4/scripts/04_process_landcover.py
python -u WP4/scripts/05_validate_landcover.py
```

To process an individual MODIS download batch:

```powershell
python -u WP4/scripts/07_process_ndvi.py `
  --input-dir WP4/data/raw/modis/2016_2019 `
  --start 2016-04 --end 2019-12 `
  --output WP4/outputs/veg_2016_2019.parquet
```

Repeat with the two other batches and their corresponding dates. Combine the three batch outputs only after checking schema consistency, continuity, lag-one alignment, and unique keys. The verified merged deliverable is `outputs/veg_cell_month.parquet`. Run:

```powershell
python -u WP4/scripts/09_validate_ndvi_maps.py
python -u WP4/scripts/10_validate_wp4_integration.py
```

**Note:** The final batch-merging step was performed separately during development; it is not currently exposed as a dedicated numbered script. If full one-command reproducibility is required, this should be added before the next handoff.

## Final validation results

| Check | Result |
|---|---|
| Terrain | 6,225 rows; unique, non-null `cell_id`; feature-range and coverage checks passed |
| Land cover | 12,450 rows; exactly 6,225 cells for each of 2012 and 2018; shares sum to 1 (±0.001) |
| NDVI | 771,900 rows; all 124 source months present; 6,225 cells per source month; target date = source month + 1 |
| NDVI missingness | 60 / 771,900 cell-months (**0.0078%**) |
| NDVI duplicate keys | 0 for both source-month and target-month cell keys |
| Cross-dataset spatial keys | All three datasets contain the identical set of 6,225 `cell_id` values |
| Overall integration report | **0 failing checks, 0 warnings** |

Diagnostics on the 2016–2019 and 2020–2023 batches found that the completely missing NDVI cell-months were concentrated in high-elevation locations and had no pixels passing the reliability filter. Snow/ice and cloud flags explained those excluded observations. The 2024–2026 batch had 14 missing cell-months; its individual quality-code breakdown was not separately established in the recorded validation.

The NDVI map suite includes April/August 2022 snapshots, August-minus-April 2022 seasonal change, August 2022 minus August 2020 change, and full-period missingness frequency. The terrain and land-cover map suites are also kept under `outputs/`.

## Caveats for downstream modeling

1. **CLC epoch selection:** Do not blindly forward-fill the 2018 map to months before 2018 or use a future epoch without a defensible temporal policy. Agree on a mapping with the modeling team.
2. **NDVI availability and missing values:** The 60 missing grid-month NDVI observations remain `NaN` rather than being filled with zero. Decide on imputation or model-specific missing-data handling downstream.
3. **Pixel-center coverage diagnostic:** `spatial_coverage_approx` estimates coverage from assigned pixel centers and approximate pixel area, **not** exact polygon intersection. It exceeds 1 in 381,176 records; values above 1 do not imply physically greater than 100% coverage. Do not interpret this raw column as a bounded fractional predictor.
4. **Coastal and water cells:** Cell masks, raster-valid pixels, and land area are not identical. Investigate coastal or water-dominated cells if the model is sensitive to those areas.
5. **Different spatial and temporal scales:** DEM is static, CLC has two epochs, and NDVI is monthly. Preserve correct join cardinalities and feature-time semantics.
6. **Data distribution:** Raw datasets and output Parquet files are ignored by Git; arrange file transfer or shared storage with downstream work packages.

## Suggested join patterns

- Terrain: `monthly_df.merge(terrain, on="cell_id", how="left", validate="many_to_one")`
- NDVI: `monthly_df.merge(vegetation, on=["cell_id", "year", "month"], how="left", validate="one_to_one")` **only when** `monthly_df` itself has unique cell-month keys; otherwise choose the cardinality appropriate to the downstream label table.
- Land cover: first select or map an appropriate CLC epoch; then merge on `cell_id` with `validate="many_to_one"`. Do **not** merge both epochs using `cell_id` alone: that would duplicate each model record.

## Status

The three core WP4 outputs are processed, spatially reviewed, and structurally validated. The remaining handoff tasks are to publish the code and validation figures, transfer the ignored Parquet datasets securely, and agree on land-cover epoch and forecasting-date conventions with downstream modelers.
