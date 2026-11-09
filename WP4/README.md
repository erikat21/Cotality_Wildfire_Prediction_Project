# WP4 — Terrain, Land Cover, and Vegetation Features

## Overview

Work Package 4 (WP4) develops environmental predictors for the **European Wildfire and Property Intelligence Modeling** project, focusing on Spain and Portugal. All final features use the current WP2 0.1° grid and its **6,555 in-mask cells**, identified by the unchanged `cell_id` convention. The updated WP2 mask retains every grid cell that intersects Spanish or Portuguese land, including coastal and international-border cells; the previous 6,225-cell mask excluded 330 of these cells.

WP4 provides three feature tables for downstream wildfire-occurrence modeling:

| Final output | Granularity | Rows | Join keys |
|---|---|---:|---|
| `outputs/terrain_cell.parquet` | Static terrain, one row per cell | 6,555 | `cell_id` |
| `outputs/landcover_cell.parquet` | CORINE land cover, two epochs per cell | 13,110 | `cell_id`, `epoch` |
| `outputs/veg_cell_month.parquet` | Monthly NDVI lagged by one month | 812,820 | `cell_id`, `year`, `month` |

The three final Parquet datasets are **tracked with DVC**, not stored as raw Parquet blobs in Git. Git tracks their corresponding `.parquet.dvc` pointer files. After cloning and configuring access to the team's DVC remote, use `dvc pull` to retrieve the files.

The updated outputs passed `scripts/11_validate_wp4_integration.py` against the current WP2 mask: **0 failures and 0 warnings**. This establishes structural consistency, not causal validity or model predictive performance.

## Data sources and methods

### WP2 spatial definition

The WP2 grid contains **11,726** cells in total, of which **6,555** have `in_mask = 1`. The new mask includes **330 additional coastal and border cells**. WP2 also exposes `land_frac`, `iberia_share`, and `edge_type` (`interior`, `coast`, `border`, `coast_border`). These fields should remain available during downstream modeling, particularly because some included cells contain very little Spanish or Portuguese land.

Cell boundaries and terrain processing use **EPSG:3035** (ETRS89 / LAEA Europe) where area-preserving spatial operations are required. The `cell_id` indexing rule remains unchanged.

### Terrain — Copernicus DEM GLO-30

Copernicus GLO-30 elevation tiles are processed in buffered geographic blocks and aggregated to WP2 cells on a 30 m projected working lattice. The resulting features include `elev_mean`, `elev_std`, `slope_mean`, `slope_p90`, `aspect_south_share`, `aspect_resultant`, and supporting pixel-count/coverage diagnostics.

The updated mask required **29 DEM blocks**, including one additional southern-edge block (`cop30_w-006.00_s+34.00.tif`). All **6,555** cells have DEM pixels and non-missing elevation and slope features. **Four coastal cells** (`867`, `9380`, `9447`, `10041`) have undefined aspect features because `n_aspect_pixels = 0` after the minimum-slope directional-pixel criterion; their aspect values are deliberately kept as `NaN`, not changed to zero.

`dem_coverage_frac_cell` describes DEM data availability across the entire grid cell. A high value is **not** evidence that the cell is mostly land: coastal cells can contain sea-level raster pixels.

### Land cover — CORINE Land Cover (CLC), 2012 and 2018

CORINE 100 m categorical rasters are read using their raster value-attribute tables (VAT), including `CODE_12` and `CODE_18`. Raster class values are mapped to seven broad groups: artificial, agriculture, forest, shrub/herbaceous, bare/sparse, wetland, and water. Unclassified `CODE=999` is excluded.

Each epoch contains **6,555** cells, yielding **13,110** rows. Both epochs have no cells with zero classified pixels; land-cover shares sum to one for the valid classified data, and no duplicate `cell_id` + `epoch` keys were found. The mean classified-pixel `valid_fraction` was approximately **0.999525** in both epochs.

**WP4 does not assign a CORINE epoch to each model month.** The downstream modeling team must choose a defensible time mapping, avoiding use of future information as a historical predictor.

### Vegetation — NASA MODIS MOD13A3.061 monthly NDVI

MODIS monthly NDVI and `pixel_reliability` GeoTIFFs were downloaded via NASA AppEEARS for **April 2016–July 2026** (124 source months), originally in three batches:

| Batch | Source months | Number of months |
|---|---|---:|
| `2016_2019` | Apr 2016–Dec 2019 | 45 |
| `2020_2023` | Jan 2020–Dec 2023 | 48 |
| `2024_2026` | Jan 2024–Jul 2026 | 31 |

NDVI uses the GeoTIFF's band scale (observed as `0.0001`) and offset. Pixels are retained only when scaled NDVI is in `[-1,1]` and pixel reliability is **0 (good)** or **1 (marginal)**; codes **2 (snow/ice)** and **3 (cloud)** are excluded. Valid pixels are allocated to WP2 grid cells by pixel center, then summarized to `ndvi_lag1`, `ndvi_std`, valid/quality fractions, and pixel counts.

**Lag-one convention:** `source_year` and `source_month` identify the satellite observation; `year` and `month` identify the **following** model month. For example, April 2022 NDVI is used for May 2022. The output spans **May 2016–August 2026**, with **812,820** cell-month rows. This lag is intended to reduce same-month burn-scar leakage but does not by itself establish leakage-free forecasts.

#### Supplemental AOI for the expanded WP2 mask

After expanding the WP2 mask to 6,555 cells, two southern-edge cells lacked source MODIS pixels in all 124 months:

| `cell_id` | Grid center `(lon, lat)` | `land_frac` | Originally missing source months |
|---|---|---:|---:|
| `11627` | `(-5.4, 35.9)` | 0.6239 | 124 |
| `11628` | `(-5.3, 35.9)` | 0.0932 | 124 |

The supplemental request geometry was generated using `scripts/create_ndvi_supplement_aoi.py` from the full WP2 polygons for these cells with a **1,500 m buffer**. It is saved as `data/reference/ndvi_supplement_aoi.geojson` (EPSG:4326, one valid feature; approximately 295.01 km²). A separate AppEEARS download used the same MOD13A3.061 NDVI/Pixel Reliability layers and April 2016–July 2026 source-date range.

The supplemental GeoTIFFs were processed separately with `scripts/07_process_ndvi.py`, restricting `--cells` to the two target polygons. The resulting 248 supplemental records all contained valid NDVI and replaced the **248 originally zero-coverage** cell-month records in the expanded baseline dataset. All other **812,572** records remained unchanged, as verified by a dataframe equality check.

The final NDVI dataset contains **2,976 missing NDVI values** out of **812,820** rows (**approximately 0.366%**) and **zero records with `n_total_pixels = 0`**. The remaining missing observations have no valid pixels after the source/quality screening; they remain `NaN` and should not be filled with zero automatically. Some high-water-fraction coastal cells have limited reliable vegetation signal even when source pixels exist.

## Repository structure

```text
WP4/
├── .gitignore
├── README.md
├── data/
│   ├── raw/                              # Local raw DEM, CORINE and MODIS files
│   │   ├── dem/blocks/
│   │   ├── corine/{2012,2018}/
│   │   └── modis/
│   │       ├── 2016_2019/
│   │       ├── 2020_2023/
│   │       ├── 2024_2026/
│   │       └── supplement_6555/          # Supplemental AppEEARS downloads
│   └── reference/
│       ├── ndvi_iberia_aoi.geojson
│       └── ndvi_supplement_aoi.geojson
├── scripts/
│   ├── 01_download_dem.py
│   ├── 02_process_terrain.py
│   ├── 03_validate_terrain_maps.py
│   ├── 04_process_landcover.py
│   ├── 05_validate_landcover.py
│   ├── 06_prepare_ndvi_aoi.py
│   ├── 07_process_ndvi.py
│   ├── 08_diagnose_ndvi_quality.py
│   ├── 09_merge_ndvi_batches.py
│   ├── 10_validate_ndvi_maps.py
│   ├── 11_validate_wp4_integration.py
│   └── create_ndvi_supplement_aoi.py
└── outputs/
    ├── terrain_cell.parquet              # Final DVC-managed dataset
    ├── terrain_cell.parquet.dvc
    ├── landcover_cell.parquet            # Final DVC-managed dataset
    ├── landcover_cell.parquet.dvc
    ├── veg_cell_month.parquet            # Final DVC-managed dataset
    ├── veg_cell_month.parquet.dvc
    ├── terrain_validation_maps/
    ├── landcover_validation_maps/
    └── ndvi_validation_maps/
```

Temporary checkpoints, experimental Parquet files, intermediate monthly batches, and most raw inputs are not required for routine use of the final features. Do not commit API keys or GCS service-account credentials.

## Data availability and DVC

The final Parquet datasets are downloaded from the project's **Google Cloud Storage DVC remote**. Install the DVC GCS extension, then pull from the repository root:

```powershell
python -m pip install dvc dvc-gs
dvc remote list
dvc pull
```

The repository configuration specifies `gcs_remote` at `gs://cotality-wildfire-project/dvc_store`. Downloading requires read access to the referenced GCS objects. **Uploading a new dataset version requires separate authenticated write permission; public read access does not grant push access.**

```python
import pandas as pd

terrain = pd.read_parquet("WP4/outputs/terrain_cell.parquet")
landcover = pd.read_parquet("WP4/outputs/landcover_cell.parquet")
vegetation = pd.read_parquet("WP4/outputs/veg_cell_month.parquet")

print(terrain.shape)     # (6555, 12)
print(landcover.shape)   # (13110, 14)
print(vegetation.shape)  # (812820, 13)
```

Run the code from the repository root with a Parquet engine such as `pyarrow` available. The `epoch` dimension in land cover and the `year`/`month` dimension in vegetation must be handled explicitly when joining these outputs.

When publishing an updated validated version, run `dvc add` for the three final Parquet paths, upload with `dvc push` using authorized credentials, and commit the updated `.parquet.dvc` pointers to Git. Coordinate the upload and Git commit so teammates never receive a pointer to data that have not yet been pushed.

## Processing scripts

| Script | Purpose |
|---|---|
| `01_download_dem.py` | Plan/download buffered Copernicus DEM blocks for the WP2 mask |
| `02_process_terrain.py` | Calculate terrain metrics using per-block checkpoints and merge them |
| `03_validate_terrain_maps.py` | Generate spatial terrain diagnostic figures |
| `04_process_landcover.py` | Decode and aggregate CORINE 2012/2018 land-cover classes |
| `05_validate_landcover.py` | Generate land-cover spatial and change diagnostics |
| `06_prepare_ndvi_aoi.py` | Create the main AppEEARS AOI |
| `07_process_ndvi.py` | Convert monthly MODIS NDVI/reliability rasters to lag-one features |
| `08_diagnose_ndvi_quality.py` | Inspect quality codes for missing MODIS observations |
| `09_merge_ndvi_batches.py` | Merge the three original monthly NDVI batches after validation |
| `10_validate_ndvi_maps.py` | Generate NDVI spatial/temporal diagnostics |
| `11_validate_wp4_integration.py` | Validate final WP4 tables against the current WP2 mask |
| `create_ndvi_supplement_aoi.py` | Build buffered supplemental AOI for cells 11627 and 11628 |

## Reproducing and validating outputs

All commands below assume the repository root as the working directory and the required source rasters are already available. The WP2 grid (`wp2/outputs/cells.gpkg` and `cells.parquet`) must be generated for the **current 6,555-cell mask** before running WP4 processing.

### Terrain

Plan DEM blocks with the current grid and fetch any missing files only if the OpenTopography API key is configured:

```powershell
python -u WP4/scripts/01_download_dem.py --plan
# Optional when blocks are missing and OPENTOPO_API_KEY is set:
# python -u WP4/scripts/01_download_dem.py

python -u WP4/scripts/02_process_terrain.py `
  --checkpoints WP4/outputs/terrain_checkpoints_6555 `
  --out WP4/outputs/terrain_cell_6555.parquet
```

**Do not reuse checkpoints created for the old 6,225-cell mask.** A fresh checkpoint directory is required because existing per-block files are skipped by default. Validate the new result before replacing the DVC-managed final file.

### CORINE land cover

```powershell
python -u WP4/scripts/04_process_landcover.py
```

The current land-cover script writes to `outputs/landcover_cell.parquet` by default. If testing a rebuild, make a backup or change the output path in the script before running; it does not necessarily support an `--out` option.

### MODIS NDVI, supplemental processing and merge

The original source rasters must contain both monthly NDVI and `pixel_reliability` files. **Process each original batch separately:** scanning the whole `modis/` directory after adding `supplement_6555/` would find duplicate monthly layers.

```powershell
python -u WP4/scripts/07_process_ndvi.py `
  --input-dir WP4/data/raw/modis/2016_2019 `
  --start 2016-04 --end 2019-12 `
  --output WP4/outputs/veg_2016_2019_6555.parquet

python -u WP4/scripts/07_process_ndvi.py `
  --input-dir WP4/data/raw/modis/2020_2023 `
  --start 2020-01 --end 2023-12 `
  --output WP4/outputs/veg_2020_2023_6555.parquet

python -u WP4/scripts/07_process_ndvi.py `
  --input-dir WP4/data/raw/modis/2024_2026 `
  --start 2024-01 --end 2026-07 `
  --output WP4/outputs/veg_2024_2026_6555.parquet
```

The existing `09_merge_ndvi_batches.py` is not yet compatible with this expanded grid (see compatibility note below). To create a **temporary expanded baseline**, concatenate these three non-overlapping batches and check their keys and monthly coverage before writing:

```python
import pandas as pd
from pathlib import Path

out = Path("WP4/outputs")
parts = [pd.read_parquet(out / f"veg_{period}_6555.parquet")
         for period in ("2016_2019", "2020_2023", "2024_2026")]
base = pd.concat(parts, ignore_index=True)
assert len(base) == 6555 * 124
assert not base.duplicated(["cell_id", "year", "month"]).any()
assert base.groupby(["source_year", "source_month"]).size().eq(6555).all()
assert len(base.groupby(["source_year", "source_month"])) == 124
base.to_parquet(out / "veg_cell_month_6555.parquet", index=False)
```

This is a staged rebuilding step, not a replacement for the final integration validation. Preserve the current DVC-managed output until the supplemental replacement and full validation pass.

**Keep supplemental downloads separate from the original batches.** The NDVI processor checks for duplicate layers for any source month and must not be pointed at a directory that contains both the original and supplemental monthly rasters together. Restrict the processing grid to the two supplemental cells before processing their smaller AOI:

```python
import geopandas as gpd

cells = gpd.read_file("wp2/outputs/cells.gpkg")
target = cells[cells.cell_id.isin([11627, 11628])].copy()
assert len(target) == 2
target.to_file("WP4/data/reference/ndvi_supplement_cells.gpkg",
               layer="cells", driver="GPKG")
```

```powershell
python -u WP4/scripts/07_process_ndvi.py `
  --input-dir WP4/data/raw/modis/supplement_6555 `
  --cells WP4/data/reference/ndvi_supplement_cells.gpkg `
  --output WP4/outputs/veg_supplement_6555.parquet
```

Merge only matching target keys (`cell_id`, `year`, `month`) from the two supplemental cells, and only where the baseline has `n_total_pixels == 0`. Copy the **entire supplemental feature record**, including NDVI, QA fractions, pixel counts, and coverage diagnostics; do not replace only `ndvi_lag1`. Check that the original rows for every other key are identical before moving the merged output to `outputs/veg_cell_month.parquet`.

> **Compatibility note:** `09_merge_ndvi_batches.py` in the existing repository still hardcodes `EXPECTED_CELLS = 6225`. Do not run it against expanded 6,555-cell batches without first updating its expected grid count and validating its assumptions. It also merges only the original three batches; the supplemental AOI data need a separate key-based replacement step. Do not use its `--overwrite` option on the verified final output as-is.

### Final integration validation

```powershell
python -u WP4/scripts/11_validate_wp4_integration.py
```

For a staged run before replacing the formal outputs:

```powershell
python -u WP4/scripts/11_validate_wp4_integration.py `
  --terrain WP4/outputs/terrain_cell_6555.parquet `
  --landcover WP4/outputs/landcover_cell.parquet `
  --veg WP4/outputs/veg_cell_month_6555_merged.parquet
```

`11_validate_wp4_integration.py` reads the current WP2 in-mask IDs and checks row counts, unique keys, epoch and month coverage, lag-one dates, valid numeric ranges, defined and undefined aspect values, missing NDVI against valid pixel counts, and cross-table `cell_id` alignment. Some coastal cells have `NaN` aspect by design when no directional pixels exist.

## Final validation results (6,555-cell mask)

| Check | Verified result |
|---|---|
| WP2 grid | 11,726 total cells; 6,555 in-mask |
| Terrain | 6,555 rows; valid elevation/slope; 4 undefined aspect cells with `n_aspect_pixels = 0` |
| Land cover | 13,110 rows; 6,555 cells per epoch in 2012 and 2018; shares sum to 1 |
| NDVI | 812,820 rows; 124 source months, 6,555 cells/month; one-month lag verified |
| NDVI supplemental repair | 248 zero-coverage records replaced for two cells; other 812,572 records unchanged |
| NDVI missing | 2,976 / 812,820 (approximately **0.366%**) |
| NDVI zero source-pixel coverage | 0 records after supplemental merge |
| Duplicate keys and cross-table IDs | No duplicate keys; identical current WP2 in-mask IDs across all three tables |
| Full integration | **0 failures, 0 warnings** |

These figures are from the expanded-mask processing and final integration validation. Spatial diagnostic maps originally generated for the old mask may need regeneration to display the 330 additional cells accurately.

## Caveats for downstream modeling

1. **Coastal and border cells:** WP2 deliberately includes cells touching any Spanish or Portuguese land, even when `land_frac` is very small. Retain `land_frac` and `edge_type` in model development and review sensitivity to mixed land/sea cells.
2. **Terrain aspect:** The four undefined aspect values are not equivalent to zero southern-facing slope or zero aspect concentration. Impute or handle them with an explicit model-specific policy.
3. **NDVI missingness:** The remaining 2,976 missing NDVI records have no valid retained pixels and remain `NaN`. Do not silently convert them to zero. Reliability varies substantially among newly included coastal cells.
4. **Pixel-center method:** `spatial_coverage_approx` is a pixel-area/pixel-center approximation, not exact polygon intersection. It may exceed 1 and should not be treated as a physical fraction capped at 100%.
5. **Mixed land and sea:** MODIS pixel summaries use raster pixel centers and the full grid-cell geometry, not a precise ES/PT-land-only clip. High water fractions can make environmental summaries difficult to interpret.
6. **Temporal mapping and leakage:** DEM is static; CORINE has 2012 and 2018 epochs; NDVI is monthly with a one-month lag. Choose land-cover epochs and forecast times carefully to prevent future-information leakage.
7. **DVC and reproducibility:** DVC provides versioned final tables, but rebuilding from raw rasters requires the original external DEM, CORINE and AppEEARS downloads. The supplemental AOI file and its processing steps should be kept alongside the main workflow.

## Suggested downstream joins

- **Terrain:** `monthly.merge(terrain, on="cell_id", how="left", validate="many_to_one")`.
- **NDVI:** `monthly.merge(vegetation, on=["cell_id", "year", "month"], how="left", validate="one_to_one")`, when `monthly` has unique cell-month keys.
- **Land cover:** select an epoch or create a time-aware mapping first, then join with a cardinality check. **Do not join both epochs on `cell_id` alone**, as that duplicates model records.

## Status

All three expanded-grid WP4 deliverables have been processed and passed final structural/integration validation against WP2. They are ready for the downstream modeling team once the corresponding DVC objects have been uploaded to the shared GCS remote and the pointer files have been committed. Remaining decisions about model-specific missing-data treatment, CORINE temporal mapping, and the interpretation of mixed land/sea cells belong to downstream analysis.
