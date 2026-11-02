# WP3: ERA5-Land weather, monthly features (vn0)

Builds `weather_cell_month` — one row per cell per month with the four vn0 weather
features: `tmax`, `vpdmax`, `tdmean`, `ppt`. Joins to every other table on `cell_id`
(the key WP2 sets), plus `year` and `month`.

## Setup

### 1. Copernicus (CDS) account + `~/.cdsapirc`
1. Make a free account at <https://cds.climate.copernicus.eu/> and log in.
2. On your profile page copy your **API key** (looks like `xxxxxxxx-xxxx-...`).
3. Create `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <YOUR-API-KEY>
   ```
4. Accept the licence for **ERA5-Land** and **derived ERA5-Land daily statistics**
   once on the dataset pages (the first download otherwise 403s).

### 2. Python packages
    pip install "cdsapi>=0.7" xarray cfgrib eccodes pandas numpy pyarrow

`cfgrib`/`eccodes` read the GRIB files. On macOS, if eccodes fails to import, install
the library with `conda install -c conda-forge eccodes` (or `brew install eccodes`).

### 3. WP2 cells table (optional but recommended)
Step 2 reads `data/cells.parquet` to keep only in-mask cells and attach `country`/`nuts2`.
This is a copy of the file `wp2/01_build_cells.py` produces (6,225 in-mask cells). It is
kept inside WP3 so this package is self-contained; the cell_id rule below reproduces the
same mapping either way. To refresh it, re-run WP2 and copy its output:
`cp ../wp2/outputs/cells.parquet data/cells.parquet`. If the file is absent, step 2 falls
back to every land point; pass `--all-cells` to force that on purpose.

## Run, in order

    # Download in blocks (resumable; existing files are skipped)
    python 01_download_era5.py --block pilot       # 2016-2019  -> data/era5/
    python 01_download_era5.py --block training    # 2020-2023  -> data/era5/
    python 01_download_era5.py --block margins15   # 2015       -> data/era5/ (antecedent)
    python 01_download_era5.py --block test        # 2024-2025  -> data/era5_locked/
    python 01_download_era5.py --block final       # 2026 Jan-Aug -> data/era5_locked/

    # Build the monthly table (open years only until the test step)
    python 02_weather_features.py                  # 2016-05 .. 2023-12
    python 02_weather_features.py --locked         # add 2024-2026 (test step ONLY)

Check the request bodies without downloading:  `python 01_download_era5.py --block pilot --dry-run`

## The download (vn0 box)
`area = [44.0, -9.8, 35.9, 4.4]` in CDS order **[North, West, South, East]** —
82 lats × 143 lons = 11,726 grid points (sea points come back empty). This is the same
box WP2 cut its cells from, so each ERA5-Land point lands on one cell centre.

| feature | dataset | variable | rule |
|---|---|---|---|
| tmax | `derived-era5-land-daily-statistics` | `2m_temperature`, `daily_maximum` | mean of daily max |
| tdmean | `derived-era5-land-daily-statistics` | `2m_dewpoint_temperature`, `daily_mean` | mean of daily mean |
| vpdmax | derived from the two above | — | mean of daily max VPD |
| ppt | `reanalysis-era5-land` | `total_precipitation`, time `00:00` | sum of daily totals, ×1000 for mm |

- Temperature requests carry **no `data_format` key** (per the brief); GRIB is the
  default. Precipitation is requested as GRIB (`data_format: grib`).
- **Precip day-shift:** ERA5-Land `tp` accumulates from 00 UTC, so the 00:00 value on
  day D+1 is day D's total. Step 1 saves each month's `.next.grib` sidecar (next
  month's first 00:00 stamp) so August 2026 closes out from 1 Sep 2026; step 2 moves
  every stamp back one day before summing.

## Downloading in blocks (brief slide 17)
Scoring years **2024, 2025, 2026** are written to `data/era5_locked/` and are only read
when you pass `--locked` to step 2 — keep them out of every tuning run. 2015 is
downloaded (antecedent weather) into the open folder; it is not part of the panel so
step 2 skips it when aggregating.

## The cell_id rule (shared with WP2/WP4/WP5, never change it)
    row = round((44.0 - lat) / 0.1)      # row 0 = northernmost latitude
    col = round((lon + 9.8) / 0.1)       # col 0 = westernmost longitude
    cell_id = row * 143 + col

## Choices made (each is easy to change)
- **Lite tier.** Daily statistics for temperature + hourly 00:00 for rain. `vpdmax` is
  the lite approximation `es(tmax) − es(tdmean)` (Tetens), ~0.13 kPa under the hourly
  maximum but same ranking. **Never mix lite and full VPD in one fitted model.**
- **Units:** tmax, tdmean in °C (ERA5 Kelvin − 273.15); vpdmax in kPa; ppt in mm.
- **Panel window:** May 2016 → Aug 2026. 2015 is kept on disk for antecedent features
  but is not written into `weather_cell_month`.
- **Cells kept:** in-mask cells only (joined from WP2 `cells.parquet`); `--all-cells`
  keeps every land point.
- **Aggregation:** tmax/tdmean = mean of daily values; vpdmax = mean of daily VPD;
  ppt = sum of daily totals. `weather_cell_month_check.csv` flags any month with cells
  short of a full set of days.

## Outputs (outputs/)
| file | one row per | key columns |
|---|---|---|
| weather_cell_month.parquet | cell × month | cell_id, year, month, tmax, vpdmax, tdmean, ppt (+ country, nuts2) |
| weather_cell_month_check.csv | year-month | cells, short_months, value ranges, ppt_zero_cells |

## Not in WP3 (scope)
Full-tier hourly VPD, wind, NDVI/DEM/CORINE (those are WP4), and the panel join (WP5).
