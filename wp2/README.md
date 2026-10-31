# WP2: grid, fire-by-cell overlay and labels (vn0)

Builds the three WP2 tables from the vn0 brief: `cells`, `fire_cell` and `labels`.

## Setup
    pip install geopandas pyogrio pyarrow matplotlib

Boundary files go in `data/boundaries/` (not committed; add `data/` and `outputs/` to .gitignore):
- `cntrg.json` and `nutsrg_2.json`: Eurostat NUTS 2024 at 1:3M, from
  https://raw.githubusercontent.com/eurostat/Nuts2json/master/pub/v2/2024/4326/03M/<name>.json

## Run, in order
    python 01_build_cells.py
    python 02_fire_cell.py --fires ~/Desktop/Capstone/fires/modis.ba.poly.shp
    python 03_labels.py

## The cell_id rule (shared with WP3, WP4, WP5)
Grid = the ERA5-Land download box [N 44.0, W -9.8, S 35.9, E 4.4], 0.1°, 82 rows x 143 cols.
Each cell is centred on an ERA5-Land point and spans ±0.05°.
`row = round((44.0 - lat) / 0.1)`, `col = round((lon + 9.8) / 0.1)`, `cell_id = row * 143 + col`.
Row 0 is the northernmost latitude, the order ERA5 stores latitude in.

## Choices made (each is easy to change)
- **Mask:** `in_mask = 1` if land_frac >= 0.5 and >= 50% of the cell's land is Spain + Portugal together. Gives 6,225 cells (5,293 ES, 932 PT), including the Balearics; the Canaries, Azores and Madeira are outside the box.
- **Areas** are measured in EPSG:3035 (equal area). Cell areas run from 8,912 ha (north) to 10,017 ha (south).
- **burn_frac** divides by the whole cell area, not the land area. `land_area_ha` is stored so the other choice is one line.
- **Fire size** S uses EFFIS `AREA_HA`. `fire_cell_check.csv` compares it with the polygon area.
- **Month** = month of FIREDATE (start date).
- **Overlapping fires** in the same cell-month are merged before measuring, so ground is not counted twice.
- **Small-fire-only cell-months** are rows of `labels` where `n_fires_S<S> == 0`: keep them as 0, or drop them in the fit.
- `fire_cell` keeps every touched cell, including those outside the mask, so each fire's pieces add up to its full area. Filter on `in_mask` downstream.

## Outputs (outputs/)
| file | one row per | key columns |
|---|---|---|
| cells.parquet / .gpkg | grid cell (11,726) | cell_id, lon, lat, cell_area_ha, land_frac, country, share_in_country, iberia_share, nuts2, in_mask |
| fire_cell.parquet / .gpkg | fire x cell it touches | fire_id, cell_id, year, month, fire_area_ha, burnt_ha_in_cell, burn_frac_cell, centroid_in_cell |
| labels.parquet | cell-month with any fire >= 30 ha | cell_id, year, month, n_fires_S{30,100,383}, burn_frac_S..., y_S{S}_{any,x01,x02,x05,x10} |
| label_counts.csv | year | positive cell-months for every S and x |
