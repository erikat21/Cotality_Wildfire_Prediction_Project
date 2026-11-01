# WP2: Grid, Fire-by-Cell Overlay and Labels

Builds the 0.1° grid that every other work package joins on, works out how much of each grid cell each fire burned, and turns that into fire / no-fire labels for each cell-month.

## Setup
```
pip install geopandas pyogrio pyarrow matplotlib
```
Data (not committed):
- `../data/raw/effis_layer_29thSept2026/modis.ba.poly.shp`: EFFIS burnt-area polygons (unzip `effis_layer_29thSept2026.zip`)
- `../data/processed/fires_master.parquet`: output of WP1
- `data/boundaries/cntrg.json` and `data/boundaries/nutsrg_2.json`: Eurostat NUTS 2024 (1:3M), from `https://raw.githubusercontent.com/eurostat/Nuts2json/master/pub/v2/2024/4326/03M/<name>.json`

## Run, in order
```
python 01_build_cells.py
python 02_fire_cell.py --fires ../data/raw/effis_layer_29thSept2026/modis.ba.poly.shp --master ../data/processed/fires_master.parquet
python 03_labels.py
```

## cell_id
The grid matches the ERA5-Land download box (N 44.0, W -9.8, S 35.9, E 4.4): 82 rows x 143 columns, each cell centred on an ERA5-Land point.

`cell_id = row * 143 + col`, where `row = round((44.0 - lat) / 0.1)` and `col = round((lon + 9.8) / 0.1)`. Row 0 is the northern edge.

## Choices made
- **Mask:** a cell is kept (`in_mask = 1`) if at least half of it is land and at least half of that land is in Spain or Portugal. This gives 6,225 cells (5,293 ES, 932 PT). The Canaries, Azores and Madeira are outside the grid.
- **Areas** are measured in EPSG:3035 (equal area).
- **Fires** come from WP1's `fires_master` (ES/PT, at least 30 ha). Outlines come from the EFFIS shapefile, matched on `fire_id`.
- **Month** is the month of FIREDATE.
- **burn_frac** is burnt area divided by the whole cell area. Overlapping fires in the same cell-month are merged first so nothing is counted twice.
- **Labels** are made for fire size S = 30, 100, 383 ha and burnt share x = any, 1%, 2%, 5%, 10%. Cell-months with only fires smaller than S have `n_fires_S<S> = 0`, so they are easy to drop.

## Outputs (outputs/)
| file | one row per | key columns |
|---|---|---|
| cells.parquet / .gpkg | grid cell | cell_id, lon, lat, cell_area_ha, land_frac, country, nuts2, in_mask |
| fire_cell.parquet / .gpkg | fire x cell it touches | fire_id, cell_id, year, month, burnt_ha_in_cell, burn_frac_cell, centroid_in_cell |
| labels.parquet | cell-month with a fire of at least 30 ha | cell_id, year, month, n_fires_S*, burn_frac_S*, y_S{S}_{any,x01,x02,x05,x10} |
| label_counts.csv | year | positive cell-months for each S and x |

Only cell-months with a fire are stored in `labels`. Every other in-mask cell-month is 0.

## Results
| positive cell-months | any | >=1% | >=2% | >=5% | >=10% |
|---|---|---|---|---|---|
| S = 30 ha | 6,459 | 3,220 | 2,164 | 1,190 | 739 |
| S = 100 ha | 3,820 | 2,919 | 2,036 | 1,169 | 727 |
| S = 383 ha | 2,070 | 1,694 | 1,538 | 1,125 | 707 |

5,623 fires were used. 40 fires in `fires_master` fall outside the grid (islands).
