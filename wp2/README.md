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
python 04_plot_one_fire.py          # sanity check: the Rúa fire (id 278475) cut into grid cells
```

## cell_id
The grid matches the ERA5-Land download box (N 44.0, W -9.8, S 35.9, E 4.4): 82 rows x 143 columns, each cell centred on an ERA5-Land point.

`cell_id = row * 143 + col`, where `row = round((44.0 - lat) / 0.1)` and `col = round((lon + 9.8) / 0.1)`. Row 0 is the northern edge.

## Choices made
- **Mask:** a cell is kept (`in_mask = 1`) if any Spanish or Portuguese land falls inside it, so coastal, island and border cells are included (changed on 2026-11-04 after mentor feedback). This gives 6,555 cells (5,570 ES, 985 PT). The earlier rule (at least half land, at least half of that in ES/PT) gave 6,225 and left out 330 edge cells that can still burn. The Canaries, Azores and Madeira are outside the grid. Ceuta (ES) touches the southern edge of the grid and is included.
- **edge_type** marks the in-mask cells that need care: `interior` (6,187), `coast` = less than half land, mostly sea (286), `border` = some land in France, Andorra or Morocco (77), `coast_border` = both (5). For coast cells ERA5-Land may have no land value at the cell centre, so weather may need the nearest land point. To get the old mask back: `land_frac >= 0.5 and iberia_share >= 0.5`.
- **country / nuts2** are ES or PT for every in-mask cell (the ES/PT country or region with the most land in the cell), even when most of the land is French.
- **Areas** are measured in EPSG:3035 (equal area).
- **Fires** come from WP1's `fires_master` (ES/PT, at least 30 ha). Outlines come from the EFFIS shapefile, matched on `fire_id`.
- **Month** is the month of FIREDATE.
- **burn_frac** is burnt area divided by the whole cell area. Overlapping fires in the same cell-month are merged first so nothing is counted twice.
- **Labels** are made for fire size S = 30, 100, 383 ha and burnt share x = any, 1%, 2%, 5%, 10%. Cell-months with only fires smaller than S have `n_fires_S<S> = 0`, so they are easy to drop.

## Outputs (outputs/)
| file | one row per | key columns |
|---|---|---|
| cells.parquet / .gpkg | grid cell | cell_id, lon, lat, cell_area_ha, land_frac, iberia_area_ha, iberia_share, country, nuts2, in_mask, edge_type |
| fire_cell.parquet / .gpkg | fire x cell it touches | fire_id, cell_id, year, month, burnt_ha_in_cell, burn_frac_cell, centroid_in_cell |
| labels.parquet | cell-month with a fire of at least 30 ha | cell_id, year, month, n_fires_S*, burn_frac_S*, y_S{S}_{any,x01,x02,x05,x10} |
| label_counts.csv | year | positive cell-months for each S and x |
| fire_278475_cells.png | | one fire cut into cells, from 04_plot_one_fire.py |

Only cell-months with a fire are stored in `labels`. Every other in-mask cell-month is 0.

## Results
| positive cell-months | any | >=1% | >=2% | >=5% | >=10% |
|---|---|---|---|---|---|
| S = 30 ha | 6,528 | 3,242 | 2,174 | 1,194 | 740 |
| S = 100 ha | 3,855 | 2,940 | 2,046 | 1,173 | 728 |
| S = 383 ha | 2,087 | 1,707 | 1,548 | 1,129 | 708 |

5,623 fires were used. 40 fires in `fires_master` fall outside the grid (islands).
