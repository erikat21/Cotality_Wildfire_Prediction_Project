# Cotality_Wildfire_Prediction_Project

## Setup
```
pip install dvc dvc-gs
```

Data (not committed): 
- `WP1/data/raw/effis_data_table_29thSept2026.csv` : needed to run process_effis_data.py in WP1
- `wp2/data/raw/effis_layer_29thSept2026/modis.ba.poly.dbf` : needed to run fire_cell.py in wp2
- `WP4/outputs/terrain_cell.parquet` : output from 02_process_terrain.py in WP4 needed for WP5
- `WP4/outputs/landcover_cell.parquet` : output from 04_process_landcover.py in WP4 needed for WP5
- `WP4/outputs/veg_cell_month.parquet` : output from 09_merge_ndvi_batches.py in WP4 needed for WP5

## Run (to get Data not committed)
```
dvc pull
```
