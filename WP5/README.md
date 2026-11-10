# WP5: Spatio-Temporal Panel Generation & Validation

This work package (`WP5`) constructs the final, cell-month panel required for statistical analysis and machine learning model training in **WP6**. It integrates multi-source environmental, meteorological, topographical, and historical fire datasets across a standardized spatio-temporal grid.

## Repository Structure

```text
WP5/
├── cell_month_panel.py   # Main panel construction and imputation pipeline
├── validate_panel.py     # Standalone data integrity and validation suite
└── outputs/
    └── cell_month_panel.parquet  # Final generated modeling panel
```

## Setup
```
pip install pandas numpy scipy pyarrow
```

Data Requirements: must run wp1-wp4 to get data needed for wp5
- ../wp3/outputs/weather_cell_month.parquet    dense weather features
- ../wp2/data/cells.parquet                    in-mask spatial cells and coordinates
- ../WP4/outputs/landcover_cell.parquet        Corine landcover features
- ../WP4/outputs/terrain_cell.parquet          Copernicus terrain features
- ../WP4/outputs/veg_cell_month.parquet        NDVI vegetation features
- ../wp2/outputs/fire_cell.parquet             sparse monthly fire labels

## Run, in order
```
python cell_month_panel.py                  # Open training period (May 2016 - Dec 2023)
python cell_month_panel.py --locked         # Include test years (up to Aug 2026, for test step only) 
python validate_panel.py                    # Run structural, bounds, and NaN check assertions
```

## Choices made
- **Spatio-Temporal Skeleton**: Constructs a complete rectangular matrix of 6,555 valid in-mask cells across all monthly timesteps from May 2016 onward.
- **Spatial Imputation (cKDTree)**: Resolves spatial edge gaps month-by-month for both weather and environmental features by querying the nearest valid neighbor in grid space (row, col).
- **Safety Fallbacks**: Captures the tiny fraction of missing values that spatial nearest-neighbor search misses by applying temporal forward/backward filling (ffill/bfill) per cell across the timeline, followed by a global median fallback strictly restricted to numeric features.
- **Fire Aggregation & Bounding**: Aggregates monthly fire occurrences by cell, setting fire = 1 for active cell-months and 0 for unburned negatives. Burn fractions are clipped cleanly to a valid [0.0, 1.0] range to handle microscopic floating-point rounding artifacts.

## Outputs (outputs/)
| file | one row per | key columns |
|---|---|---|
| cell_month_panel.parquet | cell_id × month x year | cell_id, year, month, nuts2, fire, burn_frac, tmax, vpdmax, tdmean, ppt, artificial_share, agriculture_share, forest_share, shrub_herbaceous_share, bare_sparse_share, wetland_share, 'water_share', valid_fraction, n_valid_pixels, n_total_pixels, share_sum, ndvi_lag1, elev_mean, elev_std, slope_mean, slope_p90, aspect_south_share|

## Results/Validation
- **Dimensions**: Guarantees all 6,555 unique in_mask cells included in the panel with one row per month/year (603060 rows for open years, 812820 rows including test years), with zero duplicate primary keys
- Zero remaining NaN values
- **Target Distribution**: Binary target 'fire' verified within {0, 1} and continuous target 'burn_frac' bounded between 0.0 and 1.0