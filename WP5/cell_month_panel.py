"""
WP5: Build a cell-month panel for statistical analysis and model training in WP6

Merges the processed weather panel (weather_cell_month.parquet), 
Corine landcover data (landcover_cell.parquet), Copernicus terrain data (terrain_cell.parquet), 
NDVI vegetation data (veg_cell_month.parquet), and monthly fire labels (fire_cell.parquet)
onto a complete 6,555 in-mask cell skeleton across the panel timeline (cells.parquet).

Inputs (outputs/ or data/):
    outputs/weather_cell_month.parquet    dense weather features
    data/cells.parquet                    in-mask spatial cells and coordinates
    outputs/landcover_cell.parquet        Corine landcover features
    outputs/terrain_cell.parquet          Copernicus terrain features
    outputs/veg_cell_month.parquet        NDVI vegetation features
    outputs/fire_cell.parquet             sparse monthly fire labels

Outputs (outputs/):
    final_wildfire_panel.parquet          complete panel with one row per cell per month 

Run:
    python 03_build_final_panel.py                  # Open years (May 2016 - Dec 2023)
    python 03_build_final_panel.py --locked         # Include test years (up to Aug 2026)

"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Setup multi-package directory paths based on your architecture
HERE = Path(__file__).resolve().parent.parent  # Adjust if root differs
WP2_OUT = HERE / "wp2" / "outputs"
WP3_OUT = HERE / "wp3" / "outputs"
WP4_OUT = HERE / "WP4" / "outputs"
WP5_OUT = HERE / "WP5" / "outputs"

WP5_OUT.mkdir(parents=True, exist_ok=True)

LOCKED_YEARS = {2024, 2025, 2026}

def get_corine_epoch(year: int) -> int:
    return 2012 if year <= 2017 else 2018


def panel_months(use_locked: bool):
    """Yields valid (year, month) tuples based on locked status."""
    for year in range(2016, 2027):
        first = 5 if year == 2016 else 1
        last = 8 if year == 2026 else 12
        for month in range(first, last + 1):
            if year in LOCKED_YEARS and not use_locked:
                continue
            yield year, month

def spatial_impute(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Performs month-by-month spatial imputation using cKDTree in grid space (row, col)
    to fill missing values from the nearest valid neighbor.
    """
    imputed_dfs = []
    for (year, month), group in df.groupby(["year", "month"]):
        group = group.copy()
        cols_with_nans = [col for col in feature_cols if group[col].isna().any()]
        
        if cols_with_nans:
            valid_mask = group[cols_with_nans[0]].notna()
            valid_pts = group[valid_mask]
            missing_pts = group[~valid_mask]
            
            if len(missing_pts) > 0 and len(valid_pts) > 0:
                tree = cKDTree(valid_pts[["row", "col"]].to_numpy())
                _, idxs = tree.query(missing_pts[["row", "col"]].to_numpy(), k=1)
                for col in cols_with_nans:
                    group.loc[~valid_mask, col] = valid_pts[col].iloc[idxs].values
                    
        imputed_dfs.append(group)
    return pd.concat(imputed_dfs, ignore_index=True)

def main():
    parser = argparse.ArgumentParser(description="Build final wildfire modeling panel for WP5.")
    parser.add_argument("--locked", action="store_true", help="Include 2024-2026 test period.")
    args = parser.parse_args()

    print(f"Building Final Spatio-Temporal Panel for WP5 (Locked: {args.locked})")

    # Load cells and filter in-mask (from wp2)
    cells_path = WP2_OUT / "cells.parquet"
    cells_df = pd.read_parquet(cells_path)
    in_mask_cells = cells_df[cells_df["in_mask"] == 1][[
        "cell_id", "nuts2", "row", "col"
    ]].copy()
    print(f"Loaded {len(in_mask_cells):,} in-mask cells from WP2.")

    # Build spatio-temporal skeleton
    valid_time_steps = list(panel_months(args.locked))
    skeleton_rows = []
    for cell in in_mask_cells.itertuples(index=False):
        for year, month in valid_time_steps:
            skeleton_rows.append({
                "cell_id": cell.cell_id,
                "year": year,
                "month": month,
                "nuts2": cell.nuts2,
                "row": cell.row,
                "col": cell.col
            })
    full_skeleton = pd.DataFrame(skeleton_rows)

    # Merge & impute weather (from wp3)
    weather_df = pd.read_parquet(WP3_OUT / "weather_cell_month.parquet").drop(columns=["country", "nuts2"], errors="ignore")
    weather_full = pd.merge(full_skeleton, weather_df, on=["cell_id", "year", "month"], how="left")

    print("Imputing weather features spatially (cKDTree)...")
    weather_feats = ["tmax", "vpdmax", "tdmean", "ppt"]
    weather_imputed = spatial_impute(weather_full, weather_feats)

    # Merge landcover (Epoch-mapped from WP4)
    weather_imputed['epoch'] = weather_imputed['year'].apply(get_corine_epoch)
    landcover_df = pd.read_parquet(WP4_OUT / "landcover_cell.parquet")
    weather_land = pd.merge(weather_imputed, landcover_df, on=['cell_id', 'epoch'], how='left')

    # Merge vegetation (ndvi_lag1 from WP4)
    veg_df = pd.read_parquet(WP4_OUT / "veg_cell_month.parquet")
    sub_veg = veg_df[['cell_id', 'year', 'month', 'ndvi_lag1']].copy()
    weather_land_veg = pd.merge(weather_land, sub_veg, on=['cell_id', 'year', 'month'], how='left')

    # Merge terrain features (from WP4)
    terrain_df = pd.read_parquet(WP4_OUT / "terrain_cell.parquet")
    cols_terrain = ['cell_id', 'elev_mean', 'elev_std', 'slope_mean', 'slope_p90', 'aspect_south_share']
    sub_terrain = terrain_df[[c for c in cols_terrain if c in terrain_df.columns]].copy()
    weather_land_veg_terrain = pd.merge(weather_land_veg, sub_terrain, on=['cell_id'], how='left')

    # Month-by-month spatial imputation for All WP4 features
    print("Running month-by-month spatial imputation for remaining features...")
    meta_cols = ["cell_id", "nuts2", "row", "col", "year", "month", "epoch"]
    weather_cols = ["tmax", "vpdmax", "tdmean", "ppt"]
    feature_cols = [c for c in weather_land_veg_terrain.columns if c not in meta_cols + weather_cols]

    panel = spatial_impute(weather_land_veg_terrain, feature_cols)

    # Numeric fallbacks (temporal fill + median fill)
    numeric_feature_cols = panel[feature_cols].select_dtypes(include=[np.number]).columns
    panel[numeric_feature_cols] = panel[numeric_feature_cols].groupby(panel["cell_id"])[numeric_feature_cols].transform(
        lambda x: x.ffill().bfill()
    )
    panel[numeric_feature_cols] = panel[numeric_feature_cols].fillna(panel[numeric_feature_cols].median())

    # Drop temporary tracking columns
    panel = panel.drop(columns=['row', 'col', 'epoch'], errors='ignore')

    # Merge monthly fire labels (from wp2)
    print("Aggregating and merging fire labels...")
    fire_cell_df = pd.read_parquet(WP2_OUT / "fire_cell.parquet")
    fire_monthly = fire_cell_df.groupby(['cell_id', 'year', 'month']).agg(
        burn_frac=('burn_frac_cell', 'max')
    ).reset_index()
    fire_monthly['burn_frac'] = fire_monthly['burn_frac'].clip(lower=0.0, upper=1.0)
    fire_monthly['fire'] = 1

    panel_cell_month = pd.merge(panel, fire_monthly, on=['cell_id', 'year', 'month'], how='left')
    panel_cell_month['fire'] = panel_cell_month['fire'].fillna(0).astype(int)
    panel_cell_month['burn_frac'] = panel_cell_month['burn_frac'].fillna(0.0)

    # Reorder columns to put primary keys and targets at the front
    front_cols = ["cell_id", "year", "month", "nuts2", "fire", "burn_frac"]
    other_cols = [c for c in panel_cell_month.columns if c not in front_cols]

    panel_cell_month = panel_cell_month[front_cols + other_cols]

    # Export final panel to WP5 outputs
    out_path = WP5_OUT / "cell_month_panel.parquet"
    panel_cell_month.to_parquet(out_path, index=False)

    print("\nWP5 Pipeline completed successfully")
    print(f"Output saved to: {out_path}")

if __name__ == "__main__":
    main()