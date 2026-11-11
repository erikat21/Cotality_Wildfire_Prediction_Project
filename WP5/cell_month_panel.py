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

FIX NOTES (P0 review, 2026-11):
    P0-1: Spatial nearest-neighbor imputation now builds a separate valid/missing
          mask for each feature and preserves all original non-missing values.
    P0-2: Temporal filling uses chronological forward fill only; backward fill
          and whole-panel medians (potential future/test leakage) were removed.
    P0-3: NDVI/aspect original-missing indicators are preserved. Only selected
          physical predictors are imputed, not land-cover diagnostics or labels.
    Validation: Additional unique-key and input-schema checks prevent silent
          row duplication and make missing values visible rather than hiding them.

CURRENT PATHS AND COMMANDS (supersede the legacy names above):
    WP2 input: wp2/outputs/cells.parquet
    WP5 output: WP5/outputs/cell_month_panel.parquet
    python -u WP5/cell_month_panel.py             # Open years
    python -u WP5/cell_month_panel.py --locked    # Includes locked years
    Open mode writes to WP5/outputs/cell_month_panel.parquet.
    Locked mode writes to WP5/outputs/cell_month_panel_locked.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Setup multi-package directory paths based on your architecture
REPO_ROOT = Path(__file__).resolve().parents[1]
WP2_OUT = REPO_ROOT / "wp2" / "outputs"
WP3_OUT = REPO_ROOT / "wp3" / "outputs"
WP4_OUT = REPO_ROOT / "WP4" / "outputs"
WP5_OUT = REPO_ROOT / "WP5" / "outputs"

WEATHER_FEATURES = ["tmax", "vpdmax", "tdmean", "ppt"]
TERRAIN_FEATURES = [
    "elev_mean", "elev_std", "slope_mean", "slope_p90", "aspect_south_share"
]
LOCKED_YEARS = {2024, 2025, 2026}
KEYS = ["cell_id", "year", "month"]

def get_corine_epoch(year: int) -> int:
    """Keep the team's existing CLC mapping (2012 for <=2017, 2018 onward)."""
    return 2012 if year <= 2017 else 2018


def panel_months(include_locked: bool) -> list[tuple[int, int]]:
    months = []
    for year in range(2016, 2027):
        if year in LOCKED_YEARS and not include_locked:
            continue
        first = 5 if year == 2016 else 1
        last = 8 if year == 2026 else 12
        months.extend((year, month) for month in range(first, last + 1))
    return months

def read_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}. Check upstream outputs / dvc pull.")
    return pd.read_parquet(path)


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def require_unique(df: pd.DataFrame, keys: list[str], source: str) -> None:
    if df[keys].isna().any().any():
        raise ValueError(f"{source} has missing key values: {keys}")
    if df.duplicated(keys).any():
        raise ValueError(f"{source} has duplicate keys: {keys}")

# FIX P0-1: Correct the original spatial_impute() behavior.
# The old version chose valid_mask from cols_with_nans[0] for ALL features.
# That could overwrite good values and leave other features missing.
def spatial_impute(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Fill a feature's NaNs using the nearest *valid* cell in the same month.

    Each feature gets its own missing/valid masks, so valid observations are
    preserved. If no donors exist for a feature/month, NaNs remain unchanged.
    Diagnostics and labels must not be passed as feature_cols.
    """
    pieces = []
    for (_, _), group in df.groupby(["year", "month"], sort=False):
        group = group.copy()
        coordinates = group[["row", "col"]].to_numpy(dtype=float)
        for col in feature_cols:
            if col not in group.columns:
                continue
            if not pd.api.types.is_numeric_dtype(group[col]):
                raise TypeError(f"Spatial imputation requires numeric data: {col}")

            # FIX P0-1: Compute valid/missing masks PER COLUMN, not once per month.
            valid = group[col].notna().to_numpy()
            missing = ~valid
            if not missing.any() or not valid.any():
                continue

            tree = cKDTree(coordinates[valid])
            _, nearest_indices = tree.query(coordinates[missing], k=1)
            donor_values = group.loc[valid, col].to_numpy()
            # FIX P0-1: Update ONLY the current feature's missing observations.
            group.loc[missing, col] = donor_values[nearest_indices]
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)

# FIX P0-2: Avoid bfill() and global median fill using later/test periods.
def forward_fill_by_cell(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Only carry data from earlier months of the same cell."""
    # FIX P0-2: Sort by cell_id, year, month before any temporal fill.
    df = df.sort_values(KEYS, kind="stable").reset_index(drop=True)
    available = [c for c in feature_cols if c in df.columns]
    if available:
        df[available] = df.groupby("cell_id", sort=False)[available].ffill()
    return df

def print_missing(df: pd.DataFrame, label: str, columns: list[str]) -> None:
    counts = df[columns].isna().sum()
    nonzero = counts[counts.gt(0)]
    print(f"{label}: " + (nonzero.to_string() if len(nonzero) else "no missing values"))

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the WP5 cell-month panel")
    parser.add_argument(
        "--locked", action="store_true",
        help="Include locked years 2024-2026 (save to a separate output file)",
    )
    args = parser.parse_args()
    WP5_OUT.mkdir(parents=True, exist_ok=True)

    # Load cells and filter in-mask (from wp2)
    # FIX: Check that the WP2 mask has unique IDs and valid grid coordinates.
    # WP2 spatial backbone.
    cells = read_parquet(WP2_OUT / "cells.parquet", "WP2 cells")
    require_columns(cells, ["cell_id", "in_mask", "nuts2", "row", "col"], "WP2 cells")
    cells = cells.loc[cells["in_mask"].eq(1), ["cell_id", "nuts2", "row", "col"]].copy()
    require_unique(cells, ["cell_id"], "WP2 in-mask cells")
    if cells[["row", "col"]].isna().any().any():
        raise ValueError("WP2 grid row/col coordinates cannot be null")
    n_cells = len(cells)
    months = panel_months(args.locked)
    print(f"WP5: {n_cells:,} in-mask cells x {len(months)} months; locked={args.locked}")

    # Build spatio-temporal skeleton
    months_df = pd.DataFrame(months, columns=["year", "month"])
    # Cross join preserves exactly one row per in-mask cell and target month.
    panel = cells.merge(months_df, how="cross")
    require_unique(panel, KEYS, "Panel skeleton")
    expected_rows = n_cells * len(months)
    if len(panel) != expected_rows:
        raise AssertionError("Unexpected skeleton size")

    # Merge & impute weather (from wp3)
    # FIX P0-1: Impute weather predictors independently by month.
    # WP3: weather, one row per (cell_id, year, month).
    weather = read_parquet(WP3_OUT / "weather_cell_month.parquet", "WP3 weather")
    require_columns(weather, KEYS + WEATHER_FEATURES, "WP3 weather")
    require_unique(weather, KEYS, "WP3 weather")
    panel = panel.merge(
        weather[KEYS + WEATHER_FEATURES], on=KEYS, how="left", validate="one_to_one"
    )
    print_missing(panel, "Weather before imputation", WEATHER_FEATURES)
    panel = spatial_impute(panel, WEATHER_FEATURES)

    # Merge landcover (Epoch-mapped from WP4)
    # Original team mapping retained; review publication-time assumptions for WP6.
    # WP4: CLC shares for an explicitly selected epoch.
    panel["epoch"] = panel["year"].map(get_corine_epoch)
    landcover = read_parquet(WP4_OUT / "landcover_cell.parquet", "WP4 land cover")
    require_columns(landcover, ["cell_id", "epoch"], "WP4 land cover")
    require_unique(landcover, ["cell_id", "epoch"], "WP4 land cover")
    lc_features = [c for c in landcover.columns if c not in ("cell_id", "epoch")]
    panel = panel.merge(landcover, on=["cell_id", "epoch"], how="left", validate="many_to_one")
    if panel[lc_features].isna().any().any():
        raise ValueError("Missing CLC values after join; check WP2/WP4 alignment")

    # Merge vegetation (ndvi_lag1 from WP4)
    # WP4: vegetation is already lagged to the target year/month.
    vegetation = read_parquet(WP4_OUT / "veg_cell_month.parquet", "WP4 vegetation")
    require_columns(vegetation, KEYS + ["ndvi_lag1"], "WP4 vegetation")
    require_unique(vegetation, KEYS, "WP4 vegetation")
    panel = panel.merge(
        vegetation[KEYS + ["ndvi_lag1"]], on=KEYS,
        how="left", validate="one_to_one"
    )
    # FIX P0-3: Record missing NDVI BEFORE any imputation.
    panel["ndvi_missing_original"] = panel["ndvi_lag1"].isna().astype("int8")

    # Merge terrain features (from WP4)
    # WP4: static terrain; keep a flag for the four undefined-aspect cells.
    terrain = read_parquet(WP4_OUT / "terrain_cell.parquet", "WP4 terrain")
    require_columns(terrain, ["cell_id"] + TERRAIN_FEATURES, "WP4 terrain")
    require_unique(terrain, ["cell_id"], "WP4 terrain")
    panel = panel.merge(
        terrain[["cell_id"] + TERRAIN_FEATURES],
        on="cell_id", how="left", validate="many_to_one"
    )
    # FIX P0-3: Keep track of undefined aspect (4 coastal cells in WP4).
    panel["aspect_missing_original"] = panel["aspect_south_share"].isna().astype("int8")

    # Month-by-month spatial imputation for All WP4 features
    # FIX P0-3: Restrict it to actual model features (not diagnostic columns).
    # P0 fix: impute each physical predictor independently, not every numeric
    # diagnostic column. Preserve CLC shares and original missingness flags.
    wp4_impute_features = ["ndvi_lag1"] + TERRAIN_FEATURES
    panel = spatial_impute(panel, wp4_impute_features)

    # Numeric fallbacks (temporal fill + median fill) -- original strategy
    # FIX P0-2: Only causal ffill remains; remove bfill and full-panel median.
    # P0 fix: no bfill and no median fitted on the full/open+locked panel.
    # Missing cells with no observed donor remain NaN for honest validation.
    temporal_features = WEATHER_FEATURES + ["ndvi_lag1"]
    panel = forward_fill_by_cell(panel, temporal_features)
    print_missing(panel, "Predictor NaNs remaining", WEATHER_FEATURES + wp4_impute_features)

    # Drop temporary tracking columns (performed after label merge below).
    # Merge monthly fire labels (from wp2)
    # FIX P1-6: Use WP2's official spatially aggregated fire labels.
    # The original WP5 implementation used max(burn_frac_cell), which
    # underestimated the combined burned fraction when multiple fires
    # affected the same cell-month.
    #
    # WP2/03_labels.py merges overlapping fire polygons before
    # calculating burn_frac_S30, avoiding double-counted burned areas.

    labels = read_parquet(
        WP2_OUT / "labels.parquet",
        "WP2 official fire labels"
    )

    require_columns(
        labels,
        KEYS + ["y_S30_any", "burn_frac_S30"],
        "WP2 official fire labels"
    )

    require_unique(
        labels,
        KEYS,
        "WP2 official fire labels"
    )

    fire_monthly = labels[
        KEYS + ["y_S30_any", "burn_frac_S30"]
    ].rename(
        columns={
            "y_S30_any": "fire",
            "burn_frac_S30": "burn_frac"
        }
    )

    panel = panel.merge(
        fire_monthly,
        on=KEYS,
        how="left",
        validate="one_to_one"
    )

    # A cell-month absent from WP2 labels has no recorded qualifying fire.
    panel["fire"] = panel["fire"].fillna(0).astype("int8")
    panel["burn_frac"] = panel["burn_frac"].fillna(0.0)

    # Reorder columns to put primary keys and targets at the front
    # Preserve existing WP5 interface for its validator and WP6.
    panel = panel.drop(columns=["row", "col", "epoch"])
    front = ["cell_id", "year", "month", "nuts2", "fire", "burn_frac"]
    panel = panel[front + [c for c in panel.columns if c not in front]]
    require_unique(panel, KEYS, "Final panel")
    if len(panel) != expected_rows:
        raise AssertionError(f"Final panel has {len(panel):,} rows, expected {expected_rows:,}")

    # Export final panel to WP5 outputs
    # FIX P2-1: Separate Open and Locked outputs to prevent overwriting.
    # The original script wrote both modes to cell_month_panel.parquet.
    # The locked mode now writes to a separate file.
    if args.locked:
        destination = WP5_OUT / "cell_month_panel_locked.parquet"
    else:
        destination = WP5_OUT / "cell_month_panel.parquet"

    panel.to_parquet(destination, index=False)
    print("\nWP5 Pipeline completed successfully")
    print(f"WP5 panel saved to {destination}")
    print(f"Rows: {len(panel):,}; unique cells: {panel.cell_id.nunique():,}")
    print(f"Fire positives: {panel.fire.sum():,}")
    print("Original NDVI missing:", int(panel.ndvi_missing_original.sum()))
    print("Original aspect missing (cell-months):", int(panel.aspect_missing_original.sum()))
    print("Remaining null columns:")
    print(panel.isna().sum()[lambda x: x.gt(0)].to_string() or "none")

if __name__ == "__main__":
    main()