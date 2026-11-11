"""WP5 final panel validation.

Preserves the original WP5 structural, missing-value, and target checks.
Adds independent P1 checks for expected grid IDs, complete monthly coverage,
feature ranges, and the original-missingness indicator columns.

Run from the repository root:
    python -u WP5/validate_panel.py

Optional (when testing the locked panel):
    python -u WP5/validate_panel.py --locked

NOTE: This validates the *output panel*. It cannot prove the imputation
algorithm used only information available at forecast time; that requires
reviewing the building script and the issue-time conventions of the inputs.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# Setup paths relative to WP5 structure
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
WP5_OUT = HERE / "outputs"
PANEL_PATH = WP5_OUT / "cell_month_panel.parquet"
CELLS_PATH = REPO_ROOT / "wp2" / "outputs" / "cells.parquet"

EXPECTED_CELLS = 6555
REQUIRED_COLUMNS = {
    "cell_id", "year", "month", "nuts2", "fire", "burn_frac",
    "tmax", "vpdmax", "tdmean", "ppt", "ndvi_lag1",
    "elev_mean", "elev_std", "slope_mean", "slope_p90",
    "aspect_south_share", "ndvi_missing_original", "aspect_missing_original",
    "artificial_share", "agriculture_share", "forest_share",
    "shrub_herbaceous_share", "bare_sparse_share", "wetland_share", "water_share",
}
SHARE_COLUMNS = [
    "artificial_share", "agriculture_share", "forest_share",
    "shrub_herbaceous_share", "bare_sparse_share", "wetland_share", "water_share",
]


def require(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


def expected_months(locked):
    return pd.period_range("2016-05", "2026-08" if locked else "2023-12", freq="M")


def check_bounded(df, column, lower=None, upper=None, tolerance=1e-8):
    if column not in df.columns:
        return
    x = df[column]
    numeric = pd.to_numeric(x, errors="coerce")
    require(numeric.notna().all() and np.isfinite(numeric.to_numpy(dtype=float)).all(),
            f"{column} contains only finite numeric values")
    if lower is not None:
        require((numeric >= lower - tolerance).all(), f"{column} >= {lower}")
    if upper is not None:
        require((numeric <= upper + tolerance).all(), f"{column} <= {upper}")


def main():
    parser = argparse.ArgumentParser(description="Validate WP5 panel against the current WP2 grid.")
    parser.add_argument("--locked", action="store_true", help="Expect 2016-05 through 2026-08 (124 months).")
    parser.add_argument("--panel", type=Path, default=PANEL_PATH, help="Parquet panel to check.")
    parser.add_argument("--cells", type=Path, default=CELLS_PATH, help="Current WP2 cells.parquet.")
    args = parser.parse_args()

    print("--- Loading WP5 Final Panel for Validation ---")
    if not args.panel.exists():
        raise FileNotFoundError(f"Panel file not found at {args.panel}. Please run cell_month_panel.py first.")
    if not args.cells.exists():
        raise FileNotFoundError(f"WP2 grid missing at {args.cells}. The grid is required to validate cell IDs.")
    df = pd.read_parquet(args.panel)
    cells = pd.read_parquet(args.cells)
    print(f"Successfully loaded panel with shape: {df.shape}")

    # FIX P1-1: Require the expected schema before using columns for checks.
    require(REQUIRED_COLUMNS.issubset(df.columns),
            f"Required WP5 columns present (missing: {sorted(REQUIRED_COLUMNS - set(df.columns))})")

    # Dimension and keys check (original checks retained and made stricter)
    print("\nRunning dimension/keys check")
    require(df[["cell_id", "year", "month"]].notna().all().all(), "Primary keys are non-null")
    actual_cells = df["cell_id"].nunique()
    print(f"Total unique cells: {actual_cells:,} (Expected: {EXPECTED_CELLS:,})")
    require(actual_cells == EXPECTED_CELLS, f"Panel has {EXPECTED_CELLS:,} unique cells")
    duplicates = df.duplicated(subset=["cell_id", "year", "month"]).sum()
    print(f"Duplicate (cell_id, year, month) records: {duplicates}")
    require(duplicates == 0, "Zero duplicate (cell_id, year, month) records")

    # FIX P1-2: Compare against actual in_mask IDs, not only the number of cells.
    require({"cell_id", "in_mask"}.issubset(cells.columns), "WP2 grid has cell_id and in_mask")
    mask_ids = set(cells.loc[cells["in_mask"] == 1, "cell_id"])
    panel_ids = set(df["cell_id"])
    require(len(mask_ids) == EXPECTED_CELLS, "Current WP2 mask has 6,555 unique cell IDs")
    require(panel_ids == mask_ids, "Panel cell IDs exactly match current WP2 mask")

    # FIX P1-3: Check *every* expected month and all cells in that month.
    print("\nChecking time span and coverage...")
    require(df["year"].apply(lambda v: isinstance(v, (int, np.integer))).all(), "Year keys are integers")
    require(df["month"].apply(lambda v: isinstance(v, (int, np.integer))).all(), "Month keys are integers")
    require(df["month"].between(1, 12).all(), "Month keys are within 1..12")
    periods = pd.PeriodIndex.from_fields(year=df["year"].to_numpy(), month=df["month"].to_numpy(), freq="M")
    expected = expected_months(args.locked)
    actual = set(periods.unique())
    require(actual == set(expected), f"Exact expected {len(expected)} months present")
    require(len(df) == EXPECTED_CELLS * len(expected),
            f"Total rows = {EXPECTED_CELLS:,} x {len(expected)} = {EXPECTED_CELLS * len(expected):,}")
    month_counts = pd.Series(periods).value_counts()
    require(month_counts.eq(EXPECTED_CELLS).all(), "Each month has exactly 6,555 rows")
    months_per_cell = df.groupby("cell_id")["month"].count()
    print(f"Min time steps per cell: {months_per_cell.min()}")
    print(f"Max time steps per cell: {months_per_cell.max()}")
    require(months_per_cell.eq(len(expected)).all(), f"Each cell has exactly {len(expected)} months")

    # Missing value checks (upgrade original warning to a real assertion)
    print("\nChecking missing values")
    nan_counts = df.isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    if len(cols_with_nans):
        print("WARNING: Found columns with remaining NaNs:")
        print(cols_with_nans)
    else:
        print("Zero NaNs detected across all columns")
    require(cols_with_nans.empty, "No remaining NaNs across all columns")

    # Target variable checks (original checks retained)
    print("\nValidating target variables ('fire' & 'burn_frac')")
    unique_fires = set(df["fire"].unique())
    print(f"Unique values in 'fire': {sorted(unique_fires)}")
    require(unique_fires.issubset({0, 1}), "Target 'fire' belongs to {0, 1}")
    burn_min, burn_max = df["burn_frac"].min(), df["burn_frac"].max()
    print(f"Burn fraction range: [{burn_min:.4f}, {burn_max:.4f}]")
    check_bounded(df, "burn_frac", 0, 1)
    fire_counts = df["fire"].value_counts()
    fire_pct = df["fire"].mean() * 100
    print(f"Class Balance:\n{fire_counts.to_string()}")
    print(f"Positive Fire Event Rate: {fire_pct:.2f}%")

    # FIX P1-4: Validate meaningful numerical feature ranges.
    print("\nChecking feature ranges and numerical consistency...")
    for col in SHARE_COLUMNS:
        check_bounded(df, col, 0, 1)
    share_total = df[SHARE_COLUMNS].sum(axis=1)
    require(np.isclose(share_total.to_numpy(), 1, atol=1e-3).all(),
            "Seven CORINE land-cover shares sum to 1 (tolerance 0.001)")
    if "share_sum" in df:
        check_bounded(df, "share_sum", 0, 1.001, tolerance=1e-6)
        require(np.isclose(df["share_sum"], 1, atol=1e-3).all(), "Reported share_sum is approximately 1")
    for col in ("valid_fraction",):
        check_bounded(df, col, 0, 1)
    for col in ("n_valid_pixels", "n_total_pixels"):
        check_bounded(df, col, 0)
    if {"n_valid_pixels", "n_total_pixels"}.issubset(df):
        require((df["n_valid_pixels"] <= df["n_total_pixels"]).all(),
                "NDVI valid pixel counts do not exceed total pixel counts")
    check_bounded(df, "ndvi_lag1", -1, 1)
    check_bounded(df, "slope_mean", 0, 90)
    check_bounded(df, "slope_p90", 0, 90)
    check_bounded(df, "aspect_south_share", 0, 1)
    check_bounded(df, "elev_std", 0)
    check_bounded(df, "vpdmax", 0)
    check_bounded(df, "ppt", 0)
    for col in ("tmax", "tdmean", "elev_mean"):
        check_bounded(df, col)

    # FIX P1-5: Missingness indicators are flags, not imputed measurements.
    for flag in ("ndvi_missing_original", "aspect_missing_original"):
        vals = set(df[flag].unique())
        require(vals.issubset({0, 1}), f"{flag} is binary")
    aspect_count_per_cell = df.groupby("cell_id")["aspect_missing_original"].nunique()
    require(aspect_count_per_cell.eq(1).all(),
            "Original terrain aspect missingness is constant across months within each cell")

    print("\nEnsure all necessary features are present in the panel:")
    print(df.columns.tolist())
    print("\nValidation complete: all assertions passed successfully")
    print("NOTE: Output checks cannot independently establish absence of future-data leakage.")


if __name__ == "__main__":
    main()
