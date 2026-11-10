from pathlib import Path
import numpy as np
import pandas as pd

# Setup paths relative to WP5 structure
HERE = Path(__file__).resolve().parent
WP5_OUT = HERE / "outputs"
PANEL_PATH = WP5_OUT / "cell_month_panel.parquet"

EXPECTED_CELLS = 6555


def main():
    print(f"--- Loading WP5 Final Panel for Validation ---")
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Panel file not found at {PANEL_PATH}. Please run cell_month_panel.py first."
        )

    df = pd.read_parquet(PANEL_PATH)
    print(f"Successfully loaded panel with shape: {df.shape}")

    # Dimension and keys check
    print("\nRunning dimension/keys check")
    
    # Verify unique cells
    actual_cells = df['cell_id'].nunique()
    print(f"Total unique cells: {actual_cells:,} (Expected: {EXPECTED_CELLS:,})")
    assert actual_cells == EXPECTED_CELLS, f"Cell count mismatch! Got {actual_cells}, expected {EXPECTED_CELLS}."

    # Check for duplicate primary keys (cell_id, year, month)
    duplicates = df.duplicated(subset=['cell_id', 'year', 'month']).sum()
    print(f"Duplicate (cell_id, year, month) records: {duplicates}")
    assert duplicates == 0, f"Found {duplicates} duplicate records"

    # Missing value checks
    print("\nChecking missing values")
    nan_counts = df.isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    
    if len(cols_with_nans) > 0:
        print("warning: Found columns with remaining NaNs:")
        print(cols_with_nans)
    else:
        print("Zero NaNs detected across all columns")

    # Target variable checks
    print("\nValidating target variables ('fire' & 'burn_frac')")
    
    # Check binary bounds for 'fire'
    unique_fires = df['fire'].unique()
    print(f"Unique values in 'fire': {unique_fires}")
    assert set(unique_fires).issubset({0, 1}), "Target 'fire' contains values outside {0, 1}!"

    # Check range bounds for 'burn_frac'
    burn_min, burn_max = df['burn_frac'].min(), df['burn_frac'].max()
    print(f"Burn fraction range: [{burn_min:.4f}, {burn_max:.4f}]")
    assert df['burn_frac'].between(0.0, 1.0).all(), "Target 'burn_frac' contains values outside [0.0, 1.0]!"

    # Print class balance summary
    fire_counts = df['fire'].value_counts()
    fire_pct = df['fire'].mean() * 100
    print(f"Class Balance:\n{fire_counts.to_string()}")
    print(f"Positive Fire Event Rate: {fire_pct:.2f}%")

    # Time range checks
    print("\nChecking time span and coverage...")
    min_yr, max_yr = df['year'].min(), df['year'].max()
    print(f"Year range covered: {min_yr} to {max_yr}")
    
    months_per_cell = df.groupby('cell_id')['month'].count()
    print(f"Min time steps per cell: {months_per_cell.min()}")
    print(f"Max time steps per cell: {months_per_cell.max()}")

    print("\nEnsure all necessary features are present in the panel:")
    print(df.columns.tolist())

    print("\nValidation complete: all assertions passed successfully")

if __name__ == "__main__":
    main()