"""Merge and validate three MODIS NDVI batches for WP4.

Run from any working directory:
    python WP4/scripts/09_merge_ndvi_batches.py

Required intermediate inputs (regenerate with 07_process_ndvi.py if deleted):
    WP4/outputs/veg_2016_2019.parquet
    WP4/outputs/veg_2020_2023.parquet
    WP4/outputs/veg_2024_2026.parquet

Output:
    WP4/outputs/veg_cell_month.parquet

The output is written only after all validation checks pass. The existing
output is replaced atomically (on the same filesystem).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "WP4" / "outputs"
BATCHES = (
    ("veg_2016_2019.parquet", "2016-04", "2019-12"),
    ("veg_2020_2023.parquet", "2020-01", "2023-12"),
    ("veg_2024_2026.parquet", "2024-01", "2026-07"),
)
EXPECTED_CELLS = 6225
EXPECTED_MONTHS = pd.period_range("2016-04", "2026-07", freq="M")
REQUIRED_COLUMNS = {
    "cell_id", "year", "month", "source_year", "source_month",
    "ndvi_lag1", "ndvi_std", "valid_fraction", "quality_pass_fraction",
    "n_valid_pixels", "n_total_pixels", "n_source_valid_pixels",
    "spatial_coverage_approx",
}


class ValidationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def month_period(frame: pd.DataFrame, year_col: str, month_col: str) -> pd.PeriodIndex:
    return pd.PeriodIndex.from_fields(
        year=frame[year_col].to_numpy(),
        month=frame[month_col].to_numpy(),
        freq="M",
    )


def verify_batch(df: pd.DataFrame, name: str, start: str, end: str,
                 reference_columns: set[str] | None,
                 reference_ids: set[int] | None) -> tuple[set[str], set[int]]:
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    require(not missing_cols, f"{name}: missing columns: {sorted(missing_cols)}")
    if reference_columns is not None:
        require(set(df.columns) == reference_columns, f"{name}: schema differs from first batch")

    key_cols = ["cell_id", "source_year", "source_month", "year", "month"]
    require(df[key_cols].notna().all().all(), f"{name}: null key fields")
    for column in key_cols:
        require(pd.api.types.is_integer_dtype(df[column]), f"{name}: {column} must have integer dtype")
    require(df["source_month"].between(1, 12).all() and df["month"].between(1, 12).all(),
            f"{name}: month outside 1..12")
    require(not df.duplicated(["cell_id", "source_year", "source_month"]).any(),
            f"{name}: duplicate source cell-month keys")
    require(not df.duplicated(["cell_id", "year", "month"]).any(),
            f"{name}: duplicate target cell-month keys")

    periods = month_period(df, "source_year", "source_month")
    expected = pd.period_range(start, end, freq="M")
    require(len(df) == len(expected) * EXPECTED_CELLS,
            f"{name}: expected {len(expected) * EXPECTED_CELLS:,} rows, found {len(df):,}")
    actual = set(periods.unique())
    require(actual == set(expected), f"{name}: missing/unexpected source months: "
            f"missing={sorted(set(expected) - actual)}, extra={sorted(actual - set(expected))}")
    counts = df.groupby(["source_year", "source_month"]).size()
    require(counts.eq(EXPECTED_CELLS).all(), f"{name}: expected {EXPECTED_CELLS} rows in every month")

    ids = set(df["cell_id"].astype(int))
    require(len(ids) == EXPECTED_CELLS, f"{name}: expected {EXPECTED_CELLS} unique cell IDs")
    require(df.groupby("cell_id").size().eq(len(expected)).all(),
            f"{name}: not all cells occur once per source month")
    if reference_ids is not None:
        require(ids == reference_ids, f"{name}: cell_id set differs from first batch")

    target = month_period(df, "year", "month")
    require((target == periods + 1).all(), f"{name}: target month is not exactly source + 1")

    for col in ("ndvi_lag1", "ndvi_std", "valid_fraction", "quality_pass_fraction",
                "spatial_coverage_approx"):
        require(pd.api.types.is_numeric_dtype(df[col]), f"{name}: nonnumeric {col}")
        require(not np.isinf(df[col].to_numpy(dtype=float)).any(), f"{name}: infinite values in {col}")
    require(df["ndvi_lag1"].dropna().between(-1, 1).all(), f"{name}: NDVI out of [-1, 1]")
    require(df["ndvi_std"].dropna().ge(0).all(), f"{name}: negative NDVI std")
    require(df["valid_fraction"].dropna().between(0, 1).all(), f"{name}: invalid valid_fraction")
    require(df["quality_pass_fraction"].dropna().between(0, 1).all(),
            f"{name}: invalid quality_pass_fraction")
    require(df["spatial_coverage_approx"].dropna().ge(0).all(),
            f"{name}: negative spatial_coverage_approx")
    for col in ("n_valid_pixels", "n_source_valid_pixels", "n_total_pixels"):
        require(df[col].notna().all() and df[col].ge(0).all(), f"{name}: invalid {col}")
    require((df["n_valid_pixels"] <= df["n_source_valid_pixels"]).all() and
            (df["n_source_valid_pixels"] <= df["n_total_pixels"]).all(),
            f"{name}: inconsistent pixel counts")
    require((df["ndvi_lag1"].isna() == df["n_valid_pixels"].eq(0)).all(),
            f"{name}: missing NDVI inconsistent with zero valid pixels")

    print(f"PASS {name}: {len(df):,} rows, {len(expected)} months, "
          f"missing NDVI={df['ndvi_lag1'].isna().sum():,}")
    return set(df.columns), ids


def merge(output_dir: Path, output_path: Path, overwrite: bool) -> None:
    inputs = [(output_dir / name, start, end) for name, start, end in BATCHES]
    missing = [str(path) for path, _, _ in inputs if not path.is_file()]
    if missing:
        details = "\n".join("  - " + x for x in missing)
        commands = "\n".join(
            f"  python -u WP4/scripts/07_process_ndvi.py --input-dir WP4/data/raw/modis/{name.replace('veg_', '').replace('.parquet', '')} "
            f"--start {start} --end {end} --output WP4/outputs/{name}"
            for name, start, end in BATCHES
            if not (output_dir / name).is_file()
        )
        raise FileNotFoundError(
            "Required batch Parquet files are missing:\n" + details +
            "\n\nRegenerate missing batches from raw AppEEARS GeoTIFFs:\n" + commands
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}\nUse --overwrite to replace it after validation.")

    parts: list[pd.DataFrame] = []
    reference_columns = None
    reference_ids = None
    for path, start, end in inputs:
        frame = pd.read_parquet(path)
        reference_columns, reference_ids = verify_batch(
            frame, path.name, start, end, reference_columns, reference_ids
        )
        parts.append(frame)

    result = pd.concat(parts, ignore_index=True)
    require(len(result) == EXPECTED_CELLS * len(EXPECTED_MONTHS), "Combined row count mismatch")
    require(not result.duplicated(["cell_id", "year", "month"]).any(),
            "Duplicate combined target cell-month keys")
    source = month_period(result, "source_year", "source_month")
    require(set(source.unique()) == set(EXPECTED_MONTHS), "Combined month coverage mismatch")
    require(result.groupby(["source_year", "source_month"]).size().eq(EXPECTED_CELLS).all(),
            "Combined cells-per-month mismatch")

    result = result.sort_values(["year", "month", "cell_id"]).reset_index(drop=True)
    print(f"PASS combined: {len(result):,} rows; {len(EXPECTED_MONTHS)} source months; "
          f"{len(reference_ids):,} matching cells")
    print(f"INFO missing NDVI: {result['ndvi_lag1'].isna().sum():,}")
    print(f"INFO spatial_coverage_approx > 1: "
          f"{result['spatial_coverage_approx'].gt(1).sum():,} (diagnostic approximation)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}_", suffix=".parquet", dir=output_path.parent,
            delete=False
        ) as tmp:
            temp_path = Path(tmp.name)
        result.to_parquet(temp_path, index=False)
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    print(f"SAVED {output_path}")
    print(f"Shape: {result.shape}; target period: 2016-05 through 2026-08")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory containing the three batch Parquet inputs")
    parser.add_argument("--output", type=Path, default=None,
                        help="Combined output path (default: OUTPUT_DIR/veg_cell_month.parquet)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing final output after all checks pass")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_path = args.output.resolve() if args.output else output_dir / "veg_cell_month.parquet"
    try:
        merge(output_dir, output_path, args.overwrite)
    except (ValidationError, FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
