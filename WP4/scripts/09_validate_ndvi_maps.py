"""WP4 MODIS NDVI spatial validation maps.

Run from repository root:
    python -u WP4/scripts/09_validate_ndvi_maps.py

Inputs:
    wp2/outputs/cells.gpkg
    WP4/outputs/veg_cell_month.parquet

Outputs:
    WP4/outputs/ndvi_validation_maps/*.png
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GRID_PATH = ROOT / "wp2" / "outputs" / "cells.gpkg"
NDVI_PATH = ROOT / "WP4" / "outputs" / "veg_cell_month.parquet"
OUT = ROOT / "WP4" / "outputs" / "ndvi_validation_maps"
OUT.mkdir(parents=True, exist_ok=True)


def plot_map(cells, values, title, filename, *, cmap="YlGn", vmin=0, vmax=0.85,
             cbar_label="NDVI", missing_color="#d9d9d9"):
    df = cells[["cell_id", "geometry"]].merge(
        values.rename("value"), left_on="cell_id", right_index=True,
        how="left", validate="one_to_one"
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    df.plot(column="value", ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
            linewidth=0, legend=True,
            legend_kwds={"label": cbar_label, "shrink": 0.75},
            missing_kwds={"color": missing_color, "label": "Missing"})
    ax.set_title(title, fontsize=14, pad=14)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    path = OUT / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def month_series(df, year, month):
    subset = df.loc[(df.source_year == year) & (df.source_month == month),
                    ["cell_id", "ndvi_lag1"]]
    if len(subset) != 6225 or subset.cell_id.duplicated().any():
        raise ValueError(f"Unexpected/incomplete source month {year}-{month:02d}")
    return subset.set_index("cell_id")["ndvi_lag1"]


def main():
    cells = gpd.read_file(GRID_PATH)
    if "in_mask" in cells.columns:
        cells = cells.loc[cells["in_mask"].astype(bool)].copy()
    if cells.crs is None:
        raise ValueError("WP2 grid CRS missing")
    cells = cells[["cell_id", "geometry"]].copy()
    if len(cells) != 6225 or cells.cell_id.duplicated().any():
        raise ValueError("Expected 6,225 unique WP2 in-mask cells")

    df = pd.read_parquet(NDVI_PATH)
    if df.duplicated(["cell_id", "year", "month"]).any():
        raise ValueError("Duplicate target cell-month keys")

    apr22 = month_series(df, 2022, 4)
    aug22 = month_series(df, 2022, 8)
    aug20 = month_series(df, 2020, 8)

    plot_map(cells, apr22, "April 2022 — MODIS NDVI (source month)",
             "01_ndvi_april_2022.png")
    plot_map(cells, aug22, "August 2022 — MODIS NDVI (source month)",
             "02_ndvi_august_2022.png")
    plot_map(cells, aug22 - apr22, "Seasonal NDVI change: August − April 2022",
             "03_ndvi_seasonal_change_2022.png", cmap="RdBu", vmin=-0.4,
             vmax=0.4, cbar_label="ΔNDVI")
    plot_map(cells, aug22 - aug20, "August NDVI change: 2022 − 2020",
             "04_ndvi_august_change_2020_2022.png", cmap="RdBu", vmin=-0.3,
             vmax=0.3, cbar_label="ΔNDVI")

    missing_counts = df["ndvi_lag1"].isna().groupby(df["cell_id"]).sum()
    max_missing = int(missing_counts.max())
    plot_map(cells, missing_counts.astype(float),
             "Missing NDVI observations per cell, April 2016–July 2026",
             "05_ndvi_missing_frequency.png", cmap="magma_r", vmin=0,
             vmax=max(1, max_missing), cbar_label="Missing source months",
             missing_color="#d9d9d9")

    print("\nValidation summary:")
    print("Grid cells:", len(cells))
    print("Source months:", df[["source_year", "source_month"]].drop_duplicates().shape[0])
    print("Missing NDVI records:", int(df.ndvi_lag1.isna().sum()))
    print("Cells with any missing NDVI:", int((missing_counts > 0).sum()))
    print("Maximum missing months in one cell:", max_missing)
    print("\nMean NDVI in selected months:")
    for label, values in [("Apr 2022", apr22), ("Aug 2022", aug22), ("Aug 2020", aug20)]:
        print(f"  {label}: {values.mean():.4f}")


if __name__ == "__main__":
    main()
