from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

CELLS = ROOT / "wp2/outputs/cells.gpkg"
LAND = ROOT / "WP4/outputs/landcover_cell.parquet"
OUT = ROOT / "WP4/outputs/landcover_validation_maps"
OUT.mkdir(parents=True, exist_ok=True)

cells = gpd.read_file(CELLS)
cells = cells[cells["in_mask"] == 1].copy()

land = pd.read_parquet(LAND)

groups = [
    "artificial",
    "agriculture",
    "forest",
    "shrub_herbaceous",
    "bare_sparse",
    "wetland",
    "water",
]

assert len(land) == 12450
assert not land.duplicated(["cell_id", "epoch"]).any()

# 1. Maps for each epoch
for group in groups:
    column = f"{group}_share"

    fig, axes = plt.subplots(
        1, 2, figsize=(15, 6),
        constrained_layout=True
    )

    for ax, year in zip(axes, [2012, 2018]):
        df = cells.merge(
            land[land["epoch"] == year],
            on="cell_id",
            how="left",
            validate="one_to_one"
        )

        df.plot(
            column=column,
            ax=ax,
            cmap="viridis",
            vmin=0,
            vmax=1,
            legend=True,
            legend_kwds={"shrink": 0.7}
        )

        ax.set_title(f"{group} share — {year}")
        ax.set_axis_off()

    fig.savefig(
        OUT / f"{group}_comparison.png",
        dpi=180
    )
    plt.close(fig)

# 2. Changes between epochs
a = land[land.epoch == 2012].set_index("cell_id")
b = land[land.epoch == 2018].set_index("cell_id")

for group in ["forest", "agriculture", "shrub_herbaceous"]:
    column = f"{group}_share"

    change = (b[column] - a[column]).rename("change")

    df = cells.merge(
        change.reset_index(),
        on="cell_id",
        how="left",
        validate="one_to_one"
    )

    limit = max(float(df["change"].abs().max()), 0.01)

    fig, ax = plt.subplots(figsize=(10, 8))

    df.plot(
        column="change",
        ax=ax,
        cmap="RdBu",
        vmin=-limit,
        vmax=limit,
        legend=True,
        legend_kwds={"shrink": 0.7}
    )

    ax.set_title(f"{group}: 2018 minus 2012")
    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(
        OUT / f"{group}_change.png",
        dpi=180
    )
    plt.close(fig)

# 3. Summary of changes
print("\nLand cover change summary:")

for group in groups:
    column = f"{group}_share"
    delta = b[column] - a[column]

    print(
        f"{group:20s} "
        f"mean={delta.mean():+.5f} "
        f"changed={(delta.abs() > 1e-6).sum():,} cells "
        f"max_abs={delta.abs().max():.5f}"
    )

print("\nSaved maps to:", OUT)
