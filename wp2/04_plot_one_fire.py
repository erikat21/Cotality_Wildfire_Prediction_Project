"""
WP2 check: draw one fire cut into grid cells, to see that the overlay did the right thing.

Each piece of the fire is coloured by the cell it falls in, the grid lines are drawn on top,
and every touched cell is labelled with the share of that cell that burnt (burn_frac_cell).
The pieces should fit back together into the fire's outline with nothing missing.

Inputs:  outputs/fire_cell.gpkg  (from 02_fire_cell.py)
         outputs/cells.gpkg      (from 01_build_cells.py)
Output:  outputs/fire_<id>_cells.png

Run:  python 04_plot_one_fire.py              # the Rúa fire, id 278475
      python 04_plot_one_fire.py --fire 123456
"""
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", default="278475", help="EFFIS fire id (default: the Rúa fire)")
    args = ap.parse_args()

    pieces = gpd.read_file(OUT / "fire_cell.gpkg", layer="fire_cell",
                           where=f"fire_id = '{args.fire}'")
    if pieces.empty:
        raise SystemExit(f"fire {args.fire} is not in fire_cell.gpkg (wrong id, under 30 ha, or outside the grid)")
    cells = gpd.read_file(OUT / "cells.gpkg", layer="cells", bbox=tuple(pieces.total_bounds))
    cells = cells[cells.intersects(pieces.union_all().envelope.buffer(2_000))]

    # Plot in lon/lat so the grid squares look square and the axes read in degrees
    pieces, cells = pieces.to_crs("EPSG:4326"), cells.to_crs("EPSG:4326")
    fig, ax = plt.subplots(figsize=(9, 8))
    pieces.plot(ax=ax, column="cell_id", categorical=True, cmap="tab20", edgecolor="black", linewidth=0.3)
    cells.boundary.plot(ax=ax, color="grey", linewidth=0.6)
    for _, p in pieces.iterrows():
        cell = cells[cells.cell_id == p.cell_id].geometry.iloc[0].centroid
        ax.annotate(f"{p.cell_id}\n{p.burn_frac_cell:.0%}", (cell.x, cell.y), ha="center", va="center", fontsize=7)

    f = pieces.iloc[0]
    total = pieces.burnt_ha_in_cell.sum()
    ax.set_title(f"Fire {args.fire} ({f.firedate:%Y-%m-%d}): {total:,.0f} ha over {len(pieces)} cells\n"
                 f"EFFIS area {f.fire_area_ha:,.0f} ha; label in each cell = cell_id and share of the cell burnt",
                 fontsize=10)
    ax.set_xlabel("longitude (°)")
    ax.set_ylabel("latitude (°)")
    fig.tight_layout()
    out = OUT / f"fire_{args.fire}_cells.png"
    fig.savefig(out, dpi=150)
    print(pieces[["cell_id", "burnt_ha_in_cell", "burn_frac_cell", "centroid_in_cell", "in_mask"]]
          .sort_values("burnt_ha_in_cell", ascending=False).to_string(index=False))
    print(f"sum of pieces {total:,.1f} ha vs EFFIS AREA_HA {f.fire_area_ha:,.1f} ha -> saved {out.name}")


if __name__ == "__main__":
    main()
