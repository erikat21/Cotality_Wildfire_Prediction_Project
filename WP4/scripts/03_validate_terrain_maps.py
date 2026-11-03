from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
terrain_path = ROOT / "outputs" / "terrain_cell.parquet"
cells_path = ROOT.parents[0] / "wp2" / "outputs" / "cells.gpkg"
out_dir = ROOT / "outputs" / "terrain_validation_maps"
out_dir.mkdir(parents=True, exist_ok=True)

# read data
terrain = pd.read_parquet(terrain_path)
cells = gpd.read_file(cells_path)

# keep in-mask cells only
cells = cells[cells["in_mask"] == 1].copy()

# join terrain features
g = cells.merge(terrain, on="cell_id", how="left")

print("Rows:", len(g))
print("Missing terrain rows after join:")
print(g[["elev_mean", "slope_mean", "slope_p90", "dem_coverage_frac_cell"]].isna().sum())

def save_map(gdf, column, title, filename):
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(column=column, ax=ax, legend=True)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200)
    plt.close(fig)

# maps
save_map(g, "elev_mean", "Terrain validation: mean elevation", "map_elev_mean.png")
save_map(g, "slope_mean", "Terrain validation: mean slope", "map_slope_mean.png")
save_map(g, "slope_p90", "Terrain validation: slope p90", "map_slope_p90.png")
save_map(g, "dem_coverage_frac_cell", "Terrain validation: DEM coverage fraction", "map_dem_coverage.png")

# optional: aspect maps
save_map(g, "aspect_south_share", "Terrain validation: south-facing share", "map_aspect_south_share.png")
save_map(g, "aspect_resultant", "Terrain validation: aspect resultant", "map_aspect_resultant.png")

print(f"Saved maps to: {out_dir}")