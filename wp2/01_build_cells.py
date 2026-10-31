"""
WP2 step 1: build the 0.1 degree grid (`cells` table) and the Iberia land mask.

The grid lines up exactly with the ERA5-Land download box used by WP3:
    [N 44.0, W -9.8, S 35.9, E 4.4]  ->  82 latitudes x 143 longitudes = 11,726 points.
Each ERA5-Land value belongs to the point at the CENTRE of a cell, so every cell
runs from its point -0.05 to +0.05 degrees in latitude and longitude.

cell_id rule (shared with WP3/WP4/WP5, never change it):
    row = round((44.0 - lat) / 0.1)    # row 0 is the northernmost latitude (44.0)
    col = round((lon - (-9.8)) / 0.1)  # col 0 is the westernmost longitude (-9.8)
    cell_id = row * 143 + col
Row 0 at the north matches the order ERA5 files store latitude in (descending).

Areas are always measured in EPSG:3035 (ETRS89 / LAEA Europe), an equal-area
projection, never in degrees.

Inputs  (data/boundaries/, see README):
    cntrg.json     country polygons incl. Morocco/France/Andorra (coastline = land)
    nutsrg_2.json  NUTS 2 regions (2024)
Outputs (outputs/):
    cells.parquet  one row per grid cell, no geometry (the table other WPs join on)
    cells.gpkg     same table with the cell squares, for QGIS / plotting
    cells_map.png  quick look at the mask

Run:  python 01_build_cells.py
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

HERE = Path(__file__).resolve().parent
BOUNDARIES = HERE / "data" / "boundaries"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

# Grid definition (must match the ERA5-Land download box)
NORTH, WEST, SOUTH, EAST = 44.0, -9.8, 35.9, 4.4
STEP = 0.1
N_ROWS = int(round((NORTH - SOUTH) / STEP)) + 1  # 82
N_COLS = int(round((EAST - WEST) / STEP)) + 1    # 143
EQUAL_AREA = "EPSG:3035"

# Mask rule from the vn0 brief: keep a cell if >= 50% land and >= 50% of the land is in Spain/Portugal
MIN_LAND_FRAC = 0.5
MIN_IBERIA_SHARE = 0.5
IBERIA = {"ES", "PT"}


def make_grid() -> gpd.GeoDataFrame:
    rows, cols = np.meshgrid(np.arange(N_ROWS), np.arange(N_COLS), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    lat = np.round(NORTH - rows * STEP, 1)
    lon = np.round(WEST + cols * STEP, 1)
    h = STEP / 2
    geoms = [box(x - h, y - h, x + h, y + h) for x, y in zip(lon, lat)]
    grid = gpd.GeoDataFrame(
        {"cell_id": rows * N_COLS + cols, "row": rows, "col": cols, "lon": lon, "lat": lat},
        geometry=geoms, crs="EPSG:4326",
    )
    # Add points along the edges first so the squares keep their true shape after reprojecting
    grid["geometry"] = grid.geometry.segmentize(0.01)
    return grid.to_crs(EQUAL_AREA)


def area_by_group(cells: gpd.GeoDataFrame, polys: gpd.GeoDataFrame, key: str) -> pd.DataFrame:
    """Area (ha) of each polygon group inside each cell: columns cell_id, key, area_ha."""
    pieces = gpd.overlay(cells[["cell_id", "geometry"]], polys[[key, "geometry"]], how="intersection")
    pieces["area_ha"] = pieces.area / 10_000
    return pieces.groupby(["cell_id", key], as_index=False)["area_ha"].sum()


def main():
    cells = make_grid()
    cells["cell_area_ha"] = cells.area / 10_000

    countries = gpd.read_file(BOUNDARIES / "cntrg.json").to_crs(EQUAL_AREA)
    nuts2 = gpd.read_file(BOUNDARIES / "nutsrg_2.json").to_crs(EQUAL_AREA)
    # Only countries that can touch the box, to keep the overlay fast
    countries = countries[countries.intersects(cells.union_all().envelope)].rename(columns={"id": "country"})
    nuts2 = nuts2[nuts2.intersects(cells.union_all().envelope)].rename(columns={"id": "nuts2"})

    # Land area per cell and country
    by_country = area_by_group(cells, countries, "country")
    land = by_country.groupby("cell_id")["area_ha"].sum().rename("land_area_ha")
    iberia = by_country[by_country.country.isin(IBERIA)].groupby("cell_id")["area_ha"].sum().rename("iberia_area_ha")
    top = by_country.sort_values("area_ha", ascending=False).drop_duplicates("cell_id").set_index("cell_id")

    # NUTS 2 region with the most area in the cell
    by_nuts = area_by_group(cells, nuts2, "nuts2")
    top_nuts = by_nuts.sort_values("area_ha", ascending=False).drop_duplicates("cell_id").set_index("cell_id")["nuts2"]

    cells = cells.set_index("cell_id")
    cells = cells.join(land).join(iberia)
    cells[["land_area_ha", "iberia_area_ha"]] = cells[["land_area_ha", "iberia_area_ha"]].fillna(0.0)
    cells["land_frac"] = (cells.land_area_ha / cells.cell_area_ha).clip(upper=1.0)
    cells["country"] = top["country"]                       # country with the most land in the cell
    cells["share_in_country"] = top["area_ha"] / cells.land_area_ha
    cells["iberia_share"] = cells.iberia_area_ha / cells.land_area_ha  # ES + PT together, share of the land
    cells["nuts2"] = top_nuts
    cells["in_mask"] = (
        (cells.land_frac >= MIN_LAND_FRAC) & (cells.iberia_share >= MIN_IBERIA_SHARE)
    ).astype(int)
    cells = cells.reset_index()

    cols = ["cell_id", "row", "col", "lon", "lat", "cell_area_ha", "land_area_ha", "land_frac",
            "country", "share_in_country", "iberia_share", "nuts2", "in_mask"]
    cells = cells[cols + ["geometry"]]
    for c in ["cell_area_ha", "land_area_ha"]:
        cells[c] = cells[c].round(1)
    for c in ["land_frac", "share_in_country", "iberia_share"]:
        cells[c] = cells[c].round(4)

    pd.DataFrame(cells.drop(columns="geometry")).to_parquet(OUT / "cells.parquet", index=False)
    cells.to_file(OUT / "cells.gpkg", layer="cells", driver="GPKG")

    # Checks printed for the README / meeting
    m = cells[cells.in_mask == 1]
    print(f"grid: {N_ROWS} x {N_COLS} = {len(cells):,} cells")
    print(f"in mask: {len(m):,} cells  (ES {sum(m.country == 'ES'):,}, PT {sum(m.country == 'PT'):,}, other {sum(~m.country.isin(IBERIA)):,})")
    print(f"cell area (ha): min {cells.cell_area_ha.min():,.0f}, max {cells.cell_area_ha.max():,.0f} (cells shrink going north)")
    print(f"mask land area: {m.land_area_ha.sum() / 100:,.0f} km2")
    print("in-mask cells by NUTS 2:")
    print(m.groupby("nuts2").size().to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7))
    cells4326 = cells.to_crs("EPSG:4326")
    land_out = cells4326[(cells4326.in_mask == 0) & (cells4326.land_frac > 0)]
    land_out.plot(ax=ax, color="lightgrey", linewidth=0)
    cells4326[cells4326.in_mask == 1].plot(ax=ax, color="tab:orange", edgecolor="white", linewidth=0.1)
    ax.set_title(f"vn0 grid: {len(m):,} of {len(cells):,} cells in the Iberia mask (orange); grey = other land")
    ax.set_xlabel("longitude (°)")
    ax.set_ylabel("latitude (°)")
    fig.tight_layout()
    fig.savefig(OUT / "cells_map.png", dpi=150)


if __name__ == "__main__":
    main()
