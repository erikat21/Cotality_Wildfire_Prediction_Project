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

# Mask rule: keep every cell that has ANY Spanish or Portuguese land in it (coast, islands,
# borders included), as the mentors asked on 2026-11-04. The vn0 brief only says "land mask";
# the earlier rule (>= 50% land and >= 50% of the land in ES/PT) was our own choice and dropped
# 330 coastal and border cells that can still burn.
# land_frac and iberia_share stay in the table so anyone can apply a stricter filter later.
IBERIA = {"ES", "PT"}
MOSTLY_SEA_BELOW = 0.5   # edge_type "coast": less than half of the cell is land



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
    # Country with the most land in the cell. Spain/Portugal win over France/Andorra/Morocco
    # whenever they have any land there, so every in-mask cell is labelled ES or PT.
    by_country["is_iberia"] = by_country.country.isin(IBERIA)
    top = (by_country.sort_values(["is_iberia", "area_ha"], ascending=False)
           .drop_duplicates("cell_id").set_index("cell_id"))

    # NUTS 2 region with the most area in the cell (ES/PT regions first, same as country)
    by_nuts = area_by_group(cells, nuts2, "nuts2")
    by_nuts["is_iberia"] = by_nuts.nuts2.str[:2].isin(IBERIA)
    top_nuts = (by_nuts.sort_values(["is_iberia", "area_ha"], ascending=False)
                .drop_duplicates("cell_id").set_index("cell_id")["nuts2"])

    cells = cells.set_index("cell_id")
    cells = cells.join(land).join(iberia)
    cells[["land_area_ha", "iberia_area_ha"]] = cells[["land_area_ha", "iberia_area_ha"]].fillna(0.0)
    cells["land_frac"] = (cells.land_area_ha / cells.cell_area_ha).clip(upper=1.0)
    cells["country"] = top["country"]
    cells["share_in_country"] = top["area_ha"] / cells.land_area_ha
    cells["iberia_share"] = cells.iberia_area_ha / cells.land_area_ha  # ES + PT together, share of the land
    cells["nuts2"] = top_nuts
    cells["in_mask"] = (cells.iberia_area_ha > 0).astype(int)
    # edge_type, for in-mask cells only (empty for cells outside the mask):
    #   interior     at least half land, all of it ES/PT
    #   coast        less than half land (mostly sea): ERA5-Land has few or no land points here,
    #                so weather may need the nearest land point
    #   border       some of the land is France, Andorra or Morocco (ES/PT share < 1)
    #   coast_border both
    coast = cells.land_frac < MOSTLY_SEA_BELOW
    border = cells.iberia_share < 0.9999
    cells["edge_type"] = np.select(
        [coast & border, coast, border], ["coast_border", "coast", "border"], default="interior")
    cells.loc[cells.in_mask == 0, "edge_type"] = None
    cells = cells.reset_index()

    cols = ["cell_id", "row", "col", "lon", "lat", "cell_area_ha", "land_area_ha", "land_frac",
            "iberia_area_ha", "country", "share_in_country", "iberia_share", "nuts2", "in_mask", "edge_type"]
    cells = cells[cols + ["geometry"]]
    for c in ["cell_area_ha", "land_area_ha", "iberia_area_ha"]:
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
    print("in-mask cells by edge_type:")
    print(m.edge_type.value_counts().to_string())
    print(f"mask ES/PT land area: {m.iberia_area_ha.sum() / 100:,.0f} km2")
    print("in-mask cells by NUTS 2:")
    print(m.groupby("nuts2").size().to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 7))
    cells4326 = cells.to_crs("EPSG:4326")
    land_out = cells4326[(cells4326.in_mask == 0) & (cells4326.land_frac > 0)]
    land_out.plot(ax=ax, color="lightgrey", linewidth=0)
    colours = {"interior": "tab:orange", "coast": "tab:blue", "border": "tab:purple", "coast_border": "tab:green"}
    from matplotlib.patches import Patch
    handles = [Patch(color="lightgrey", label="land in other countries (not in mask)")]
    for kind, colour in colours.items():
        part = cells4326[cells4326.edge_type == kind]
        part.plot(ax=ax, color=colour, edgecolor="white", linewidth=0.1)
        handles.append(Patch(color=colour, label=f"{kind}: {len(part):,}"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set_title(f"vn0 grid: {len(m):,} of {len(cells):,} cells touch Spain or Portugal")
    ax.set_xlabel("longitude (°)")
    ax.set_ylabel("latitude (°)")
    fig.tight_layout()
    fig.savefig(OUT / "cells_map.png", dpi=150)


if __name__ == "__main__":
    main()
