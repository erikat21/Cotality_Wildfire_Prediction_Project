"""
WP2 step 2: overlay the fire perimeters on the grid (`fire_cell` table).

One row per (fire, cell it touches), for every Spanish/Portuguese fire of at least 30 ha,
so the size cutoff S can be changed later without redoing the overlay.

Both layers are reprojected to EPSG:3035 (equal area) BEFORE any area is measured.

Inputs:
    outputs/cells.gpkg                      from 01_build_cells.py
    EFFIS burnt-area polygons               shapefile path given with --fires
        (shapes always come from here)
    WP1 fires_master.parquet                optional, --master: then WP1 decides which fires are kept and
                                            their country, area_ha and firedate (matched on fire_id = EFFIS id)
Outputs (outputs/):
    fire_cell.parquet   fire_id, cell_id, firedate, year, month, fire_area_ha,
                        burnt_ha_in_cell, cell_area_ha, burn_frac_cell, centroid_in_cell
    fire_cell.gpkg      same rows with the burnt piece's geometry (03_labels.py needs it
                        to merge fires that overlap in the same cell and month)
    fire_cell_check.csv per-fire area check: EFFIS AREA_HA vs polygon area vs sum over cells

Run:  python 02_fire_cell.py --fires ~/Desktop/Capstone/fires/modis.ba.poly.shp
      add --master ../WP1/data/processed/fires_master.parquet to use WP1's cleaned fire table
      (add --table ~/Desktop/Capstone/fires/effis_data_table_29thSept2026.csv if it says attributes are missing)
"""
import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
EQUAL_AREA = "EPSG:3035"
MIN_FIRE_HA = 30            # smallest S we will ever test
COUNTRIES = {"ES", "PT"}
GRID_BOUNDS = (-9.85, 35.85, 4.45, 44.05)  # lon/lat bounds of the whole grid, for a fast pre-filter


def apply_master(fires: gpd.GeoDataFrame, master_path: Path) -> gpd.GeoDataFrame:
    """Keep only WP1's fires and take country, size and start date from fires_master."""
    master = pd.read_parquet(master_path)
    master.columns = [c.lower() for c in master.columns]   # WP1 stores FIREDATE in capitals
    master = master[["fire_id", "country", "area_ha", "firedate"]].copy()
    master["fire_id"] = master.fire_id.astype(str)
    fires["ID"] = fires.ID.astype(str)
    big = master[master.area_ha >= MIN_FIRE_HA]
    no_shape = ~big.fire_id.isin(fires.ID)
    print(f"fires_master: {len(master):,} fires, {len(big):,} of at least {MIN_FIRE_HA} ha; "
          f"{no_shape.sum()} of those have no polygon in the shapefile box")
    fires = fires.drop(columns=["COUNTRY", "AREA_HA", "FIREDATE"], errors="ignore").merge(
        master.rename(columns={"fire_id": "ID", "country": "COUNTRY", "area_ha": "AREA_HA", "firedate": "FIREDATE"}),
        on="ID", how="inner")
    return fires


def load_fires(path: Path, table: Path | None = None, master: Path | None = None) -> gpd.GeoDataFrame:
    fires = gpd.read_file(path, bbox=GRID_BOUNDS)
    fires.columns = [c if c == "geometry" else c.upper() for c in fires.columns]
    print(f"read {len(fires):,} polygons inside the grid box; CRS = {fires.crs}")
    print(f"shapefile columns: {[c for c in fires.columns if c != 'geometry']}")

    if master is not None:
        fires = apply_master(fires, master)

    # If the shapefile lacks the attributes, take them from the EFFIS CSV table (joined on id)
    needed = {"COUNTRY", "AREA_HA", "FIREDATE"}
    if not needed <= set(fires.columns):
        if table is None:
            raise SystemExit(f"shapefile is missing {needed - set(fires.columns)}: pass the EFFIS CSV with --table")
        t = pd.read_csv(table, encoding="utf-8")
        t.columns = [c.upper() for c in t.columns]
        keep = ["ID"] + sorted(needed - set(fires.columns))
        fires["ID"] = fires.ID.astype(str)
        t["ID"] = t.ID.astype(str)
        fires = fires.merge(t[keep], on="ID", how="left")
        print(f"joined {sorted(needed)} from {table.name}; unmatched ids: {fires.COUNTRY.isna().sum()}")

    fires = fires[fires.COUNTRY.isin(COUNTRIES)].copy()
    fires["AREA_HA"] = pd.to_numeric(fires.AREA_HA, errors="coerce")
    fires = fires[fires.AREA_HA >= MIN_FIRE_HA].copy()
    fires["firedate"] = pd.to_datetime(fires.FIREDATE, errors="coerce", format="ISO8601")  # some dates carry milliseconds
    bad_dates = fires.firedate.isna().sum()
    if bad_dates:
        print(f"WARNING: {bad_dates} fires have an unreadable FIREDATE and are dropped")
        fires = fires[fires.firedate.notna()]

    fires = fires.rename(columns={"ID": "fire_id", "AREA_HA": "fire_area_ha"})
    if fires.fire_id.duplicated().any():
        raise SystemExit("fire ids are not unique: check the id column before going on")

    fires = fires.to_crs(EQUAL_AREA)
    fires["geometry"] = fires.geometry.make_valid()
    fires["poly_area_ha"] = fires.area / 10_000
    fires["year"] = fires.firedate.dt.year
    fires["month"] = fires.firedate.dt.month          # a fire belongs to the month it started
    print(f"kept {len(fires):,} ES/PT fires of at least {MIN_FIRE_HA} ha")
    return fires[["fire_id", "firedate", "year", "month", "COUNTRY", "fire_area_ha", "poly_area_ha", "geometry"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fires", required=True, type=Path, help="EFFIS burnt-area shapefile")
    ap.add_argument("--master", type=Path, help="WP1 fires_master.parquet (optional)")
    ap.add_argument("--table", type=Path, help="EFFIS CSV table, only needed if the shapefile lacks the attributes")
    args = ap.parse_args()

    cells = gpd.read_file(OUT / "cells.gpkg", layer="cells")[["cell_id", "cell_area_ha", "in_mask", "geometry"]]
    assert cells.crs.to_epsg() == 3035, "cells.gpkg should already be in EPSG:3035"
    fires = load_fires(args.fires, args.table, args.master)

    # The overlay: cut every fire into the pieces that fall in each cell
    pieces = gpd.overlay(fires, cells, how="intersection", keep_geom_type=True)
    pieces["burnt_ha_in_cell"] = pieces.area / 10_000
    pieces = pieces[pieces.burnt_ha_in_cell > 0].copy()
    pieces["burn_frac_cell"] = pieces.burnt_ha_in_cell / pieces.cell_area_ha

    # Which single cell holds each fire's centroid
    cent = gpd.GeoDataFrame(fires[["fire_id"]], geometry=fires.geometry.centroid, crs=EQUAL_AREA)
    cent = gpd.sjoin(cent, cells[["cell_id", "geometry"]], predicate="within")[["fire_id", "cell_id"]]
    pieces = pieces.merge(cent.assign(centroid_in_cell=1), on=["fire_id", "cell_id"], how="left")
    pieces["centroid_in_cell"] = pieces.centroid_in_cell.fillna(0).astype(int)

    cols = ["fire_id", "cell_id", "firedate", "year", "month", "fire_area_ha", "burnt_ha_in_cell",
            "cell_area_ha", "burn_frac_cell", "centroid_in_cell", "in_mask"]
    pieces = pieces[cols + ["geometry"]].sort_values(["fire_id", "cell_id"])
    pd.DataFrame(pieces.drop(columns="geometry")).to_parquet(OUT / "fire_cell.parquet", index=False)
    pieces.to_file(OUT / "fire_cell.gpkg", layer="fire_cell", driver="GPKG")

    # Check: the pieces of each fire should add back up to the whole polygon
    check = fires.set_index("fire_id")[["fire_area_ha", "poly_area_ha"]].join(
        pieces.groupby("fire_id").agg(sum_over_cells_ha=("burnt_ha_in_cell", "sum"),
                                      n_cells=("cell_id", "size"),
                                      n_cells_in_mask=("in_mask", "sum")))
    check["pieces_vs_polygon"] = check.sum_over_cells_ha / check.poly_area_ha
    check["polygon_vs_effis"] = check.poly_area_ha / check.fire_area_ha
    check.to_csv(OUT / "fire_cell_check.csv")

    print(f"fire_cell rows: {len(pieces):,} from {pieces.fire_id.nunique():,} fires")
    print(f"cells touched per fire: median {check.n_cells.median():.0f}, max {check.n_cells.max():.0f}")
    lost = check[(check.pieces_vs_polygon - 1).abs() > 0.001]
    print(f"fires whose pieces don't add up to the polygon (>0.1% off): {len(lost)}  (expected only near the grid edge)")
    print("polygon area / EFFIS AREA_HA: "
          f"median {check.polygon_vs_effis.median():.3f}, 5-95% {check.polygon_vs_effis.quantile(.05):.3f}-{check.polygon_vs_effis.quantile(.95):.3f}")
    print(f"fires with no cell in the mask: {(check.n_cells_in_mask == 0).sum()}")


if __name__ == "__main__":
    main()
