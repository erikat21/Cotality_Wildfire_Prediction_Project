"""
WP4 — Prepare Full Iberia MODIS NDVI AOI

Creates two GeoJSON files for NASA AppEEARS:

1. ndvi_iberia_aoi.geojson
   Union of all WP2 in-mask ERA5-Land cells,
   with 1 km buffer and simplified boundary.

2. ndvi_iberia_bbox.geojson
   Simple bounding rectangle covering all cells.

Input:
    wp2/outputs/cells.gpkg

Output:
    WP4/data/reference/
"""

from pathlib import Path
import json

import geopandas as gpd
from shapely.geometry import box, mapping


ROOT = Path(__file__).resolve().parents[2]

CELLS_PATH = ROOT / "wp2" / "outputs" / "cells.gpkg"

OUT_DIR = ROOT / "WP4" / "data" / "reference"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AOI_PATH = OUT_DIR / "ndvi_iberia_aoi.geojson"
BBOX_PATH = OUT_DIR / "ndvi_iberia_bbox.geojson"

# Processing parameters
TARGET_CRS = "EPSG:4326"
BUFFER_METERS = 1000
SIMPLIFY_METERS = 200


def save_geojson(geom, name, path):
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": name},
                "geometry": mapping(geom),
            }
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(feature_collection, f, indent=2)

    print(f"Saved: {path}")


def main():

    print("Loading WP2 grid...")

    cells = gpd.read_file(CELLS_PATH)

    cells = cells[
        cells["in_mask"] == 1
    ].copy()

    print(f"In-mask cells: {len(cells):,}")

    if len(cells) != 6225:
        print("Warning: expected 6,225 cells.")

    # WP2 grid uses EPSG:3035 (meters).
    cells = cells.to_crs("EPSG:3035")

    print("Merging cell polygons...")

    # Union all grid geometries.
    merged = cells.geometry.union_all()

    print("Applying 1 km buffer...")
    buffered = merged.buffer(BUFFER_METERS)

    print("Simplifying geometry...")
    simplified = buffered.simplify(
        SIMPLIFY_METERS,
        preserve_topology=True
    )

    # Confirm simplified AOI still contains all original
    # in-mask grid polygons.
    if not simplified.covers(merged):
        print(
            "Simplification clipped part of the grid; "
            "using buffered AOI without simplification."
        )
        simplified = buffered

    # Convert to geographic coordinates for AppEEARS.
    aoi_geo = gpd.GeoSeries(
        [simplified],
        crs="EPSG:3035"
    ).to_crs(TARGET_CRS).iloc[0]

    save_geojson(
        aoi_geo,
        "Full_Iberia_MODIS_NDVI",
        AOI_PATH
    )

    # Simple rectangle covering the buffered grid.
    minx, miny, maxx, maxy = aoi_geo.bounds
    bbox_geo = box(minx, miny, maxx, maxy)

    save_geojson(
        bbox_geo,
        "Full_Iberia_MODIS_NDVI_BBOX",
        BBOX_PATH
    )

    print("\nAOI bounds (EPSG:4326):")
    print(f"West:  {minx:.5f}")
    print(f"South: {miny:.5f}")
    print(f"East:  {maxx:.5f}")
    print(f"North: {maxy:.5f}")

    # Verify reprojected AOI covers original cells.
    check = gpd.GeoSeries(
        [aoi_geo],
        crs="EPSG:4326"
    ).to_crs("EPSG:3035").iloc[0]

    print(
        "\nCovers all original cells:",
        check.buffer(0.1).covers(merged)
    )

    print("\nAOI preparation complete.")


if __name__ == "__main__":
    main()
