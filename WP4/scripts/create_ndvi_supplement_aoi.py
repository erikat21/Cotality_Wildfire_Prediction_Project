"""Create a small supplemental NASA AppEEARS Area Request AOI.

Run from the repository root:
    python WP4/scripts/create_ndvi_supplement_aoi.py

Requires: geopandas, shapely.
"""
from pathlib import Path
import argparse
import json

import geopandas as gpd

TARGET_IDS = (11627, 11628)
ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cells', type=Path, default=ROOT / 'wp2/outputs/cells.gpkg')
    p.add_argument('--out', type=Path, default=ROOT / 'WP4/data/reference/ndvi_supplement_aoi.geojson')
    p.add_argument('--buffer-m', type=float, default=1500.0)
    args = p.parse_args()
    if args.buffer_m < 0:
        p.error('--buffer-m must be nonnegative')
    if not args.cells.is_file():
        p.error(f'Missing WP2 cells file: {args.cells}')

    cells = gpd.read_file(args.cells)
    required = {'cell_id', 'in_mask', 'geometry'}
    if not required.issubset(cells.columns):
        p.error(f'Missing columns: {sorted(required - set(cells.columns))}')
    if cells.crs is None:
        p.error('Input grid has no CRS')

    target = cells[cells['cell_id'].isin(TARGET_IDS)].copy()
    if set(target['cell_id'].astype(int)) != set(TARGET_IDS):
        p.error(f'Expected exactly cell IDs {TARGET_IDS}, got {sorted(target.cell_id.tolist())}')
    if not target['in_mask'].eq(1).all():
        p.error('One or more requested cells are not in the new Iberian mask')
    if not target.geometry.is_valid.all():
        p.error('Input cell geometry is invalid')

    # Buffer in an equal-area metric projection, never in longitude/latitude degrees.
    metric = target.to_crs('EPSG:3035')
    polygon = metric.geometry.union_all().buffer(args.buffer_m)
    aoi = gpd.GeoDataFrame(
        {'name': ['iberia_ndvi_supplement_11627_11628']},
        geometry=[polygon], crs='EPSG:3035'
    ).to_crs('EPSG:4326')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Standard GeoJSON WGS84 FeatureCollection (supported by AppEEARS Area Requests).
    aoi.to_file(args.out, driver='GeoJSON')

    bounds = aoi.total_bounds
    print('Saved supplemental AOI:', args.out)
    print('Included WP2 cells:', sorted(target.cell_id.astype(int).tolist()))
    print('Buffer:', f'{args.buffer_m:,.0f} m')
    print('EPSG:4326 bounds [west, south, east, north]:', [round(float(v), 6) for v in bounds])
    print('AOI area:', f'{polygon.area / 1_000_000:.2f} km^2')
    print('GeoJSON features:', len(aoi))


if __name__ == '__main__':
    main()
