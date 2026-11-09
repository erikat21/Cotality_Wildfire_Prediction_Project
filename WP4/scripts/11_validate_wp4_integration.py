"""Validate WP4 outputs against the CURRENT WP2 Iberia land mask.

Examples (from repository root):
  python WP4/scripts/11_validate_wp4_integration.py --terrain-only --terrain WP4/outputs/terrain_cell_6555.parquet
  python WP4/scripts/11_validate_wp4_integration.py --terrain WP4/outputs/terrain_cell_6555.parquet --landcover WP4/outputs/landcover_cell_6555.parquet --veg WP4/outputs/veg_cell_month_6555.parquet
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "WP4" / "outputs"
MONTHS = pd.period_range("2016-04", "2026-07", freq="M")
TERRAIN_BASE = ["cell_id", "elev_mean", "elev_std", "slope_mean", "slope_p90", "aspect_south_share", "aspect_resultant", "n_aspect_pixels", "dem_coverage_frac_cell"]
LC_SHARES = ["artificial_share", "agriculture_share", "forest_share", "shrub_herbaceous_share", "bare_sparse_share", "wetland_share", "water_share"]
LC_BASE = ["cell_id", "epoch", *LC_SHARES, "valid_fraction", "n_valid_pixels", "dominant_class"]
NDVI_BASE = ["cell_id", "year", "month", "source_year", "source_month", "ndvi_lag1", "ndvi_std", "valid_fraction", "n_valid_pixels", "n_total_pixels", "n_source_valid_pixels"]
failures = []
warnings = []


def check(condition, message):
    ok = bool(condition)
    print(f"{'PASS' if ok else 'FAIL'}  {message}")
    if not ok:
        failures.append(message)


def warn(message):
    print(f"WARN  {message}")
    warnings.append(message)


def load(path, columns):
    path = Path(path)
    data = pd.read_parquet(path)
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    print(f"\n[{path.name}] {len(data):,} rows")
    return data


def unique(df, columns):
    return df[columns].notna().all().all() and not df.duplicated(columns).any()


def bounded(series, low, high, tol=1e-6):
    return series.dropna().between(low - tol, high + tol).all()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cells', type=Path, default=ROOT / 'wp2/outputs/cells.parquet')
    p.add_argument('--terrain', type=Path, default=OUT / 'terrain_cell.parquet')
    p.add_argument('--landcover', type=Path, default=OUT / 'landcover_cell.parquet')
    p.add_argument('--veg', type=Path, default=OUT / 'veg_cell_month.parquet')
    p.add_argument('--terrain-only', action='store_true', help='Validate terrain now, before the other datasets are reprocessed')
    args = p.parse_args()

    cells = load(args.cells, ['cell_id', 'in_mask', 'edge_type', 'land_frac'])
    target = cells.loc[cells.in_mask.eq(1), ['cell_id', 'edge_type', 'land_frac']].copy()
    expected_ids = set(target.cell_id)
    n = len(target)
    print(f'Current WP2 mask: {n:,} cells; expected NDVI rows: {n * len(MONTHS):,}')
    check(unique(target, ['cell_id']), 'WP2 cell_id unique')

    t = load(args.terrain, TERRAIN_BASE)
    check(len(t) == n, f'Terrain has {n:,} rows')
    check(unique(t, ['cell_id']), 'Terrain keys are unique and non-null')
    check(set(t.cell_id) == expected_ids, 'Terrain cell IDs match current WP2 mask')
    check(t[['elev_mean', 'elev_std', 'slope_mean', 'slope_p90']].notna().all().all(),
          'Elevation and slope metrics are non-null')
    check((t.elev_std.dropna() >= 0).all(), 'Elevation std nonnegative')
    check(bounded(t.slope_mean, 0, 90) and bounded(t.slope_p90, 0, 90), 'Slope metrics in [0,90]')
    check(bounded(t.dem_coverage_frac_cell, 0, 1, 1e-3), 'DEM coverage in [0,1]')
    check((t.n_aspect_pixels >= 0).all(), 'Aspect pixel count nonnegative')
    south_missing = t.aspect_south_share.isna()
    result_missing = t.aspect_resultant.isna()
    check((south_missing == result_missing).all(), 'Both aspect metrics have matching missingness')
    check((south_missing == t.n_aspect_pixels.eq(0)).all(),
          'Aspect is missing exactly when no directional pixels exist')
    check(bounded(t.aspect_south_share, 0, 1) and bounded(t.aspect_resultant, 0, 1),
          'Non-missing aspect metrics in [0,1]')
    if south_missing.any():
        diag = t.loc[south_missing, ['cell_id','n_aspect_pixels']].merge(target, on='cell_id', how='left')
        print('\nAspect missing diagnostics:')
        print(diag.to_string(index=False))
        if not diag.edge_type.isin(['coast', 'coast_border']).all():
            warn('Some no-aspect cells are not marked as coastal; inspect them')
    if args.terrain_only:
        print(f'\nTerrain-only validation: {len(failures)} failures, {len(warnings)} warnings')
        return 1 if failures else 0

    lc = load(args.landcover, LC_BASE)
    check(len(lc) == 2*n, f'Land cover has {2*n:,} rows')
    check(unique(lc, ['cell_id','epoch']), 'Land-cover keys unique/non-null')
    check(set(lc.cell_id) == expected_ids, 'Land-cover IDs match WP2')
    check(set(lc.epoch.dropna().astype(int)) == {2012, 2018}, 'Land-cover epochs exactly 2012 and 2018')
    check((lc.groupby('epoch').size().reindex([2012,2018], fill_value=0) == n).all(),
          f'Each land-cover epoch has {n:,} rows')
    check(all(bounded(lc[c], 0, 1) for c in [*LC_SHARES, 'valid_fraction']),
          'Land-cover shares/fractions in [0,1] where defined')
    valid_lc = lc.n_valid_pixels.gt(0)
    check(lc.loc[valid_lc, LC_SHARES].notna().all().all(), 'Classified land cover shares non-null')
    check(np.isclose(lc.loc[valid_lc, LC_SHARES].sum(axis=1), 1, atol=0.001).all(),
          'Classified land cover shares sum to 1')
    no_lc = int((~valid_lc).sum())
    if no_lc:
        warn(f'{no_lc} land-cover rows have no classified pixels; keep as missing and examine edge_type')

    v = load(args.veg, NDVI_BASE)
    check(len(v) == n*len(MONTHS), f'NDVI has {n*len(MONTHS):,} rows')
    check(unique(v, ['cell_id','year','month']), 'NDVI target keys unique/non-null')
    check(unique(v, ['cell_id','source_year','source_month']), 'NDVI source keys unique/non-null')
    check(set(v.cell_id) == expected_ids, 'NDVI IDs match WP2')
    check(v.month.between(1,12).all() and v.source_month.between(1,12).all(), 'Month values in [1,12]')
    src = pd.PeriodIndex.from_fields(year=v.source_year, month=v.source_month, freq='M')
    dst = pd.PeriodIndex.from_fields(year=v.year, month=v.month, freq='M')
    check(set(src.unique()) == set(MONTHS), 'All expected source months present')
    check((v.groupby(['source_year','source_month']).size() == n).all(),
          f'Each source month has {n:,} rows')
    check((src + 1 == dst).all(), 'Lag-1 convention: target month = source month + 1')
    check(bounded(v.ndvi_lag1, -1, 1), 'NDVI in [-1,1] where defined')
    check((v.ndvi_std.dropna() >= 0).all(), 'NDVI std nonnegative')
    check(bounded(v.valid_fraction, 0, 1), 'NDVI valid fraction in [0,1]')
    check(((v.n_valid_pixels <= v.n_source_valid_pixels) &
           (v.n_source_valid_pixels <= v.n_total_pixels) &
           (v.n_valid_pixels >= 0)).all(), 'NDVI pixel counts consistent')
    check((v.ndvi_lag1.isna() == v.n_valid_pixels.eq(0)).all(),
          'NDVI missing exactly when no valid pixels')
    print(f'INFO  Missing NDVI: {v.ndvi_lag1.isna().sum():,} / {len(v):,}')
    if v.n_total_pixels.eq(0).any():
        warn(f'{v.n_total_pixels.eq(0).sum():,} NDVI rows have zero source pixels: inspect MODIS AOI coverage')
    print(f'\nFull WP4 validation: {len(failures)} failures, {len(warnings)} warnings')
    return 1 if failures else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f'ERROR  {type(e).__name__}: {e}', file=sys.stderr)
        sys.exit(1)
