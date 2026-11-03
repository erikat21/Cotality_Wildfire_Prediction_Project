"""WP4 final integration validation (read-only).

Run from repository root:
    python -u WP4/scripts/10_validate_wp4_integration.py

Checks alignment of terrain, CORINE land cover, and lagged MODIS NDVI.
Does not assign land-cover epochs to modeling months or generate a merged table.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "WP4" / "outputs"
EXPECTED_CELLS = 6225
SOURCE_START = pd.Period("2016-04", freq="M")
SOURCE_END = pd.Period("2026-07", freq="M")
EXPECTED_PERIODS = pd.period_range(SOURCE_START, SOURCE_END, freq="M")
REQUIRED_TERRAIN = ["cell_id", "elev_mean", "elev_std", "slope_mean", "slope_p90", "aspect_south_share", "aspect_resultant", "dem_coverage_frac_cell"]
LC_SHARES = ["artificial_share", "agriculture_share", "forest_share", "shrub_herbaceous_share", "bare_sparse_share", "wetland_share", "water_share"]
REQUIRED_LC = ["cell_id", "epoch", *LC_SHARES, "valid_fraction", "dominant_class"]
REQUIRED_NDVI = ["cell_id", "year", "month", "source_year", "source_month", "ndvi_lag1", "ndvi_std", "valid_fraction", "n_valid_pixels", "n_total_pixels", "n_source_valid_pixels"]
problems = []
warnings = []


def check(ok, message):
    print(f"{'PASS' if ok else 'FAIL'}  {message}")
    if not ok:
        problems.append(message)


def warn(message):
    warnings.append(message)
    print("WARN  " + message)


def load(name, required):
    path = OUT / name
    if not path.is_file():
        raise FileNotFoundError(f"Required output not found: {path}")
    df = pd.read_parquet(path)
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name}: missing columns: {missing}")
    print(f"\n[{name}] {len(df):,} rows; {len(df.columns)} columns")
    return df


def is_unique(df, keys):
    return not df.duplicated(keys).any() and df[keys].notna().all().all()


def bounded(series, lo, hi, tol=1e-6):
    return bool(series.dropna().between(lo-tol, hi+tol).all())


def main():
    print("WP4 FINAL INTEGRATION VALIDATION")
    print("Repository:", REPO)
    terrain = load("terrain_cell.parquet", REQUIRED_TERRAIN)
    lc = load("landcover_cell.parquet", REQUIRED_LC)
    veg = load("veg_cell_month.parquet", REQUIRED_NDVI)

    print("\n-- Terrain --")
    check(len(terrain) == EXPECTED_CELLS, "Terrain has 6,225 rows")
    check(is_unique(terrain, ["cell_id"]), "Terrain cell_id unique/non-null")
    check(terrain[REQUIRED_TERRAIN].notna().all().all(), "Required terrain features non-null")
    check(bool((terrain["elev_std"] >= 0).all()), "Elevation std is nonnegative")
    check(bounded(terrain["slope_mean"], 0, 90) and bounded(terrain["slope_p90"], 0, 90), "Slope values in [0, 90]")
    check(bounded(terrain["aspect_south_share"], 0, 1) and bounded(terrain["aspect_resultant"], 0, 1), "Aspect features in [0, 1]")
    check(bounded(terrain["dem_coverage_frac_cell"], 0, 1, 1e-3), "DEM coverage within [0, 1] (tolerance 0.001)")

    print("\n-- CORINE land cover --")
    check(len(lc) == EXPECTED_CELLS * 2, "Land cover has 12,450 rows")
    check(is_unique(lc, ["cell_id", "epoch"]), "Land-cover cell_id + epoch unique/non-null")
    check(set(lc["epoch"].dropna().astype(int)) == {2012, 2018}, "CORINE epochs are exactly 2012 and 2018")
    check(all((lc.groupby("epoch").size() == EXPECTED_CELLS).reindex([2012, 2018], fill_value=0)), "Each CORINE epoch has 6,225 cells")
    check(all(bounded(lc[c], 0, 1) for c in LC_SHARES + ["valid_fraction"]), "Land-cover shares and valid_fraction in [0, 1]")
    check(bool((lc[LC_SHARES].notna().all(axis=1)).all()), "Land-cover shares non-null")
    share_sum = lc[LC_SHARES].sum(axis=1)
    check(bool(np.isclose(share_sum, 1, atol=0.001).all()), "Land-cover shares sum to 1 (±0.001)")

    print("\n-- MODIS NDVI --")
    nmonths = len(EXPECTED_PERIODS)
    check(nmonths == 124, "Expected source period contains 124 months")
    check(len(veg) == EXPECTED_CELLS * nmonths, "NDVI has 771,900 rows")
    check(is_unique(veg, ["cell_id", "year", "month"]), "NDVI target cell-month keys unique/non-null")
    check(is_unique(veg, ["cell_id", "source_year", "source_month"]), "NDVI source cell-month keys unique/non-null")
    check(veg[["year", "month", "source_year", "source_month"]].notna().all().all(), "NDVI date fields non-null")
    check(veg["month"].between(1, 12).all() and veg["source_month"].between(1, 12).all(), "Month integers in [1, 12]")
    source = pd.PeriodIndex.from_fields(year=veg["source_year"], month=veg["source_month"], freq="M")
    target = pd.PeriodIndex.from_fields(year=veg["year"], month=veg["month"], freq="M")
    counts = veg.groupby(["source_year", "source_month"]).size()
    check(set(source.unique()) == set(EXPECTED_PERIODS), "NDVI source months complete, Apr 2016–Jul 2026")
    check(len(counts) == 124 and (counts == EXPECTED_CELLS).all(), "Each NDVI source month has 6,225 rows")
    check(bool((source + 1 == target).all()), "Target dates are exactly source + 1 month")
    check(bounded(veg["ndvi_lag1"], -1, 1), "NDVI means lie in [-1, 1]")
    check(bool((veg["ndvi_std"].dropna() >= 0).all()), "NDVI std nonnegative")
    check(bounded(veg["valid_fraction"], 0, 1), "NDVI valid_fraction in [0, 1]")
    check(bool((veg["n_valid_pixels"] <= veg["n_source_valid_pixels"]).all() and (veg["n_source_valid_pixels"] <= veg["n_total_pixels"]).all() and (veg["n_valid_pixels"] >= 0).all()), "Pixel counts obey valid ≤ source-valid ≤ total")
    check(bool((veg["ndvi_lag1"].isna() == (veg["n_valid_pixels"] == 0)).all()), "Missing NDVI iff zero valid pixels")
    missing = int(veg["ndvi_lag1"].isna().sum())
    print(f"INFO  Missing NDVI: {missing:,} / {len(veg):,} ({missing / len(veg):.4%})")
    if missing != 60:
        warn(f"NDVI missing count differs from previously observed 60: now {missing}")
    if "spatial_coverage_approx" in veg.columns:
        high = int((veg["spatial_coverage_approx"] > 1).sum())
        print(f"INFO  spatial_coverage_approx > 1: {high:,} rows (pixel-centre approximation, not true >100% coverage)")

    print("\n-- Cross-dataset cell_id alignment --")
    t_ids = set(terrain["cell_id"].dropna())
    lc_ids = set(lc["cell_id"].dropna())
    v_ids = set(veg["cell_id"].dropna())
    check(len(t_ids) == EXPECTED_CELLS and t_ids == lc_ids == v_ids, "Exactly the same 6,225 cell_ids across all three tables")
    for year in (2012, 2018):
        check(set(lc.loc[lc["epoch"] == year, "cell_id"]) == t_ids, f"CORINE {year} covers all Terrain cell_ids")

    print("\n-- Summary --")
    print(f"Passed with {len(problems)} failing check(s) and {len(warnings)} warning(s).")
    print("Final outputs remain unchanged; no temporal mapping of CORINE epochs has been assumed.")
    if problems:
        print("Failures:")
        for p in problems:
            print("  -", p)
        return 1
    print("WP4 tables are structurally compatible for downstream integration.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR  {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
