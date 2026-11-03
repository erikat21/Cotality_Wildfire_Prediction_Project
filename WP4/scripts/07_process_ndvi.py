"""WP4 MOD13A3.061 AppEEARS GeoTIFF -> lagged ERA5 cell-month panel.

Example pilot (only the two source months):
 python WP4/scripts/07_process_ndvi.py --input-dir WP4/data/raw/modis/pilot_2025 --start 2025-04 --end 2025-05 --output WP4/outputs/veg_pilot_fullschema.parquet

Full run (all source months Apr 2016 through Jul 2026):
 python WP4/scripts/07_process_ndvi.py --input-dir WP4/data/raw/modis --output WP4/outputs/veg_cell_month.parquet

Requires: numpy pandas geopandas rasterio pyarrow.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import re
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "WP4/data/raw/modis"
DEFAULT_CELLS = ROOT / "wp2/outputs/cells.gpkg"
DEFAULT_OUTPUT = ROOT / "WP4/outputs/veg_cell_month.parquet"
DATE_RE = re.compile(r"doy(\d{7})", re.IGNORECASE)


def parse_month(s: str) -> pd.Period:
    try:
        return pd.Period(s, freq="M")
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Invalid month YYYY-MM: {s}") from exc


def file_month(path: Path) -> pd.Period | None:
    m = DATE_RE.search(path.name)
    if m is None:
        return None
    d = datetime.strptime(m.group(1), "%Y%j")
    return pd.Period(d, freq="M")


def collect_pairs(input_dir: Path, start: pd.Period, end: pd.Period) -> dict[pd.Period, dict[str, Path]]:
    pairs: dict[pd.Period, dict[str, Path]] = {}
    for p in input_dir.rglob("*.tif"):
        # When scanning the full modis folder, exclude separately-downloaded Pilot files.
        # Explicit --input-dir .../pilot_2025 still works for standalone pilot tests.
        if input_dir.name.lower() != "pilot_2025" and any(part.lower().startswith("pilot") for part in p.relative_to(input_dir).parts[:-1]):
            continue
        name = p.name.lower()
        if "mod13a3" not in name or "vi_quality" in name:
            continue
        kind = "quality" if "pixel_reliability" in name else ("ndvi" if "monthly_ndvi" in name else None)
        if kind is None:
            continue
        month = file_month(p)
        if month is None or not start <= month <= end:
            continue
        target = pairs.setdefault(month, {})
        if kind in target:
            raise ValueError(f"Duplicate {kind} for {month}:\n  {target[kind]}\n  {p}\n"
                             "Keep pilot files outside the production input directory or specify --input-dir.")
        target[kind] = p

    expected = pd.period_range(start=start, end=end, freq="M")
    problems = [(str(m), sorted(set(("ndvi", "quality")) - set(pairs.get(m, {})))) for m in expected
                if set(pairs.get(m, {})) != {"ndvi", "quality"}]
    if problems:
        print(f"Missing or incomplete source months: {len(problems)} / {len(expected)}")
        for m, needed in problems[:16]:
            print(f"  {m}: missing {', '.join(needed)}")
        if len(problems) > 16:
            print(f"  ... and {len(problems)-16} more")
        raise ValueError("Missing MODIS NDVI / reliability files; output was not written.")
    print(f"Paired {len(expected)} complete months, from {start} to {end}")
    return {m: pairs[m] for m in expected}


def load_cells(path: Path) -> gpd.GeoDataFrame:
    cells = gpd.read_file(path)
    for c in ("cell_id", "in_mask", "geometry"):
        if c not in cells:
            raise ValueError(f"Missing WP2 column: {c}")
    cells = cells.loc[cells.in_mask.eq(1), ["cell_id", "geometry"]].copy()
    cells = cells.sort_values("cell_id").reset_index(drop=True)
    if cells.cell_id.duplicated().any():
        raise ValueError("Duplicate WP2 cell_id")
    if cells.crs is None:
        raise ValueError("WP2 grid has no CRS")
    print(f"Loaded {len(cells)} WP2 in-mask cells")
    return cells


def process_month(month: pd.Period, pair: dict[str, Path], cells: gpd.GeoDataFrame) -> pd.DataFrame:
    with rasterio.open(pair["ndvi"]) as nd, rasterio.open(pair["quality"]) as qa:
        if nd.crs is None or qa.crs is None:
            raise ValueError(f"Missing source CRS for {month}")
        if nd.count != 1 or qa.count != 1 or nd.shape != qa.shape or nd.transform != qa.transform or nd.crs != qa.crs:
            raise ValueError(f"NDVI and reliability do not align in {month}")
        sc = float(nd.scales[0])
        off = float(nd.offsets[0])
        if not (np.isfinite(sc) and sc > 0 and np.isfinite(off)):
            raise ValueError(f"Invalid NDVI scale/offset in {month}")
        # MOD13A3.061 raw int16 NDVI usually uses scale=0.0001. Do not double-scale already-scaled data.
        project = cells.to_crs(nd.crs)
        raster_cell = rasterize(
            ((geom, i + 1) for i, geom in enumerate(project.geometry)),
            out_shape=nd.shape, transform=nd.transform,
            fill=0, dtype="int32", all_touched=False
        )
        raw = nd.read(1, masked=True)
        quality = qa.read(1, masked=True)
        raw_mask = np.ma.getmaskarray(raw)
        quality_mask = np.ma.getmaskarray(quality)
        scaled = raw.data.astype(np.float32) * sc + off
        inside = raster_cell > 0
        source_valid = ~raw_mask & np.isfinite(scaled) & (scaled >= -1) & (scaled <= 1)
        reliable = ~quality_mask & np.isin(quality.data, (0, 1))
        valid = inside & source_valid & reliable
        size = len(cells) + 1
        total = np.bincount(raster_cell[inside], minlength=size)
        source_count = np.bincount(raster_cell[inside & source_valid], minlength=size)
        valid_count = np.bincount(raster_cell[valid], minlength=size)
        sums = np.bincount(raster_cell[valid], weights=scaled[valid], minlength=size)
        sums_sq = np.bincount(raster_cell[valid], weights=scaled[valid] ** 2, minlength=size)
        ix = np.arange(1, size)
        n = valid_count[ix]
        with np.errstate(divide="ignore", invalid="ignore"):
            means = np.where(n > 0, sums[ix] / n, np.nan)
            variance = np.where(n > 0, sums_sq[ix] / n - means ** 2, np.nan)
            stds = np.sqrt(np.maximum(variance, 0))
            valid_fraction = np.where(total[ix] > 0, n / total[ix], np.nan)
            quality_fraction = np.where(source_count[ix] > 0, n / source_count[ix], np.nan)

        # Spatial coverage denominator is expected ~ pixels per whole cell;
        # native MODIS Sinusoidal is equal-area (pixel_area in m^2).
        pixel_area = abs(nd.transform.a * nd.transform.e - nd.transform.b * nd.transform.d)
        cell_areas = project.geometry.area.to_numpy()
        coverage_approx = total[ix] * pixel_area / cell_areas
        if np.any(coverage_approx > 1.20):
            raise ValueError(f"Implausibly large pixel spatial coverage in {month}")
        target = month + 1
        out = pd.DataFrame({
            "cell_id": cells.cell_id.to_numpy(dtype="int64"),
            "year": target.year, "month": target.month,
            "source_year": month.year, "source_month": month.month,
            "ndvi_lag1": means, "ndvi_std": stds,
            "valid_fraction": valid_fraction,
            "quality_pass_fraction": quality_fraction,
            "spatial_coverage_approx": coverage_approx,
            "n_valid_pixels": n, "n_total_pixels": total[ix],
            "n_source_valid_pixels": source_count[ix]
        })
        print(f"{month} -> {target}: valid cells={int((n>0).sum())}/{len(cells)}, "
              f"mean NDVI={np.nanmean(means):.4f}, "
              f"zero-coverage cells={int((total[ix]==0).sum())}, "
              f"scale={sc}")
        return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--start", type=parse_month, default=parse_month("2016-04"), help="First source NDVI month")
    p.add_argument("--end", type=parse_month, default=parse_month("2026-07"), help="Last source NDVI month")
    p.add_argument("--check-only", action="store_true", help="Check that every source month has both GeoTIFF layers")
    args = p.parse_args()
    if args.end < args.start:
        p.error("--end must be >= --start")
    if not args.input_dir.is_dir():
        p.error(f"Input directory not found: {args.input_dir}")
    pairs = collect_pairs(args.input_dir, args.start, args.end)
    if args.check_only:
        print("All required NDVI and pixel_reliability files found.")
        return
    cells = load_cells(args.cells)
    chunks = [process_month(m, pair, cells) for m, pair in pairs.items()]
    result = pd.concat(chunks, ignore_index=True)
    expected_n = len(cells) * len(pairs)
    if len(result) != expected_n or result.duplicated(["cell_id", "year", "month"]).any():
        raise ValueError("Row count or unique cell-month check failed")
    if not result["valid_fraction"].dropna().between(0, 1).all():
        raise ValueError("Invalid fraction")
    if not result["ndvi_lag1"].dropna().between(-1, 1).all():
        raise ValueError("Unphysical scaled NDVI")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(f"Saved {len(result):,} rows to {args.output}")
    print("Missing NDVI records:", int(result.ndvi_lag1.isna().sum()))
    print(result.groupby(["year", "month"])[["ndvi_lag1", "ndvi_std", "valid_fraction"]].mean().head(6).to_string())
    print("NOTE: spatial_coverage_approx is a pixel-centre approximation; near-coast coverage and ocean pixels need land-mask QA.")


if __name__ == "__main__":
    main()
