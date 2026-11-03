"""WP4: resumable Copernicus GLO-30 downloads for the Iberian grid.

Run from repository root. Uses the WP2 in-mask cell polygons to avoid
requesting OpenTopography blocks that do not intersect the study area.

Examples:
  python WP4/scripts/01_download_dem.py --plan
  python WP4/scripts/01_download_dem.py --limit 1
  python WP4/scripts/01_download_dem.py

Set OPENTOPO_API_KEY in the environment before downloading. ($env:OPENTOPO_API_KEY = "YOUR_API_KEY")
All output DEMs include a 0.06-degree buffer (halo) beyond the named 2-degree
core block. Keep the manifest: later processing must use each block's core
bounds to prevent overlapping DEM areas from being counted twice.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import requests
import rasterio
from shapely.geometry import box

API = "https://portal.opentopography.org/API/globaldem"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CELLS = Path("wp2/outputs/cells.gpkg")
DEFAULT_OUT = ROOT / "data/raw/dem/blocks"
DEFAULT_MANIFEST = ROOT / "outputs/dem_blocks_manifest.csv"


def make_blocks(cells_file: Path, block_size: float, halo: float):
    if not cells_file.exists():
        raise FileNotFoundError(f"WP2 grid not found: {cells_file}")
    cells = gpd.read_file(cells_file, columns=["cell_id", "in_mask", "geometry"])
    if cells.crs is None:
        raise ValueError("WP2 geometry has no CRS")
    cells = cells.loc[cells.in_mask == 1, ["cell_id", "geometry"]].to_crs("EPSG:4326")
    if cells.empty:
        raise ValueError("No in-mask WP2 cells")
    # Covers the cells, including their half-grid-width edges.
    x0, y0, x1, y1 = cells.total_bounds
    import math
    first_w = math.floor(x0 / block_size) * block_size
    first_s = math.floor(y0 / block_size) * block_size
    blocks = []
    for j in range(math.ceil((y1-first_s) / block_size)):
        s = first_s + j * block_size
        n = s + block_size
        for i in range(math.ceil((x1-first_w) / block_size)):
            w = first_w + i * block_size
            e = w + block_size
            # A block must contain at least some part of an in-mask ERA5 cell.
            core = box(w, s, e, n)
            hits = cells.intersects(core)
            if not hits.any():
                continue
            name = f"cop30_w{w:+07.2f}_s{s:+06.2f}"
            blocks.append({
                "block": name, "west": round(w, 5), "south": round(s, 5),
                "east": round(e, 5), "north": round(n, 5),
                "download_west": round(w-halo, 5),
                "download_south": round(s-halo, 5),
                "download_east": round(e+halo, 5),
                "download_north": round(n+halo, 5),
                "intersecting_cells": int(hits.sum()),
                "file": f"{name}.tif",
            })
    return sorted(blocks, key=lambda x: (-x["intersecting_cells"], x["block"]))


def verify_geotiff(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 100_000:
        return False
    try:
        with rasterio.open(path) as ds:
            return ds.count >= 1 and ds.width > 100 and ds.height > 100 and ds.crs is not None
    except Exception:
        return False


def download_one(row, out: Path, key: str, retries: int, timeout: int):
    destination = out / row["file"]
    if verify_geotiff(destination):
        print(f"SKIP (verified) {destination.name}", flush=True)
        return "existing"
    params = {
        "demtype": "COP30", "south": row["download_south"],
        "north": row["download_north"], "west": row["download_west"],
        "east": row["download_east"], "outputFormat": "GTiff", "API_Key": key,
    }
    tmp = destination.with_suffix(".tif.part")
    for attempt in range(1, retries+1):
        try:
            # Never print response.url: it includes the private API key.
            with requests.get(API, params=params, stream=True, timeout=(30, timeout)) as resp:
                if resp.status_code in (401, 403, 429):
                    raise RuntimeError(f"HTTP {resp.status_code}: API key, quota or access issue; stop and check OpenTopography account")
                resp.raise_for_status()
                ctype = resp.headers.get("Content-Type", "").lower()
                if "text/" in ctype or "json" in ctype or "html" in ctype:
                    raise RuntimeError(f"Unexpected content type: {ctype}; no GeoTIFF saved")
                with tmp.open("wb") as f:
                    for part in resp.iter_content(chunk_size=1024*1024):
                        if part:
                            f.write(part)
            if not verify_geotiff(tmp):
                raise RuntimeError("Downloaded file was not a valid GeoTIFF")
            tmp.replace(destination)
            print(f"OK {destination.name}: {destination.stat().st_size/1024**2:.1f} MiB", flush=True)
            return "downloaded"
        except Exception as ex:
            tmp.unlink(missing_ok=True)
            if "API key, quota" in str(ex):
                raise
            if attempt == retries:
                print(f"FAILED {row['block']}: {type(ex).__name__}: {ex}", file=sys.stderr)
                return "failed"
            print(f"Retry {attempt}/{retries} for {row['block']}: {type(ex).__name__}", flush=True)
            time.sleep(min(10*attempt, 40))
    return "failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--block-size", type=float, default=2.0)
    parser.add_argument("--halo", type=float, default=0.06)
    parser.add_argument("--plan", action="store_true", help="Create manifest only; no requests")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of blocks to visit; 0=all")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    a = parser.parse_args()
    if a.block_size <= 0 or a.halo <= 0:
        parser.error("block-size and halo must be positive")
    blocks = make_blocks(a.cells, a.block_size, a.halo)
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    with a.manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(blocks[0]))
        writer.writeheader()
        writer.writerows(blocks)
    print(f"Planned {len(blocks)} geographic blocks -> {a.manifest}", flush=True)
    print("Sample blocks:", flush=True)
    for r in blocks[:5]:
        print(f"  {r['block']}: {r['intersecting_cells']} intersecting cells", flush=True)
    if a.plan:
        return
    key = os.getenv("OPENTOPO_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set OPENTOPO_API_KEY before downloading. Do not commit API keys.")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    chosen = blocks[:a.limit] if a.limit > 0 else blocks
    status = {"existing": 0, "downloaded": 0, "failed": 0}
    for i, row in enumerate(chosen, 1):
        print(f"[{i}/{len(chosen)}] {row['block']}", flush=True)
        try:
            status[download_one(row, a.out_dir, key, a.retries, a.timeout)] += 1
        except RuntimeError as ex:
            print(str(ex), file=sys.stderr)
            break
    print(f"Completed: {status}. Re-run to skip verified downloads.")


if __name__ == "__main__":
    main()
