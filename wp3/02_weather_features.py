"""
WP3 step 2: turn the downloaded daily ERA5-Land files into `weather_cell_month`.

One row per (cell_id, year, month) with the four vn0 weather features (brief slide 14):
    tmax    mean of daily max 2m temperature          (deg C)
    tdmean  mean of daily mean 2m dewpoint             (deg C)
    vpdmax  mean of daily max VPD                      (kPa, lite-tier approx)
    ppt     sum of daily precipitation totals          (mm)

Lite-tier VPD: each day's max VPD is approximated from that day's max temperature and
mean dewpoint, VPD = es(tmax) - es(tdmean), with the Tetens saturation formula. The
brief notes this sits ~0.13 kPa under the true hourly maximum but keeps the same
ranking, so never mix it with full-tier VPD in one fitted model.

Every ERA5-Land point sits on exactly one WP2 cell centre, so cell_id comes straight
from the point's lat/lon with the shared rule (never change it):
    row = round((44.0 - lat) / 0.1)      # row 0 = northernmost
    col = round((lon + 9.8) / 0.1)
    cell_id = row * 143 + col

Precipitation day-shift: ERA5-Land tp accumulates from 00 UTC, so the 00:00 stamp on
day D+1 is day D's total. Step 1 saved each month with the next day's stamp as a
`.next.grib` sidecar; here each stamp is moved back one day before summing by month.

Inputs:
    data/era5/{tmax,tdmean,tp}/<var>_<year>_<month>.nc          open years (NetCDF)
    data/era5_locked/{tmax,tdmean,tp}/...                       2024-2026 (read only at --locked)
    ../wp2/outputs/cells.parquet  (optional) to keep in-mask cells and attach country/nuts2
Outputs (outputs/):
    weather_cell_month.parquet    cell_id, year, month, tmax, vpdmax, tdmean, ppt
    weather_cell_month_check.csv   coverage per month (cells, missing, ranges)

Run:
    pip install xarray netCDF4 pandas numpy pyarrow     # cfgrib/eccodes only if any GRIB files
    python 02_weather_features.py                 # open years only (2015-2023)
    python 02_weather_features.py --locked        # include 2024-2026 (test step only!)
    python 02_weather_features.py --all-cells      # keep every land point, not just the mask
"""
import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
OPEN_DIR = HERE / "data" / "era5"
LOCKED_DIR = HERE / "data" / "era5_locked"
OUT = HERE / "outputs"
# WP2's cells table, copied into WP3 so this package is self-contained (same file that
# `wp2/01_build_cells.py` produces -- the cell_id rule below reproduces its mapping).
CELLS = HERE / "data" / "cells.parquet"

N_COLS = 143
NORTH, WEST = 44.0, -9.8
LOCKED_YEARS = {2024, 2025, 2026}
KELVIN = 273.15

LAT_NAMES = ("latitude", "lat")
LON_NAMES = ("longitude", "lon")
TIME_NAMES = ("valid_time", "time")


def es_kpa(t_c: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure (kPa) from temperature (deg C), Tetens formula."""
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def cell_id_from(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    row = np.round((NORTH - lat) / 0.1).astype(int)
    col = np.round((lon - WEST) / 0.1).astype(int)
    return row * N_COLS + col


def _pick(cols, names):
    for n in names:
        if n in cols:
            return n
    raise KeyError(f"none of {names} found in {list(cols)}")


def _engine_for(path: Path) -> str:
    """Pick an xarray engine from the file's magic bytes (CDS sends NetCDF4/HDF5;
    cfgrib is kept as a fallback in case a file was fetched as GRIB)."""
    with open(path, "rb") as fh:
        head = fh.read(4)
    if head[:3] == b"CDF" or head == b"\x89HDF":
        return "netcdf4"
    if head == b"GRIB":
        return "cfgrib"
    raise ValueError(f"{path.name}: unrecognised format (first bytes {head!r}); "
                     "a zip usually means a failed/partial download -- delete and refetch")


def open_long(path: Path, value_name: str) -> pd.DataFrame:
    """Read one ERA5-Land file to a long frame: cell_id, date (daily), <value_name>."""
    engine = _engine_for(path)
    kwargs = {"backend_kwargs": {"indexpath": ""}} if engine == "cfgrib" else {}
    ds = xr.open_dataset(path, engine=engine, **kwargs)
    var = list(ds.data_vars)[0]
    df = ds[var].to_dataframe(name=value_name).reset_index()
    ds.close()

    latc = _pick(df.columns, LAT_NAMES)
    lonc = _pick(df.columns, LON_NAMES)
    timec = _pick(df.columns, TIME_NAMES)

    df = df.dropna(subset=[value_name])
    df["cell_id"] = cell_id_from(df[latc].to_numpy(), df[lonc].to_numpy())
    df["date"] = pd.to_datetime(df[timec]).dt.normalize()
    return df[["cell_id", "date", value_name]]


def file_for(var: str, year: int, month: int, use_locked: bool) -> Path | None:
    base = LOCKED_DIR if year in LOCKED_YEARS else OPEN_DIR
    if year in LOCKED_YEARS and not use_locked:
        return None
    p = base / var / f"{var}_{year}_{month:02d}.nc"
    return p if p.exists() else None


def monthly_temp_vpd(year: int, month: int, use_locked: bool) -> pd.DataFrame | None:
    tmax_f = file_for("tmax", year, month, use_locked)
    td_f = file_for("tdmean", year, month, use_locked)
    if tmax_f is None or td_f is None:
        return None
    tmax = open_long(tmax_f, "tmax_k")
    td = open_long(td_f, "td_k")
    d = tmax.merge(td, on=["cell_id", "date"], how="inner")
    d["tmax_c"] = d["tmax_k"] - KELVIN
    d["td_c"] = d["td_k"] - KELVIN
    d["vpd"] = (es_kpa(d["tmax_c"]) - es_kpa(d["td_c"])).clip(lower=0.0)
    g = d.groupby("cell_id").agg(
        tmax=("tmax_c", "mean"), tdmean=("td_c", "mean"), vpdmax=("vpd", "mean"),
        n_days=("date", "nunique"))
    return g.reset_index()


def monthly_ppt(year: int, month: int, use_locked: bool) -> pd.DataFrame | None:
    tp_f = file_for("tp", year, month, use_locked)
    if tp_f is None:
        return None
    parts = [open_long(tp_f, "tp_m")]
    nxt = tp_f.with_suffix(".next.nc")
    if nxt.exists():
        parts.append(open_long(nxt, "tp_m"))
    tp = pd.concat(parts, ignore_index=True)
    # 00:00 stamp of day D+1 is day D's total -> move every stamp back one day
    tp["day"] = tp["date"] - pd.Timedelta(days=1)
    tp = tp[(tp["day"].dt.year == year) & (tp["day"].dt.month == month)]
    tp["ppt"] = tp["tp_m"] * 1000.0  # m -> mm
    # one stamp per day already; sum the daily totals over the month
    daily = tp.groupby(["cell_id", "day"], as_index=False)["ppt"].first()
    return daily.groupby("cell_id", as_index=False)["ppt"].sum()


def panel_months(use_locked: bool):
    """May 2016 to Aug 2026 (the panel). 2015 is downloaded for antecedent use but is
    not part of the panel, so it is skipped here."""
    for year in range(2016, 2027):
        first = 5 if year == 2016 else 1
        last = 8 if year == 2026 else 12
        for month in range(first, last + 1):
            if year in LOCKED_YEARS and not use_locked:
                continue
            yield year, month


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", action="store_true",
                    help="include 2024-2026 (read the locked folder) -- test step only")
    ap.add_argument("--all-cells", action="store_true",
                    help="keep every ERA5 land point, not just the WP2 in-mask cells")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    mask = None
    if CELLS.exists():
        cells = pd.read_parquet(CELLS)
        mask = cells if args.all_cells else cells[cells.in_mask == 1]
        mask = mask[["cell_id", "country", "nuts2"]]
    else:
        print(f"note: {CELLS} not found -- run WP2 step 1 for the mask; keeping all points")

    rows, checks = [], []
    for year, month in panel_months(args.locked):
        tv = monthly_temp_vpd(year, month, args.locked)
        pp = monthly_ppt(year, month, args.locked)
        if tv is None and pp is None:
            print(f"{year}-{month:02d}: no files, skipped")
            continue
        if tv is None or pp is None:
            print(f"{year}-{month:02d}: WARNING missing {'temp' if tv is None else 'precip'}, skipped")
            continue
        m = tv.merge(pp, on="cell_id", how="outer")
        m["year"], m["month"] = year, month
        if mask is not None:
            m = m.merge(mask, on="cell_id", how="inner")
        rows.append(m)
        exp_days = calendar.monthrange(year, month)[1]
        checks.append({
            "year": year, "month": month, "cells": len(m),
            "short_months": int((m["n_days"] < exp_days).sum()),
            "tmax_min": round(m.tmax.min(), 1), "tmax_max": round(m.tmax.max(), 1),
            "vpdmax_max": round(m.vpdmax.max(), 2), "ppt_max": round(m.ppt.max(), 1),
            "ppt_zero_cells": int((m.ppt == 0).sum()),
        })
        print(f"{year}-{month:02d}: {len(m):,} cells")

    if not rows:
        raise SystemExit("no monthly files were read -- run 01_download_era5.py first")

    panel = pd.concat(rows, ignore_index=True)
    keys = ["cell_id", "year", "month"]
    feat = ["tmax", "vpdmax", "tdmean", "ppt"]
    extra = [c for c in ("country", "nuts2") if c in panel.columns]
    panel = panel[keys + feat + extra].sort_values(keys)
    # cast to float64 first so rounding stores clean values, not float32 noise (21.700001)
    for c in ("tmax", "tdmean"):
        panel[c] = panel[c].astype("float64").round(2)
    panel["vpdmax"] = panel["vpdmax"].astype("float64").round(3)
    panel["ppt"] = panel["ppt"].astype("float64").round(2)

    panel.to_parquet(OUT / "weather_cell_month.parquet", index=False)
    pd.DataFrame(checks).to_csv(OUT / "weather_cell_month_check.csv", index=False)

    print(f"\nweather_cell_month: {len(panel):,} rows, "
          f"{panel.cell_id.nunique():,} cells x {len(checks)} months")
    print(f"tmax  C: {panel.tmax.min():.1f} to {panel.tmax.max():.1f}")
    print(f"vpdmax kPa: {panel.vpdmax.min():.2f} to {panel.vpdmax.max():.2f}")
    print(f"ppt  mm: {panel.ppt.min():.1f} to {panel.ppt.max():.1f}")
    miss = panel[feat].isna().sum()
    if miss.any():
        print(f"missing values per feature:\n{miss[miss > 0].to_string()}")


if __name__ == "__main__":
    main()
