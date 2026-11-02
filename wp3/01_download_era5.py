"""
WP3 step 1: download ERA5-Land over the vn0 box, one file per variable per month.

Three variables, two datasets (see the vn0 brief, "The ERA5-Land download"):
    tmax   derived-era5-land-daily-statistics  2m_temperature          daily_maximum
    tdmean derived-era5-land-daily-statistics  2m_dewpoint_temperature daily_mean
    tp     reanalysis-era5-land                total_precipitation     time 00:00 only

The grid is the SAME box WP2 built its cells on, so every ERA5-Land point lands on
exactly one cell centre:
    area = [N 44.0, W -9.8, S 35.9, E 4.4]   (82 lats x 143 lons = 11,726 points)

Lite tier (start here): daily statistics for temperature, hourly 00:00 for rain.
Files come back as NetCDF (.nc): the derived daily-statistics dataset has no GRIB
option, and the monthly requests are small enough that we ask rain for NetCDF too so
step 2 reads everything the same way.

Downloading in blocks (brief slide "Downloading in blocks"). Test/final years are
written to a SEPARATE locked folder so no analysis script reads them by accident:
    pilot    2016-2019   -> data/era5/
    training 2020-2023   -> data/era5/
    margins15 2015       -> data/era5/        (antecedent weather, used by training)
    test     2024-2025   -> data/era5_locked/
    final    2026 Jan-Aug -> data/era5_locked/

Precipitation needs one extra timestamp: the 00:00 value on day D+1 is day D's total,
so every month is downloaded with the FIRST day of the next month appended. That makes
each monthly file self-contained for step 2's day-shift, and means we fetch 2026 through
September 1 to close out August.

Setup:
    pip install "cdsapi>=0.7"
    A Copernicus (CDS) account + ~/.cdsapirc holding your url and key. See README.

Run (resumable; existing files are skipped):
    python 01_download_era5.py --block pilot
    python 01_download_era5.py --block training
    python 01_download_era5.py --block margins15
    python 01_download_era5.py --block test       # locked folder
    python 01_download_era5.py --block final      # locked folder
    python 01_download_era5.py --block all
    python 01_download_era5.py --block pilot --vars tmax     # just one variable
    python 01_download_era5.py --block pilot --dry-run       # print requests, download nothing
"""
import argparse
import calendar
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPEN_DIR = HERE / "data" / "era5"
LOCKED_DIR = HERE / "data" / "era5_locked"

# vn0 box, in CDS order [North, West, South, East]
AREA = [44.0, -9.8, 35.9, 4.4]
# derived-era5-land-daily-statistics returns NetCDF (no data_format key, per the brief);
# we ask reanalysis-era5-land for NetCDF too so every file reads the same way. Monthly
# requests are small (~0.5 MB), so GRIB's size advantage does not matter here.
NETCDF = "netcdf"

# Which years belong to each download block, and whether they are locked.
# A year is locked (held out until the test step) if it is a scoring year: 2024-2026.
LOCKED_YEARS = {2024, 2025, 2026}
BLOCKS = {
    "pilot": [2016, 2017, 2018, 2019],
    "training": [2020, 2021, 2022, 2023],
    "margins15": [2015],
    "test": [2024, 2025],
    "final": [2026],
}
VARS = ("tmax", "tdmean", "tp")


def months_for(var: str, year: int) -> list[int]:
    """The panel is May 2016 to Aug 2026; 2015 is antecedent weather (full year).

    tp is fetched one month further than the panel because step 2 shifts the 00:00
    stamp back a day, so August 2026 needs September 2026's first day.
    """
    last = 12
    if year == 2026:
        last = 9 if var == "tp" else 8
    return list(range(1, last + 1))


def out_path(var: str, year: int, month: int) -> Path:
    base = LOCKED_DIR if year in LOCKED_YEARS else OPEN_DIR
    return base / var / f"{var}_{year}_{month:02d}.nc"


def temperature_request(variable: str, statistic: str, year: int, month: int) -> dict:
    """derived-era5-land-daily-statistics. Note: NO data_format key (brief)."""
    days = [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)]
    return {
        "variable": [variable],
        "year": str(year),
        "month": f"{month:02d}",
        "day": days,
        "daily_statistic": statistic,
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": AREA,
    }


def precip_request(year: int, month: int) -> dict:
    """reanalysis-era5-land, total_precipitation, time 00:00 only.

    Days = this month + the 1st of next month (that 00:00 stamp carries this month's
    last day's total). multiply tp by 1000 for mm happens in step 2, not here.
    """
    days = [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)]
    if month == 12:
        next_year, next_month = year + 1, 1
    else:
        next_year, next_month = year, month + 1
    req = {
        "variable": ["total_precipitation"],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": days,
        "time": ["00:00"],
        "data_format": NETCDF,
        "area": AREA,
    }
    # The next-day 00:00 stamp lives in next month: a tiny second request merged by step 2.
    req["_next"] = {
        "variable": ["total_precipitation"],
        "year": [str(next_year)],
        "month": [f"{next_month:02d}"],
        "day": ["01"],
        "time": ["00:00"],
        "data_format": NETCDF,
        "area": AREA,
    }
    return req


# dataset id and request builder for each variable
SPEC = {
    "tmax": ("derived-era5-land-daily-statistics",
             lambda y, m: temperature_request("2m_temperature", "daily_maximum", y, m)),
    "tdmean": ("derived-era5-land-daily-statistics",
               lambda y, m: temperature_request("2m_dewpoint_temperature", "daily_mean", y, m)),
    "tp": ("reanalysis-era5-land", precip_request),
}


def _fetch(client, dataset: str, request: dict, target: Path, tries: int = 4) -> None:
    """Retrieve to `target`, retrying the transient truncation the CDS client raises as
    an AssertionError ('File size mismatch'), and unzipping if CDS returns a zip.

    ERA5-Land often packs results in a .zip holding a single .nc; we extract it so every
    file on disk is a plain NetCDF that step 2 reads uniformly.
    """
    import time
    import zipfile

    tmp = target.with_suffix(target.suffix + ".download")
    for attempt in range(1, tries + 1):
        try:
            client.retrieve(dataset, request, str(tmp))
            break
        except AssertionError as e:  # multiurl size-mismatch on a truncated transfer
            if tmp.exists():
                tmp.unlink()
            if attempt == tries:
                raise
            wait = 5 * attempt
            print(f"    truncated ({e}); retry {attempt}/{tries - 1} in {wait}s")
            time.sleep(wait)

    if zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as zf:
            members = [m for m in zf.namelist() if m.endswith(".nc")] or zf.namelist()
            if len(members) != 1:
                raise RuntimeError(f"{target.name}: expected one file in the zip, got {members}")
            with zf.open(members[0]) as src, open(target, "wb") as out:
                out.write(src.read())
        tmp.unlink()
    else:
        tmp.replace(target)


def download_one(client, var: str, year: int, month: int, dry_run: bool) -> str:
    target = out_path(var, year, month)
    if target.exists() and target.stat().st_size > 0:
        return "skip (exists)"
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset, build = SPEC[var]
    request = build(year, month)

    if var == "tp":
        # main month -> target; the single next-day stamp -> a sidecar step 2 concatenates
        nxt = request.pop("_next")
        if dry_run:
            print(f"  [dry-run] {dataset} {var} {year}-{month:02d} -> {target.name}")
            print(f"  [dry-run] {dataset} {var} next-day -> {target.with_suffix('.next.nc').name}")
            return "dry-run"
        _fetch(client, dataset, request, target)
        _fetch(client, dataset, nxt, target.with_suffix(".next.nc"))
        return "ok"

    if dry_run:
        print(f"  [dry-run] {dataset} {var} {year}-{month:02d} -> {target.name}")
        print(f"            {request}")
        return "dry-run"
    _fetch(client, dataset, request, target)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=list(BLOCKS) + ["all"])
    ap.add_argument("--vars", nargs="+", default=list(VARS), choices=VARS,
                    help="which variables to fetch (default: all three)")
    ap.add_argument("--dry-run", action="store_true", help="print requests, download nothing")
    args = ap.parse_args()

    years = sorted({y for b in BLOCKS.values() for y in b}) if args.block == "all" else BLOCKS[args.block]

    client = None
    if not args.dry_run:
        import cdsapi
        client = cdsapi.Client()

    n_ok = n_skip = 0
    for var in args.vars:
        for year in years:
            for month in months_for(var, year):
                folder = "locked" if year in LOCKED_YEARS else "open"
                status = download_one(client, var, year, month, args.dry_run)
                print(f"{var} {year}-{month:02d} [{folder}]: {status}")
                n_ok += status in ("ok", "dry-run")
                n_skip += status.startswith("skip")
    print(f"\ndone: {n_ok} fetched/planned, {n_skip} skipped")


if __name__ == "__main__":
    main()
