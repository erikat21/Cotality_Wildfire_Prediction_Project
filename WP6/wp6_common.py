"""
Shared pieces for the WP6 scripts: loading the panel with every label definition, and the
feature lists. Imported by 02_dependence.py, 03_logistic.py and 04_test_scores.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PANEL = ROOT / "WP5" / "outputs" / "cell_month_panel.parquet"
LABELS = ROOT / "wp2" / "outputs" / "labels.parquet"
OUT = HERE / "outputs"

SIZES = [30, 100, 383]
XS = ["any", "x01", "x02", "x05", "x10"]
X_LABEL = {"any": "any", "x01": "1%", "x02": "2%", "x05": "5%", "x10": "10%"}
OPEN_YEARS = list(range(2016, 2024))
TEST_YEARS = [2024, 2025]          # scored once, in 04_test_scores.py
FINAL_YEAR = 2026                  # never used in WP6

# Predictors, by group. A column missing from the panel is skipped, so the scripts also run
# on a panel built without weather. water_share is left out because the shares sum to 1.
FEATURES = {
    "weather": ["tmax", "vpdmax", "tdmean", "ppt_log"],
    "vegetation": ["ndvi_lag1"],
    "terrain": ["elev_mean", "elev_std", "slope_mean", "slope_p90", "aspect_south_share"],
    "land cover": ["artificial_share", "agriculture_share", "forest_share",
                   "shrub_herbaceous_share", "bare_sparse_share", "wetland_share"],
}
NICE = {
    "tmax": "max temperature (°C)", "vpdmax": "max VPD (kPa)", "tdmean": "dewpoint (°C)",
    "ppt_log": "log(1 + rain mm)", "ndvi_lag1": "NDVI, previous month",
    "elev_mean": "elevation (m)", "elev_std": "elevation spread (m)", "slope_mean": "mean slope (°)",
    "slope_p90": "steep slope, p90 (°)", "aspect_south_share": "south-facing share",
    "artificial_share": "urban share", "agriculture_share": "farmland share",
    "forest_share": "forest share", "shrub_herbaceous_share": "shrub/grass share",
    "bare_sparse_share": "bare/sparse share", "wetland_share": "wetland share",
}


def label_cols():
    return [f"y_S{S}_{x}" for S in SIZES for x in XS]


def load_panel(years=OPEN_YEARS) -> pd.DataFrame:
    """WP5 panel for the given years, with WP2's label columns joined on (0 where no fire)."""
    if FINAL_YEAR in years:
        raise SystemExit("2026 is the final test year and is not used in WP6")
    panel = pd.read_parquet(PANEL)
    panel = panel[panel.year.isin(years)].copy()
    if set(years) - set(panel.year):
        raise SystemExit(f"panel has no rows for {sorted(set(years) - set(panel.year))}: "
                         "rebuild WP5 with --locked for the test step")

    keep = ["cell_id", "year", "month"] + label_cols() + [f"n_fires_S{S}" for S in SIZES]
    labels = pd.read_parquet(LABELS, columns=keep)
    panel = panel.merge(labels, on=["cell_id", "year", "month"], how="left")
    for c in keep[3:]:
        panel[c] = panel[c].fillna(0).astype(int)

    # WP5's own `fire` column should equal S = 30, x = any (fire_cell only holds fires >= 30 ha)
    if "fire" in panel and (panel.fire != panel.y_S30_any).any():
        print(f"note: WP5 `fire` differs from y_S30_any in {(panel.fire != panel.y_S30_any).sum()} rows")

    if "ppt" in panel:
        panel["ppt_log"] = np.log1p(panel.ppt)
    panel["region"] = panel.nuts2.astype(str)
    panel["ym"] = panel.year * 100 + panel.month       # year-month block for the bootstrap
    return panel


def available(panel: pd.DataFrame, groups=None) -> list[str]:
    groups = groups or list(FEATURES)
    return [f for g in groups for f in FEATURES[g] if f in panel.columns]
