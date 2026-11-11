
"""
Shared pieces for the WP6 scripts: loading the panel with every label definition, and the
feature lists. Imported by 02_dependence.py, 03_logistic.py and 04_test_scores.py.

FIX NOTES (2026-11):
    P0-1: Select the correct WP5 Open or Locked-inclusive panel based on the
          requested years. The original loader always read the Open panel.
    P1-1: Validate panel and label keys before merging to prevent silent
          row duplication or incorrect cell-month alignment.
    P1-2: Verify required label columns and fail on label inconsistencies.
          Preserve optional feature availability for no-weather experiments.
    P1-3: Check requested year-month coverage and reject incomplete panels.
"""

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# FIX P0-1: Separate WP5 Open and Locked-inclusive panel paths.
# Open panel: May 2016 - December 2023.
# Locked-inclusive panel: May 2016 - August 2026.
PANEL_OPEN = ROOT / "WP5" / "outputs" / "cell_month_panel.parquet"
PANEL_LOCKED = ROOT / "WP5" / "outputs" / "cell_month_panel_locked.parquet"

# Keep the original PANEL alias for backward compatibility.
PANEL = PANEL_OPEN

LABELS = ROOT / "wp2" / "outputs" / "labels.parquet"
OUT = HERE / "outputs"


SIZES = [30, 100, 383]
XS = ["any", "x01", "x02", "x05", "x10"]

X_LABEL = {
    "any": "any",
    "x01": "1%",
    "x02": "2%",
    "x05": "5%",
    "x10": "10%",
}

OPEN_YEARS = list(range(2016, 2024))
TEST_YEARS = [2024, 2025]  # scored once, in 04_test_scores.py
FINAL_YEAR = 2026          # never used in WP6


# Predictors, by group.
# A column missing from the panel is skipped, so the scripts also run
# on a panel built without weather.
# water_share is left out because the land-cover shares sum to 1.
FEATURES = {
    "weather": [
        "tmax",
        "vpdmax",
        "tdmean",
        "ppt_log",
    ],
    "vegetation": [
        "ndvi_lag1",
    ],
    "terrain": [
        "elev_mean",
        "elev_std",
        "slope_mean",
        "slope_p90",
        "aspect_south_share",
    ],
    "land cover": [
        "artificial_share",
        "agriculture_share",
        "forest_share",
        "shrub_herbaceous_share",
        "bare_sparse_share",
        "wetland_share",
    ],
}

NICE = {
    "tmax": "max temperature (°C)",
    "vpdmax": "max VPD (kPa)",
    "tdmean": "dewpoint (°C)",
    "ppt_log": "log(1 + rain mm)",
    "ndvi_lag1": "NDVI, previous month",
    "elev_mean": "elevation (m)",
    "elev_std": "elevation spread (m)",
    "slope_mean": "mean slope (°)",
    "slope_p90": "steep slope, p90 (°)",
    "aspect_south_share": "south-facing share",
    "artificial_share": "urban share",
    "agriculture_share": "farmland share",
    "forest_share": "forest share",
    "shrub_herbaceous_share": "shrub/grass share",
    "bare_sparse_share": "bare/sparse share",
    "wetland_share": "wetland share",
}


KEYS = ["cell_id", "year", "month"]


def label_cols():
    """Return all 15 WP2 binary fire-label definitions."""
    return [
        f"y_S{S}_{x}"
        for S in SIZES
        for x in XS
    ]


def require_columns(df: pd.DataFrame, columns: list[str], source: str):
    """Ensure all required columns are available."""
    missing = sorted(set(columns) - set(df.columns))

    if missing:
        raise ValueError(
            f"{source} is missing required columns: {missing}"
        )


def require_unique_keys(df: pd.DataFrame, source: str):
    """
    FIX P1-1: Reject null or duplicate cell-month keys before merging.
    """
    require_columns(df, KEYS, source)

    if df[KEYS].isna().any().any():
        raise ValueError(
            f"{source} contains missing cell-month key values"
        )

    duplicates = df.duplicated(KEYS).sum()

    if duplicates:
        raise ValueError(
            f"{source} contains {duplicates:,} duplicate cell-month keys"
        )


def expected_months(years):
    """
    FIX P1-3: Generate the expected WP5 panel months.

    The first available month is May 2016.
    WP6 never loads the 2026 final-year observations.
    """
    months = set()

    for year in years:
        first_month = 5 if year == 2016 else 1

        for month in range(first_month, 13):
            months.add((year, month))

    return months


def load_panel(years=OPEN_YEARS) -> pd.DataFrame:
    """
    Load the appropriate WP5 panel and merge WP2's fire-label definitions.

    Open-year requests use cell_month_panel.parquet.
    Test-year requests use cell_month_panel_locked.parquet.

    The 2026 final test year is never loaded by WP6.
    """
    years = list(years)

    if not years:
        raise ValueError("At least one year must be requested")

    if FINAL_YEAR in years:
        raise SystemExit(
            "2026 is the final test year and is not used in WP6"
        )

    allowed_years = set(OPEN_YEARS + TEST_YEARS)
    invalid_years = set(years) - allowed_years

    if invalid_years:
        raise ValueError(
            f"Unsupported WP6 years: {sorted(invalid_years)}"
        )

    # FIX P0-1: Choose the correct panel based on requested years.
    # Never fall back to the Open panel for a test-year request.
    use_locked = any(year in TEST_YEARS for year in years)
    panel_path = PANEL_LOCKED if use_locked else PANEL_OPEN

    if not panel_path.is_file():
        raise FileNotFoundError(
            f"Required WP5 panel not found: {panel_path}\n"
            "Generate the corresponding WP5 output first."
        )

    panel = pd.read_parquet(panel_path)

    require_columns(
        panel,
        KEYS + ["nuts2", "fire", "burn_frac"],
        "WP5 panel",
    )

    # FIX P1-1: Validate the panel before filtering and merging.
    require_unique_keys(panel, "WP5 panel")

    panel = panel.loc[panel["year"].isin(years)].copy()

    missing_years = set(years) - set(panel["year"])

    if missing_years:
        raise ValueError(
            f"WP5 panel has no records for years: "
            f"{sorted(missing_years)}"
        )

    # FIX P1-3: Check exact requested month coverage.
    observed_months = set(
        zip(panel["year"], panel["month"])
    )

    required_months = expected_months(years)

    missing_months = required_months - observed_months
    extra_months = observed_months - required_months

    if missing_months or extra_months:
        raise ValueError(
            "WP5 panel month coverage mismatch.\n"
            f"Missing: {sorted(missing_months)}\n"
            f"Unexpected: {sorted(extra_months)}"
        )

    # FIX P1-3: Ensure every included cell has one row per month.
    monthly_counts = panel.groupby(["year", "month"]).size()

    if monthly_counts.nunique() != 1:
        raise ValueError(
            "WP5 panel has inconsistent cell counts across months"
        )

    n_cells = panel["cell_id"].nunique()
    n_months = len(required_months)
    expected_rows = n_cells * n_months

    if len(panel) != expected_rows:
        raise ValueError(
            f"Incomplete WP5 panel: {len(panel):,} rows, "
            f"expected {expected_rows:,}"
        )

    # WP2 official labels for all S and x definitions.
    keep = (
        KEYS
        + label_cols()
        + [f"n_fires_S{S}" for S in SIZES]
    )

    if not LABELS.is_file():
        raise FileNotFoundError(
            f"WP2 labels not found: {LABELS}"
        )

    labels = pd.read_parquet(LABELS)

    # FIX P1-2: Validate required label columns and keys.
    require_columns(labels, keep, "WP2 labels")
    require_unique_keys(labels, "WP2 labels")

    labels = labels[keep].copy()

    original_rows = len(panel)

    # FIX P1-1: Explicit join cardinality prevents silent row expansion.
    panel = panel.merge(
        labels,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    if len(panel) != original_rows:
        raise ValueError(
            "WP2 label merge unexpectedly changed panel row count"
        )

    # WP2 labels are sparse: absent cell-months are treated as negatives.
    for col in keep[3:]:
        panel[col] = panel[col].fillna(0).astype(int)

    # FIX P1-2: The revised WP5 uses WP2's official y_S30_any label.
    # A mismatch is now a validation error rather than a printed note.
    binary_disagreements = (
        panel["fire"] != panel["y_S30_any"]
    ).sum()

    if binary_disagreements:
        raise ValueError(
            "WP5 fire labels disagree with WP2 y_S30_any: "
            f"{binary_disagreements:,} rows"
        )

    # Weather precipitation transformation.
    # Keep the original WP6 feature definition.
    if "ppt" in panel.columns:
        ppt = pd.to_numeric(panel["ppt"], errors="coerce")

        if ppt.isna().any() or (ppt < 0).any():
            raise ValueError(
                "Invalid precipitation values in WP5 panel"
            )

        panel["ppt_log"] = np.log1p(ppt)

    panel["region"] = panel["nuts2"].astype(str)

    # Year-month blocks used by the existing WP6 bootstrap.
    panel["ym"] = (
        panel["year"] * 100 + panel["month"]
    )

    return panel


def available(
    panel: pd.DataFrame,
    groups=None,
    strict: bool = False,
) -> list[str]:
    """
    Return the model predictors available in the panel.

    Default behavior preserves the original WP6 interface and supports
    deliberate no-weather experiments.

    FIX P1-2: Set strict=True to reject missing predictors in an
    official full-feature experiment.
    """
    groups = list(FEATURES) if groups is None else list(groups)

    unknown_groups = set(groups) - set(FEATURES)

    if unknown_groups:
        raise ValueError(
            f"Unknown WP6 feature groups: {sorted(unknown_groups)}"
        )

    requested = [
        feature
        for group in groups
        for feature in FEATURES[group]
    ]

    missing = [
        feature
        for feature in requested
        if feature not in panel.columns
    ]

    if strict and missing:
        raise ValueError(
            f"Missing required WP6 predictors: {missing}"
        )

    return [
        feature
        for feature in requested
        if feature in panel.columns
    ]
