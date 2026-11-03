"""
WP4 — CORINE Land Cover Processing
DSC 180A: Iberian Wildfire Prediction Project

Inputs:
    wp2/outputs/cells.gpkg
    WP4/data/raw/corine/2012/U2018_CLC2012_V2020_20u1.tif
    WP4/data/raw/corine/2018/U2018_CLC2018_V2020_20u1.tif
    Corresponding .tif.vat.dbf files

Output:
    WP4/outputs/landcover_cell.parquet

Each output row represents one ERA5-Land grid cell
for one CORINE epoch (2012 or 2018).

Land cover categories:
    artificial
    agriculture
    forest
    shrub_herbaceous
    bare_sparse
    wetland
    water
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from dbfread import DBF
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window


ROOT = Path(__file__).resolve().parents[2]
WP4 = ROOT / "WP4"

CELLS_PATH = ROOT / "wp2" / "outputs" / "cells.gpkg"

CORINE_DIR = WP4 / "data" / "raw" / "corine"

OUTPUT_DIR = WP4 / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "landcover_cell.parquet"


CORINE_FILES = {
    2012: (
        CORINE_DIR
        / "2012"
        / "U2018_CLC2012_V2020_20u1.tif"
    ),
    2018: (
        CORINE_DIR
        / "2018"
        / "U2018_CLC2018_V2020_20u1.tif"
    ),
}


GROUPS = [
    "artificial",
    "agriculture",
    "forest",
    "shrub_herbaceous",
    "bare_sparse",
    "wetland",
    "water",
]


def clc_to_group(code):
    """
    Convert a three-digit CORINE CLC code to one
    of seven grouped land cover categories.
    """

    code = str(code).strip()

    if len(code) != 3 or not code.isdigit():
        raise ValueError(f"Invalid CORINE code: {code}")

    if code.startswith("1"):
        return "artificial"

    if code.startswith("2"):
        return "agriculture"

    if code.startswith("31"):
        return "forest"

    if code.startswith("32"):
        return "shrub_herbaceous"

    if code.startswith("33"):
        return "bare_sparse"

    if code.startswith("4"):
        return "wetland"

    if code.startswith("5"):
        return "water"

    raise ValueError(f"Unknown CORINE code: {code}")


def load_clc_mapping(tif_path, year):
    """
    Read the raster attribute table (VAT).

    Raster pixel values are not necessarily the
    three-digit CORINE classification codes.

    For example:
        Raster Value 12 -> CODE_12 211 -> Agriculture
    """

    dbf_path = Path(str(tif_path) + ".vat.dbf")

    if not dbf_path.exists():
        raise FileNotFoundError(
            f"Missing raster attribute table: {dbf_path}"
        )

    table = DBF(
        str(dbf_path),
        encoding="latin1",
        load=True
    )

    code_field = f"CODE_{str(year)[-2:]}"

    if "Value" not in table.field_names:
        raise ValueError("DBF missing 'Value' field")

    if code_field not in table.field_names:
        raise ValueError(
            f"DBF missing '{code_field}' field. "
            f"Found: {table.field_names}"
        )

    mapping = {}

    for row in table:
        value = int(row["Value"])
        code = str(row[code_field]).strip()

        if value in mapping:
            raise ValueError(f"Duplicate raster value: {value}")

        # Special CORINE category: unclassified / NoData
        if code == "999":
            mapping[value] = {
                "code": code,
                "group": None,
                "label": str(row.get("LABEL3", "")),
            }
            print(f"  Excluding special category: Value={value}, CODE={code}")
            continue

        mapping[value] = {
            "code": code,
            "group": clc_to_group(code),
            "label": str(row.get("LABEL3", "")),
        }

    print(
        f"Loaded {len(mapping)} CORINE classes "
        f"for {year}"
    )

    return mapping


def load_cells():
    print("Loading WP2 cells...")

    if not CELLS_PATH.exists():
        raise FileNotFoundError(CELLS_PATH)

    cells = gpd.read_file(CELLS_PATH)

    required = {"cell_id", "in_mask", "geometry"}

    missing = required - set(cells.columns)

    if missing:
        raise ValueError(
            f"Missing WP2 columns: {missing}"
        )

    cells = cells[
        cells["in_mask"] == 1
    ].copy()

    cells = cells.sort_values(
        "cell_id"
    ).reset_index(drop=True)

    if cells["cell_id"].duplicated().any():
        raise ValueError("Duplicate WP2 cell_id")

    if cells.crs is None:
        raise ValueError("WP2 grid has no CRS")

    print(f"In-mask cells: {len(cells):,}")
    print(f"Grid CRS: {cells.crs}")

    return cells


def read_cell_pixels(src, geom):
    """
    Read only the raster window overlapping a cell.

    A pixel is assigned to the cell if its center
    falls inside the cell polygon.

    Returns:
        Pixel values inside the cell
        Number of cell pixels represented by raster
    """

    left, bottom, right, top = geom.bounds

    window = from_bounds(
        left,
        bottom,
        right,
        top,
        transform=src.transform
    )

    window = window.round_offsets().round_lengths()

    # Clip to the raster's dimensions.
    raster_window = Window(
        0,
        0,
        src.width,
        src.height
    )

    try:
        window = window.intersection(raster_window)
    except rasterio.errors.WindowError:
        return np.array([], dtype=np.int16), 0

    if window.width <= 0 or window.height <= 0:
        return np.array([], dtype=np.int16), 0

    data = src.read(
        1,
        window=window,
        masked=True
    )

    transform = src.window_transform(window)

    inside = geometry_mask(
        [geom.__geo_interface__],
        out_shape=data.shape,
        transform=transform,
        invert=True,
        all_touched=False
    )

    total_pixels = int(inside.sum())

    # Exclude masked/NoData pixels.
    valid = inside & ~np.ma.getmaskarray(data)

    values = np.asarray(data.data[valid])

    return values, total_pixels


def calculate_cell_shares(values, total_pixels, mapping):
    """
    Convert raster values into seven class shares.

    Shares are normalized by valid classified pixels.
    """

    result = {
        f"{group}_share": np.nan
        for group in GROUPS
    }

    result["valid_fraction"] = 0.0
    result["n_valid_pixels"] = 0
    result["n_total_pixels"] = total_pixels
    result["dominant_class"] = None
    result["share_sum"] = np.nan

    if total_pixels == 0 or len(values) == 0:
        return result

    unique, counts = np.unique(
        values,
        return_counts=True
    )

    group_counts = {
        group: 0
        for group in GROUPS
    }

    valid_count = 0
    unknown = []

    for value, count in zip(unique, counts):
        value = int(value)
        count = int(count)

        if value not in mapping:
            unknown.append((value, count))
            continue

        group = mapping[value]["group"]

        # Exclude unclassified / NoData categories.
        if group is None:
            continue

        group_counts[group] += count
        valid_count += count

    if unknown:
        raise ValueError(
            f"Unmapped raster values: {unknown}"
        )

    if valid_count == 0:
        return result

    for group in GROUPS:
        result[f"{group}_share"] = (
            group_counts[group] / valid_count
        )

    result["valid_fraction"] = (
        valid_count / total_pixels
    )

    result["n_valid_pixels"] = valid_count

    result["dominant_class"] = max(
        GROUPS,
        key=lambda group: group_counts[group]
    )

    result["share_sum"] = sum(
        result[f"{group}_share"]
        for group in GROUPS
    )

    return result


def process_epoch(cells, year, tif_path):

    print("\n" + "=" * 60)
    print(f"Processing CORINE {year}")
    print("=" * 60)

    if not tif_path.exists():
        raise FileNotFoundError(tif_path)

    mapping = load_clc_mapping(
        tif_path,
        year
    )

    records = []

    with rasterio.open(tif_path) as src:

        print("Raster shape:", src.shape)
        print("Raster CRS:", src.crs)
        print("Resolution:", src.res)
        print("NoData:", src.nodata)

        if src.crs is None:
            raise ValueError(
                f"CORINE {year} raster has no CRS"
            )

        # Transform the much smaller WP2 grid rather
        # than reprojecting the entire European raster.
        projected_cells = cells.to_crs(src.crs)

        for i, row in enumerate(
            projected_cells.itertuples(index=False),
            start=1
        ):
            values, total_pixels = read_cell_pixels(
                src,
                row.geometry
            )

            features = calculate_cell_shares(
                values,
                total_pixels,
                mapping
            )

            record = {
                "cell_id": int(row.cell_id),
                "epoch": year,
            }

            record.update(features)

            records.append(record)

            if i % 500 == 0 or i == len(projected_cells):
                print(
                    f"  Processed {i:,}/"
                    f"{len(projected_cells):,} cells"
                )

    df = pd.DataFrame(records)

    print(f"\nCORINE {year} complete.")
    print("Rows:", len(df))

    print(
        "Cells with no valid pixels:",
        int((df["n_valid_pixels"] == 0).sum())
    )

    print(
        "Mean valid fraction:",
        round(df["valid_fraction"].mean(), 6)
    )

    return df


def validate_output(df, expected_cells):

    print("\n" + "=" * 60)
    print("Validating Land Cover Dataset")
    print("=" * 60)

    expected_rows = expected_cells * 2

    print("Expected rows:", expected_rows)
    print("Actual rows:", len(df))

    if len(df) != expected_rows:
        raise ValueError(
            "Unexpected number of rows"
        )

    duplicates = df.duplicated(
        ["cell_id", "epoch"]
    ).sum()

    print("Duplicate keys:", duplicates)

    if duplicates:
        raise ValueError(
            "Duplicate cell_id + epoch"
        )

    valid_rows = df["n_valid_pixels"] > 0

    if not valid_rows.any():
        raise ValueError(
            "No valid CORINE pixels found"
        )

    share_columns = [
        f"{group}_share"
        for group in GROUPS
    ]

    # Every classified cell must sum to one.
    share_sums = df.loc[
        valid_rows,
        share_columns
    ].sum(axis=1)

    if not np.allclose(
        share_sums,
        1.0,
        atol=1e-6
    ):
        raise ValueError(
            "Land cover shares do not sum to 1"
        )

    shares = df.loc[
        valid_rows,
        share_columns
    ]

    if not (
        (shares >= 0).all().all()
        and (shares <= 1).all().all()
    ):
        raise ValueError(
            "Invalid land cover share values"
        )

    if not df["valid_fraction"].between(
        0, 1
    ).all():
        raise ValueError(
            "Invalid valid_fraction values"
        )

    print("All classified shares sum to 1.")

    print("\nValid fraction by epoch:")
    print(
        df.groupby("epoch")[
            "valid_fraction"
        ].describe().to_string()
    )

    print("\nMean land cover shares:")

    print(
        df.groupby("epoch")[
            share_columns
        ].mean().round(4).to_string()
    )

    print("\nDominant land cover groups:")

    print(
        pd.crosstab(
            df["epoch"],
            df["dominant_class"]
        ).to_string()
    )

    print("\nValidation passed.")


def main():

    cells = load_cells()

    outputs = []

    for year, tif_path in CORINE_FILES.items():

        df = process_epoch(
            cells,
            year,
            tif_path
        )

        outputs.append(df)

    result = pd.concat(
        outputs,
        ignore_index=True
    )

    result = result.sort_values(
        ["cell_id", "epoch"]
    ).reset_index(drop=True)

    validate_output(
        result,
        expected_cells=len(cells)
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("WP4 LAND COVER PROCESSING COMPLETE")
    print("=" * 60)

    print("Saved:", OUTPUT_PATH)
    print("Shape:", result.shape)
    print("\nPreview:")

    print(
        result.head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
