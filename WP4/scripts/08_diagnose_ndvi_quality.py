from pathlib import Path
from collections import Counter
import re

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.features import rasterize

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT / "WP4/data/raw/modis/2020_2023"
)

CELLS_PATH = ROOT / "wp2/outputs/cells.gpkg"

NDVI_OUTPUT = (
    ROOT / "WP4/outputs/veg_2020_2023.parquet"
)

OUTPUT_CSV = (
    ROOT / "WP4/outputs/ndvi_missing_quality_2020_2023.csv"
)


def file_month(path):
    match = re.search(r"doy(\d{7})", path.name)
    if match is None:
        return None
    return pd.to_datetime(
        match.group(1), format="%Y%j"
    ).strftime("%Y-%m")


def main():
    df = pd.read_parquet(NDVI_OUTPUT)
    missing = df[df["ndvi_lag1"].isna()].copy()

    cells = gpd.read_file(CELLS_PATH)
    cells = cells[cells["in_mask"] == 1].copy()

    quality_files = {
        file_month(path): path
        for path in INPUT_DIR.rglob("*.tif")
        if "pixel_reliability" in path.name.lower()
    }

    records = []

    for (year, month), subset in missing.groupby(
        ["source_year", "source_month"]
    ):
        key = f"{year:04d}-{month:02d}"
        path = quality_files.get(key)

        if path is None:
            print("Missing quality file:", key)
            continue

        with rasterio.open(path) as src:
            quality = src.read(1, masked=True)

            selected = cells[
                cells["cell_id"].isin(
                    subset["cell_id"]
                )
            ].to_crs(src.crs)

            shapes = [
                (geom, int(cid))
                for geom, cid in zip(
                    selected.geometry,
                    selected["cell_id"]
                )
            ]

            labels = rasterize(
                shapes,
                out_shape=src.shape,
                transform=src.transform,
                fill=-1,
                dtype="int32",
                all_touched=False
            )

            for cid in subset["cell_id"]:
                mask = labels == cid

                values = quality.data[mask]
                values = values[
                    ~np.ma.getmaskarray(quality)[mask]
                ]

                counts = Counter(values.tolist())

                records.append({
                    "cell_id": cid,
                    "year": year,
                    "month": month,
                    "code_0_good": counts.get(0, 0),
                    "code_1_marginal": counts.get(1, 0),
                    "code_2_snow_ice": counts.get(2, 0),
                    "code_3_cloud": counts.get(3, 0),
                    "quality_nodata": (
                        mask.sum() - len(values)
                    )
                })

    result = pd.DataFrame(records)

    OUTPUT_CSV.parent.mkdir(
        parents=True, exist_ok=True
    )

    result.to_csv(OUTPUT_CSV, index=False)

    print("\nIndividual missing records:")
    print(result.to_string(index=False))

    print("\nTotal quality codes:")
    print(
        result[
            [
                "code_0_good",
                "code_1_marginal",
                "code_2_snow_ice",
                "code_3_cloud",
                "quality_nodata"
            ]
        ].sum()
    )

    print("\nSaved:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
