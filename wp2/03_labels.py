"""
WP2 step 3: turn fire_cell into the label table (`labels`), one row per cell-month that
had at least one fire of >= 30 ha. WP5 adds every other cell-month as 0.

For each fire-size cutoff S and each cell-month:
    burn_frac_S<S>  share of the cell burnt by fires that STARTED that month and are >= S ha.
                    Overlapping perimeters are merged first, so the same ground is never counted twice.
    n_fires_S<S>    how many such fires
    y_S<S>_<x>      1 if burn_frac >= x  (x = any overlap, 1%, 2%, 5%, 10%), else 0
A row where n_fires_S<S> == 0 is a "small-fire-only" cell-month for that S: it had a fire
between 30 ha and S. Keep it (label 0) or drop it in the fit: that is the section 5 check.

Inputs:  outputs/fire_cell.gpkg  (from 02_fire_cell.py)
Outputs: outputs/labels.parquet
         outputs/label_counts.csv   positives per year for every S and x (to choose S and x on counts)

Run:  python 03_labels.py
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
SIZES_HA = [30, 100, 383]
THRESHOLDS = {"any": 0.0, "x01": 0.01, "x02": 0.02, "x05": 0.05, "x10": 0.10}
KEYS = ["cell_id", "year", "month"]


def merged_burn(pieces: gpd.GeoDataFrame) -> pd.DataFrame:
    """Burnt share per cell-month, merging overlapping fire pieces before measuring."""
    g = pieces.dissolve(by=KEYS, aggfunc={"fire_id": "nunique", "cell_area_ha": "first",
                                          "burnt_ha_in_cell": "sum"})
    g["burnt_ha_merged"] = g.area / 10_000
    out = pd.DataFrame(g.drop(columns="geometry")).rename(columns={"fire_id": "n_fires",
                                                                   "burnt_ha_in_cell": "burnt_ha_summed"})
    out["burn_frac"] = (out.burnt_ha_merged / out.cell_area_ha).clip(upper=1.0)
    return out


def main():
    pieces = gpd.read_file(OUT / "fire_cell.gpkg", layer="fire_cell")
    base = pieces.groupby(KEYS).agg(in_mask=("in_mask", "first"),
                                    max_fire_ha=("fire_area_ha", "max")).reset_index()

    labels = base
    overlap_report = []
    for S in SIZES_HA:
        b = merged_burn(pieces[pieces.fire_area_ha >= S])
        double = (b.burnt_ha_summed - b.burnt_ha_merged) > 0.01
        overlap_report.append((S, int(double.sum())))
        b = b[["n_fires", "burn_frac"]].add_suffix(f"_S{S}").reset_index()
        labels = labels.merge(b, on=KEYS, how="left")
        labels[f"n_fires_S{S}"] = labels[f"n_fires_S{S}"].fillna(0).astype(int)
        labels[f"burn_frac_S{S}"] = labels[f"burn_frac_S{S}"].fillna(0.0)
        for name, x in THRESHOLDS.items():
            frac = labels[f"burn_frac_S{S}"]
            labels[f"y_S{S}_{name}"] = ((frac > 0) if x == 0 else (frac >= x)).astype(int)

    labels = labels.sort_values(KEYS).reset_index(drop=True)
    labels.to_parquet(OUT / "labels.parquet", index=False)

    # Counts of positive cell-months (in the mask) per year, for every S and x
    m = labels[labels.in_mask == 1]
    ycols = [c for c in labels.columns if c.startswith("y_S")]
    counts = m.groupby("year")[ycols].sum()
    counts.loc["total"] = counts.sum()
    counts.to_csv(OUT / "label_counts.csv")

    print(f"labels rows (cell-months with any fire >= 30 ha): {len(labels):,}, of which in the mask: {len(m):,}")
    for S, n in overlap_report:
        print(f"S={S}: {n} cell-months where overlapping perimeters were merged")
    print("\npositive cell-months in the mask, all years:")
    print(counts.loc["total"].to_string())


if __name__ == "__main__":
    main()
