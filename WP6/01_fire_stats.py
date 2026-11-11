"""
WP6 step 1: fire-only statistics and the label sweep (no weather needed).

How many positive cell-months each label definition gives (fire size S x burnt share x),
how they spread over years, calendar months and regions, and how many "small-fire-only"
cell-months each S turns into zeros. This is what S and x are chosen on (brief section 5).

Only the open years (May 2016 - Dec 2023) are used unless --locked is passed, so the
choice of S and x never looks at the test years.

Inputs:
    ../wp2/outputs/cells.parquet      in-mask cells (denominator)
    ../wp2/outputs/labels.parquet     cell-months with a fire >= 30 ha, y_S{S}_{x} columns
Outputs (outputs/):
    label_sweep.csv              one row per (S, x): positives, rate, per-year spread, summer share
    label_by_year.csv            positives per year for every S and x
    label_by_month.csv           positives per calendar month for every S and x
    label_by_region.csv          positives and rate per NUTS2 region (x = any)
    fig_label_sweep.png          positives per S and x
    fig_seasonality.png          positives by calendar month, one line per S (x = any)
    fig_by_year.png              positives by year, one line per S (x = any)
    fig_fire_map.png             share of months each cell had a fire (S = 383, x = any)

Run:  python 01_fire_stats.py
      python 01_fire_stats.py --locked     # all years, test step only
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent
WP2_OUT = HERE.parent / "wp2" / "outputs"
OUT = HERE / "outputs"
SIZES = [30, 100, 383]
XS = ["any", "x01", "x02", "x05", "x10"]
X_LABEL = {"any": "any", "x01": "1%", "x02": "2%", "x05": "5%", "x10": "10%"}
LOCKED_YEARS = {2024, 2025, 2026}
COLORS = {30: "#2a78d6", 100: "#eb6834", 383: "#1baf7a"}   # one colour per S, same in every plot
MONTH_NAMES = list("JFMAMJJASOND")


def panel_months(locked: bool) -> pd.DataFrame:
    """Same months as WP5: May 2016 to Dec 2023 (to Aug 2026 with --locked)."""
    rows = []
    for year in range(2016, 2027):
        if year in LOCKED_YEARS and not locked:
            continue
        for month in range(5 if year == 2016 else 1, (8 if year == 2026 else 12) + 1):
            rows.append((year, month))
    return pd.DataFrame(rows, columns=["year", "month"])


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", action="store_true", help="include 2024-2026 (test step only)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    cells = pd.read_parquet(WP2_OUT / "cells.parquet")
    cells = cells[cells.in_mask == 1][["cell_id", "lon", "lat", "country", "nuts2"]]
    months = panel_months(args.locked)
    n_cells, n_months = len(cells), len(months)
    n_cell_months = n_cells * n_months

    lab = pd.read_parquet(WP2_OUT / "labels.parquet")
    lab = lab[lab.in_mask == 1].merge(months, on=["year", "month"])   # in-mask cells, panel months only
    ycols = [f"y_S{S}_{x}" for S in SIZES for x in XS]
    print(f"{n_cells:,} cells x {n_months} months = {n_cell_months:,} cell-months "
          f"({months.year.min()}-{months.year.max()}); {len(lab):,} have a fire >= 30 ha")

    # 1. The sweep table
    by_year = lab.groupby("year")[ycols].sum().reindex(sorted(months.year.unique()), fill_value=0)
    by_month = lab.groupby("month")[ycols].sum().reindex(range(1, 13), fill_value=0)
    full_years = [y for y in by_year.index if (months.year == y).sum() == 12]
    rows = []
    for S in SIZES:
        small_only = int(((lab.n_fires_S30 > 0) & (lab[f"n_fires_S{S}"] == 0)).sum())
        for x in XS:
            c = f"y_S{S}_{x}"
            n = int(lab[c].sum())
            per_year = by_year.loc[full_years, c]
            rows.append({
                "S_ha": S, "x": X_LABEL[x], "positives": n,
                "rate_per_1000": round(1000 * n / n_cell_months, 3),
                "cells_ever_positive": int(lab.loc[lab[c] == 1, "cell_id"].nunique()),
                "min_per_full_year": int(per_year.min()), "max_per_full_year": int(per_year.max()),
                "share_jun_sep": round(by_month.loc[6:9, c].sum() / max(n, 1), 3),
                "small_fire_only_zeros": small_only,
            })
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "label_sweep.csv", index=False)
    by_year.to_csv(OUT / "label_by_year.csv")
    by_month.to_csv(OUT / "label_by_month.csv")

    # 2. By region (x = any)
    cells["country"] = cells.nuts2.str[:2]   # a few border cells have country and nuts2 from different sides
    lab_r = lab.merge(cells[["cell_id", "country", "nuts2"]], on="cell_id")
    region_cells = cells.groupby(["country", "nuts2"]).size().rename("n_cells")
    region = lab_r.groupby(["country", "nuts2"])[[f"y_S{S}_any" for S in SIZES]].sum()
    region = region_cells.to_frame().join(region).fillna(0)
    for S in SIZES:
        region[f"rate_per_1000_S{S}"] = (1000 * region[f"y_S{S}_any"] / (region.n_cells * n_months)).round(3)
    region = region.sort_values("rate_per_1000_S383", ascending=False)
    region.to_csv(OUT / "label_by_region.csv")

    # 3. Figures
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.26
    for i, S in enumerate(SIZES):
        s = sweep[sweep.S_ha == S]
        ax.bar([j + (i - 1) * w for j in range(len(XS))], s.positives, width=w - 0.03,
               color=COLORS[S], label=f"S = {S} ha")
    ax.set_xticks(range(len(XS)), [f"x = {X_LABEL[x]}" for x in XS])
    ax.set_ylabel("positive cell-months")
    ax.set_title("Positives for each label definition", loc="left")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout(); fig.savefig(OUT / "fig_label_sweep.png", dpi=150); plt.close(fig)

    for name, table, xvals, xticks, title in [
        ("fig_seasonality.png", by_month, range(1, 13), MONTH_NAMES, "Positives by calendar month (x = any)"),
        ("fig_by_year.png", by_year, by_year.index, [str(y) for y in by_year.index], "Positives by year (x = any)"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for S in SIZES:
            ax.plot(list(xvals), table[f"y_S{S}_any"], color=COLORS[S], linewidth=2,
                    marker="o", markersize=4, label=f"S = {S} ha")
        ax.set_xticks(list(xvals), xticks)
        ax.set_ylabel("positive cell-months")
        ax.set_title(title, loc="left")
        ax.legend(frameon=False)
        style(ax)
        fig.tight_layout(); fig.savefig(OUT / name, dpi=150); plt.close(fig)

    # Map: one pixel per grid cell (row 0 = north), grey = in mask but never positive
    freq = lab.groupby("cell_id")["y_S383_any"].sum()
    rows, cols = cells.cell_id // 143, cells.cell_id % 143
    grid = np.full((82, 143), np.nan)
    grid[rows, cols] = cells.cell_id.map(freq).fillna(0).to_numpy()
    blues = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.3, 1, 256)))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.imshow(np.where(grid == 0, 1, np.nan), cmap=ListedColormap(["#e4e3df"]), aspect=1 / 0.77)
    im = ax.imshow(np.where(grid > 0, grid, np.nan), cmap=blues, vmin=1, aspect=1 / 0.77)
    fig.colorbar(im, ax=ax, shrink=0.7, label="months with a fire (S = 383 ha, x = any)")
    ax.set_title("Where large fires happened (grey = never)", loc="left")
    ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "fig_fire_map.png", dpi=150); plt.close(fig)

    print("\nlabel sweep:")
    print(sweep.to_string(index=False))
    print("\ntop regions by S=383 rate:")
    print(region.head(8).to_string())


if __name__ == "__main__":
    main()
