"""
WP6 step 2: how the fire rate changes with each feature (descriptive, open years only).

For every feature, cell-months are cut into about 10 equal-sized bins and the fire rate is
measured in each bin, for S = 30, 100 and 383 ha (x = any). For the monthly features
(weather, NDVI) the same is done on the anomaly: the value minus that cell's average for the
same calendar month. The anomaly shows the effect beyond season and place, which is
question 1 of the brief.

Inputs:  WP5 panel + WP2 labels (see wp6_common.py)
Outputs (outputs/):
    dependence_bins.csv      feature, kind (raw/anomaly), bin, mean value, rate for each S
    fig_dependence.png       one panel per feature, fire rate per bin (raw values)
    fig_dependence_anom.png  same for the anomalies of the monthly features

Run:  python 02_dependence.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wp6_common import NICE, OUT, SIZES, available, load_panel

COLORS = {30: "#2a78d6", 100: "#eb6834", 383: "#1baf7a"}
MONTHLY = ["tmax", "vpdmax", "tdmean", "ppt_log", "ndvi_lag1"]


def binned_rates(df, col, kind):
    # tied values stay in one bin, so a share that is mostly 0 gets fewer than 10 bins
    b = pd.qcut(df[col], 10, labels=False, duplicates="drop")
    g = df.groupby(b)
    out = pd.DataFrame({"feature": col, "kind": kind, "bin": range(1, g.ngroups + 1),
                        "n_cell_months": g.size().values,
                        "value_mean": g[col].mean().values,
                        "value_min": g[col].min().values, "value_max": g[col].max().values})
    for S in SIZES:
        out[f"rate_per_1000_S{S}"] = 1000 * g[f"y_S{S}_any"].mean().values
    return out


def plot(tab, feats, path, title):
    n = len(feats)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.6 * nrow), squeeze=False)
    for ax, f in zip(axes.flat, feats):
        t = tab[tab.feature == f]
        for S in SIZES:
            ax.plot(t.value_mean, t[f"rate_per_1000_S{S}"], color=COLORS[S], linewidth=2,
                    marker="o", markersize=3, label=f"S = {S} ha")
        ax.set_title(NICE.get(f, f) if not f.endswith("_anom") else NICE.get(f[:-5], f) + ", anomaly",
                     fontsize=9, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.tick_params(labelsize=8)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].set_ylabel("fires per 1,000 cell-months", fontsize=8)
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    df = load_panel()
    feats = available(df)
    print(f"{len(df):,} cell-months, features: {feats}")

    tabs = [binned_rates(df, f, "raw") for f in feats]
    anom = []
    for f in [m for m in MONTHLY if m in feats]:
        name = f + "_anom"
        df[name] = df[f] - df.groupby(["cell_id", "month"])[f].transform("mean")
        tabs.append(binned_rates(df, name, "anomaly"))
        anom.append(name)
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(OUT / "dependence_bins.csv", index=False)

    plot(tab, feats, OUT / "fig_dependence.png", "Fire rate across about 10 equal-sized bins of each feature")
    if anom:
        plot(tab, anom, OUT / "fig_dependence_anom.png",
             "Fire rate vs anomaly (value minus the cell's usual value for that month)")

    # Short summary: ratio of the top-bin rate to the bottom-bin rate
    s = tab.groupby("feature").apply(lambda t: pd.Series({
        "rate_low_bin": t.rate_per_1000_S383.iloc[0], "rate_high_bin": t.rate_per_1000_S383.iloc[-1]}),
        include_groups=False)
    print("\nS = 383 ha, fires per 1,000 cell-months in the lowest and highest bin:")
    print(s.round(2).to_string())


if __name__ == "__main__":
    main()
