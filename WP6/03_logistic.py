"""
WP6 step 3: does a logistic model beat simple baselines on years it has not seen?

Leave-one-year-out over the open years 2016-2023: each year is predicted by models trained on
the other seven. Everything learned from data (rates, feature scaling, coefficients) comes
from the training years of that fold only (brief rule 2). Test years are never read here.

Four models, from simplest to full:
    season          fire rate for each calendar month
    season+place    each cell's own rate for that calendar month, shrunk towards its NUTS2
                    region's rate for that month (cells with little history borrow from the region)
    logit_base      logistic regression on month + NUTS2 region only
    logit_full      logit_base + all features (weather, NDVI, terrain, land cover)
logit_full vs the best of the others answers "do the features add anything beyond season and
place". Scores on the pooled held-out predictions: ROC AUC, average precision (PR AUC, the
fairer one when positives are rare), Brier score and log loss. 95% intervals come from a
bootstrap over year-months (brief rule 5), not over rows.

This is run for every label definition (S x x). The bootstrap, the small-fire check and the
coefficient table are done for the main label (--main, default S = 383 ha, x = any).

Inputs:  WP5 panel + WP2 labels (see wp6_common.py)
Outputs (outputs/):
    cv_sweep.csv            one row per label x model: pooled held-out scores
    cv_by_year.csv          main label: AUC and AP for each held-out year and model
    cv_bootstrap.csv        main label and S sweep at x = any: scores with 95% intervals,
                            and logit_full minus the best baseline
    small_fire_check.csv    main label refit without small-fire-only cell-months
    coefficients.csv        logit_full on all open years: odds ratio per 1 SD of each feature,
                            plus the range across the 8 leave-one-year-out fits
    oof_main.parquet        held-out predictions for the main label (for plots)
    fig_sweep.png           AP of each model for every label definition
    fig_coefficients.png    odds ratios of the features (main label)
    fig_calibration.png     predicted vs observed rate, main label

Run:  python 03_logistic.py                     # all 15 labels, takes a while
      python 03_logistic.py --only-any          # x = any only (3 labels), quick check
      python 03_logistic.py --main y_S100_any
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from wp6_common import NICE, OPEN_YEARS, OUT, SIZES, X_LABEL, XS, available, label_cols, load_panel

MODELS = ["season", "season+place", "logit_base", "logit_full"]
COLORS = {"season": "#86b6ef", "season+place": "#2a78d6", "logit_base": "#eda100", "logit_full": "#eb6834"}
N_BOOT = 300
EPS = 1e-6


# ---------- models ----------

def rate_season(train, test, y):
    rate = train.groupby("month")[y].mean()
    return test.month.map(rate).fillna(train[y].mean()).to_numpy()


def rate_season_place(train, test, y, k_region=200, k_cell=20):
    """Cell x month rate, shrunk to region x month, which is shrunk to month (pseudo-counts k)."""
    p_month = train.groupby("month")[y].mean()
    r = train.groupby(["region", "month"])[y].agg(["sum", "count"]).reset_index()
    r["p_region"] = (r["sum"] + k_region * r.month.map(p_month)) / (r["count"] + k_region)
    c = train.groupby(["cell_id", "month"])[y].agg(["sum", "count"]).reset_index()
    c = c.merge(r[["region", "month", "p_region"]].merge(
        train[["cell_id", "region"]].drop_duplicates(), on="region"), on=["cell_id", "month"])
    c["p"] = (c["sum"] + k_cell * c.p_region) / (c["count"] + k_cell)
    out = test[["cell_id", "month", "region"]].merge(c[["cell_id", "month", "p"]], on=["cell_id", "month"], how="left")
    out = out.merge(r[["region", "month", "p_region"]], on=["region", "month"], how="left")
    return out.p.fillna(out.p_region).fillna(test.month.map(p_month)).to_numpy()


def logit(features):
    cats = ["month", "region"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cats),
        ("num", StandardScaler(), features),   # fitted on the training years of each fold only
    ], sparse_threshold=0)
    # FIX P0-2: Tighten solver tolerance after sensitivity checks showed
    # substantially different coefficients under the default tolerance.
    # Keep C and the solver unchanged for a controlled comparison.
    return Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(C=1.0, max_iter=3000, tol=1e-8)),
    ])


def fit_predict(model, train, test, y, feats):
    if model == "season":
        return rate_season(train, test, y)

    if model == "season+place":
        return rate_season_place(train, test, y)

    f = feats if model == "logit_full" else []

    m = logit(f).fit(
        train[["month", "region"] + f],
        train[y]
    )

    # FIX P1-2: Check logistic regression convergence.
    # Reaching max_iter may indicate that the solver has not converged.
    lr = m.named_steps["lr"]
    n_iter = int(lr.n_iter_[0])

    if n_iter >= lr.max_iter:
        print(
            f"WARNING: {model} for {y} reached "
            f"max_iter={lr.max_iter}. "
            "Check convergence before interpreting coefficients."
        )

    return m.predict_proba(
        test[["month", "region"] + f]
    )[:, 1]



def loyo(df, y, feats, models=MODELS):
    """Leave-one-year-out predictions for every row, one column per model."""
    pred = pd.DataFrame(index=df.index, columns=models, dtype=float)
    for year in OPEN_YEARS:
        tr, te = df[df.year != year], df[df.year == year]
        for m in models:
            pred.loc[te.index, m] = fit_predict(m, tr, te, y, feats)
    return pred


# ---------- scores ----------

def scores(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return {"auc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "logloss": log_loss(y, p)}


def weighted_auc_ap(score_id, pos_w, neg_w):
    """AUC and average precision from weighted positives/negatives per distinct score
    (score_id ranks distinct scores, 0 = lowest). Ties are handled like sklearn."""
    P = np.bincount(score_id, pos_w)[::-1]          # highest score first
    N = np.bincount(score_id, neg_w)[::-1]
    tp, fp = np.cumsum(P), np.cumsum(N)
    auc = np.sum(P * (N.sum() - fp + 0.5 * N)) / (P.sum() * N.sum())
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    ap = np.sum(P * prec) / P.sum()
    return auc, ap


def bootstrap(df, y, pred, models, rng, n=N_BOOT):
    """Resample whole year-months with replacement; returns one row per draw and model.
    Each draw is done with weights (how often each year-month was drawn), which is the same as
    stacking the drawn year-months but much faster."""
    block = pd.factorize(df.ym)[0]
    n_blocks = block.max() + 1
    yv = df[y].to_numpy()
    ids = {m: np.unique(pred[m].to_numpy(), return_inverse=True)[1] for m in models}
    rows = []
    for b in range(n):
        w = np.bincount(rng.integers(0, n_blocks, n_blocks), minlength=n_blocks)[block].astype(float)
        if (w * yv).sum() == 0:
            continue
        for m in models:
            auc, ap = weighted_auc_ap(ids[m], w * yv, w * (1 - yv))
            rows.append({"draw": b, "model": m, "auc": auc, "ap": ap})
    return pd.DataFrame(rows)


def summarise_boot(df, y, pred, label, rng):
    """Summarise bootstrap scores and paired model improvements."""

    bs = bootstrap(df, y, pred, MODELS, rng)
    point = {m: scores(df[y], pred[m]) for m in MODELS}

    rows = []

    # Original: bootstrap confidence intervals for each model.
    for m in MODELS:
        b = bs[bs.model == m]

        rows.append({
            "label": label,
            "model": m,
            "auc": point[m]["auc"],
            "auc_lo": b.auc.quantile(.025),
            "auc_hi": b.auc.quantile(.975),
            "ap": point[m]["ap"],
            "ap_lo": b.ap.quantile(.025),
            "ap_hi": b.ap.quantile(.975),
            "auc_share_of_draws_better": np.nan,
            "ap_share_of_draws_better": np.nan,
        })

    # Original: choose the strongest baseline using point-estimate AP.
    best = max(
        ["season", "season+place", "logit_base"],
        key=lambda m: point[m]["ap"]
    )

    w = bs.pivot(
        index="draw",
        columns="model",
        values=["auc", "ap"]
    )

    # FIX P1-1:
    # Calculate paired bootstrap improvements separately for AUC and AP.
    # Previously, groupby().first() could report the AUC win rate
    # as though it were also the AP win rate.
    auc_diff = w["auc"]["logit_full"] - w["auc"][best]
    ap_diff = w["ap"]["logit_full"] - w["ap"][best]

    rows.append({
        "label": label,
        "model": f"logit_full minus {best}",

        "auc": point["logit_full"]["auc"] - point[best]["auc"],
        "auc_lo": auc_diff.quantile(.025),
        "auc_hi": auc_diff.quantile(.975),

        "ap": point["logit_full"]["ap"] - point[best]["ap"],
        "ap_lo": ap_diff.quantile(.025),
        "ap_hi": ap_diff.quantile(.975),

        # FIX P1-1: Each metric gets its own win probability.
        "auc_share_of_draws_better": (auc_diff > 0).mean(),
        "ap_share_of_draws_better": (ap_diff > 0).mean(),
    })

    return pd.DataFrame(rows)



# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="y_S383_any", help="label for the bootstrap, coefficients and checks")
    ap.add_argument("--only-any", action="store_true", help="sweep x = any only")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(180)

    df = load_panel().reset_index(drop=True)
    feats = available(df)
    print(f"{len(df):,} cell-months, {df.region.nunique()} regions, {len(feats)} features: {feats}")
    labels = [c for c in label_cols() if c.endswith("_any")] if args.only_any else label_cols()
    if args.main not in labels:
        labels.append(args.main)

    # 1. Sweep over label definitions
    sweep, preds = [], {}
    for y in labels:
        t0 = time.time()
        pred = loyo(df, y, feats)
        preds[y] = pred
        for m in MODELS:
            sweep.append({"label": y, "S_ha": int(y.split("_")[1][1:]), "x": X_LABEL[y.split("_")[2]],
                          "positives": int(df[y].sum()), "model": m, **scores(df[y], pred[m])})
        s = pd.DataFrame(sweep)
        s = s[s.label == y].set_index("model")
        print(f"{y}: {int(df[y].sum()):,} positives  AP " +
              "  ".join(f"{m} {s.loc[m, 'ap']:.4f}" for m in MODELS) + f"  ({time.time() - t0:.0f}s)")
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(OUT / "cv_sweep.csv", index=False)

    # 2. Main label: by year, bootstrap, held-out predictions
    y = args.main
    pred = preds[y]
    by_year = []
    for year in OPEN_YEARS:
        i = df.year == year
        for m in MODELS:
            by_year.append({"year": year, "positives": int(df.loc[i, y].sum()), "model": m,
                            "auc": roc_auc_score(df.loc[i, y], pred.loc[i, m]),
                            "ap": average_precision_score(df.loc[i, y], pred.loc[i, m])})
    pd.DataFrame(by_year).to_csv(OUT / "cv_by_year.csv", index=False)
    pd.concat([df[["cell_id", "year", "month", y]], pred], axis=1).to_parquet(OUT / "oof_main.parquet", index=False)

    boot_labels = [y] + [f"y_S{S}_any" for S in SIZES if f"y_S{S}_any" != y]
    boot = pd.concat([summarise_boot(df, lab, preds[lab], lab, rng) for lab in boot_labels])
    boot.to_csv(OUT / "cv_bootstrap.csv", index=False)
    print("\nbootstrap (95% over year-months):")
    print(boot.round(4).to_string(index=False))

    # 3. Small-fire check: drop cell-months that only had fires smaller than S
    S = int(y.split("_")[1][1:])
    keep = ~((df.n_fires_S30 > 0) & (df[f"n_fires_S{S}"] == 0))
    sub = df[keep]
    p_sub = loyo(sub, y, feats, ["season+place", "logit_full"])
    check = pd.DataFrame([{"rows": "all", "model": m, **scores(df[y], pred[m])} for m in ["season+place", "logit_full"]]
                         + [{"rows": "small-fire-only dropped", "model": m, **scores(sub[y], p_sub[m])}
                            for m in ["season+place", "logit_full"]])
    check["n_dropped"] = int((~keep).sum())
    check.to_csv(OUT / "small_fire_check.csv", index=False)
    print(f"\nsmall-fire check ({(~keep).sum():,} rows dropped):")
    print(check.round(4).to_string(index=False))

    # 4. Coefficients: full fit + range over the leave-one-year-out fits
    def coefs(data):
        # FIX P1-3: Diagnose convergence for the coefficient-analysis fits.
        m = logit(feats).fit(
            data[["month", "region"] + feats],
            data[y]
        )

        lr = m.named_steps["lr"]
        n_iter = int(lr.n_iter_[0])

        print(
            f"Coefficient fit: {len(data):,} rows, "
            f"iterations={n_iter}/{lr.max_iter}"
        )

        if n_iter >= lr.max_iter:
            print(
                "WARNING: Coefficient fit reached max_iter. "
                "Interpret odds ratios with caution."
            )

        names = m.named_steps["pre"].get_feature_names_out()
        c = pd.Series(lr.coef_[0], index=names)

        return c[[f"num__{f}" for f in feats]].set_axis(feats)

    full = coefs(df)
    folds = pd.DataFrame({yr: coefs(df[df.year != yr]) for yr in OPEN_YEARS})
    coef = pd.DataFrame({"feature": feats, "coef_per_sd": full.values,
                         "odds_ratio_per_sd": np.exp(full.values),
                         "or_min_over_folds": np.exp(folds.min(axis=1).values),
                         "or_max_over_folds": np.exp(folds.max(axis=1).values)})
    coef["same_sign_all_folds"] = (np.sign(folds).nunique(axis=1) == 1).values
    coef.to_csv(OUT / "coefficients.csv", index=False)
    print("\ncoefficients (odds ratio per 1 SD, with month and region held fixed):")
    print(coef.round(3).to_string(index=False))

    # ---------- figures ----------
    lab_order = [f"y_S{S}_{x}" for S in SIZES for x in XS if f"y_S{S}_{x}" in labels]
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.2
    for i, m in enumerate(MODELS):
        v = sweep[sweep.model == m].set_index("label").loc[lab_order, "ap"]
        ax.bar(np.arange(len(lab_order)) + (i - 1.5) * w, v, width=w - 0.02, color=COLORS[m], label=m)
    ax.set_xticks(range(len(lab_order)),
                  [f"S{l.split('_')[1][1:]}\n{X_LABEL[l.split('_')[2]]}" for l in lab_order], fontsize=8)
    ax.set_ylabel("average precision (held-out years)")
    ax.set_title("Each model on every label definition (higher is better)", loc="left")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT / "fig_sweep.png", dpi=150); plt.close(fig)

    c = coef.sort_values("coef_per_sd")
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(c) + 1.2))
    yy = np.arange(len(c))
    ax.hlines(yy, c.or_min_over_folds, c.or_max_over_folds, color="#86b6ef", linewidth=4)
    ax.plot(c.odds_ratio_per_sd, yy, "o", color="#184f95", markersize=6)
    ax.axvline(1, color="#52514e", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(yy, [NICE.get(f, f) for f in c.feature], fontsize=8)
    ax.set_xlabel("odds ratio per 1 SD (dot = all years, bar = range over the 8 folds)")
    ax.set_title(f"What the full model learned ({y})", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_coefficients.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for m in ["season+place", "logit_full"]:
        q = pd.qcut(pred[m].rank(method="first"), 20, labels=False)
        g = pd.DataFrame({"p": pred[m], "y": df[y]}).groupby(q).mean()
        ax.plot(1000 * g.p, 1000 * g.y, "o-", color=COLORS[m], markersize=4, label=m)
    lim = 1000 * max(g.p.max(), g.y.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#52514e", linewidth=1, linestyle="--")
    ax.set_xlabel("predicted, per 1,000 cell-months"); ax.set_ylabel("observed, per 1,000 cell-months")
    ax.set_title(f"Calibration on held-out years ({y})", loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_calibration.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
