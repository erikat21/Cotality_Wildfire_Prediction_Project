"""
WP6 step 5 (exploratory): do other model families beat the logistic fit?

Same question and same protocol as 03_logistic.py -- leave-one-year-out over the open
years 2016-2023, scored on pooled held-out predictions -- but widening the set of learners
beyond logistic regression to see whether a more flexible model fits the cell-month fire
data better.

Reference model (as in 03_logistic.py):
    logit_full      L2 logistic on month + region + all 16 features

New models tried here:
    random_forest   bagged trees, captures interactions/non-linearities
    hist_gbt        histogram gradient-boosted trees (strong tabular default)

Every model sees the same inputs: calendar month, NUTS2 region, and the 16
weather/vegetation/terrain/land-cover features. Trees get region ordinal-encoded and
month/features passed through (no scaling); the logistic models get month+region one-hot
and the features standardised, exactly as in 03_logistic.py.

Scores on the held-out years: ROC AUC, average precision (PR AUC, the fairer one when
positives are rare, ~0.73% here), Brier and log loss. Everything learned (rates, scaling,
encoders, trees) comes from the training years of each fold only.

Label: the panel's `fire` field = WP2 y_S30_any. The full S x x sweep needs
wp2/outputs/labels.parquet; this script runs directly off the WP5 panel so it works with
only the DVC-pulled panel present.

Inputs:  WP5/outputs/cell_month_panel.parquet  (dvc pull)
Outputs (outputs/):
    other_models_cv.csv         pooled held-out scores, one row per model
    other_models_by_year.csv    AUC/AP per held-out year and model
    fig_other_models.png        AP and AUC per model

Run:  python WP6/05_other_models.py
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
PANEL = HERE.parent / "WP5" / "outputs" / "cell_month_panel.parquet"
OUT = HERE / "outputs"
OPEN_YEARS = list(range(2016, 2024))
LABEL = "fire"           # = WP2 y_S30_any
EPS = 1e-6

FEATURES = [
    "tmax", "vpdmax", "tdmean", "ppt_log",
    "ndvi_lag1",
    "elev_mean", "elev_std", "slope_mean", "slope_p90", "aspect_south_share",
    "artificial_share", "agriculture_share", "forest_share",
    "shrub_herbaceous_share", "bare_sparse_share", "wetland_share",
]

REFERENCE = ["logit_full"]
NEW = ["random_forest", "hist_gbt"]
MODELS = REFERENCE + NEW
COLORS = {
    "logit_full": "#eb6834", "random_forest": "#2e9e6b", "hist_gbt": "#9b59b6",
}


# ---------- data ----------

def load():
    df = pd.read_parquet(PANEL)
    df["ppt_log"] = np.log1p(pd.to_numeric(df["ppt"], errors="coerce"))
    df["region"] = df["nuts2"].astype(str)
    df["ym"] = df["year"] * 100 + df["month"]
    df = df[df["year"].isin(OPEN_YEARS)].reset_index(drop=True)
    missing = [c for c in FEATURES + [LABEL, "region", "month", "cell_id"] if c not in df.columns]
    if missing:
        raise SystemExit(f"panel missing columns: {missing}")
    return df


# ---------- ML pipelines ----------

def logit_pipe():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), ["month", "region"]),
        ("num", StandardScaler(), FEATURES),
    ], sparse_threshold=0)
    lr = LogisticRegression(C=1.0, max_iter=3000, tol=1e-8)
    return Pipeline([("pre", pre), ("lr", lr)])


def tree_pipe(estimator):
    # trees: region ordinal-encoded, month + features passed through, no scaling
    pre = ColumnTransformer([
        ("reg", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["region"]),
        ("pass", "passthrough", ["month"] + FEATURES),
    ])
    return Pipeline([("pre", pre), ("clf", estimator)])


def make_model(name):
    if name == "logit_full":
        return logit_pipe()
    if name == "random_forest":
        return tree_pipe(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=50, max_features="sqrt",
            n_jobs=-1, random_state=180))
    if name == "hist_gbt":
        return tree_pipe(HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=400, max_leaf_nodes=31,
            l2_regularization=1.0, min_samples_leaf=100,
            early_stopping=False, random_state=180))
    raise ValueError(name)


def fit_predict(name, train, test, y):
    cols = ["month", "region"] + FEATURES
    m = make_model(name).fit(train[cols], train[y])
    return m.predict_proba(test[cols])[:, 1]


def loyo(df, y, models):
    pred = pd.DataFrame(index=df.index, columns=models, dtype=float)
    for year in OPEN_YEARS:
        tr, te = df[df.year != year], df[df.year == year]
        for m in models:
            pred.loc[te.index, m] = fit_predict(m, tr, te, y)
    return pred


def scores(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return {"auc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "logloss": log_loss(y, p)}


# ---------- main ----------

def main():
    OUT.mkdir(exist_ok=True)
    df = load()
    n_pos = int(df[LABEL].sum())
    print(f"{len(df):,} cell-months, {df.region.nunique()} regions, "
          f"{n_pos:,} positives ({df[LABEL].mean():.3%}), label={LABEL} (= y_S30_any)")
    print(f"models: {MODELS}\n")

    preds, rows = {}, []
    for m in MODELS:
        t0 = time.time()
        p = loyo(df, LABEL, [m])[m]
        preds[m] = p
        s = scores(df[LABEL], p)
        rows.append({"model": m, "kind": "reference" if m in REFERENCE else "new", **s})
        print(f"  {m:14s}  AUC {s['auc']:.4f}  AP {s['ap']:.4f}  "
              f"Brier {s['brier']:.5f}  logloss {s['logloss']:.4f}  ({time.time()-t0:.0f}s)")

    cv = pd.DataFrame(rows)
    cv.to_csv(OUT / "other_models_cv.csv", index=False)

    best_ref = cv[cv.kind == "reference"].sort_values("ap").iloc[-1]
    best_all = cv.sort_values("ap").iloc[-1]
    print(f"\nbest reference by AP : {best_ref.model} (AP {best_ref.ap:.4f}, AUC {best_ref.auc:.4f})")
    print(f"best overall by AP   : {best_all.model} (AP {best_all.ap:.4f}, AUC {best_all.auc:.4f})")
    print(f"AP gain over best reference: {best_all.ap - best_ref.ap:+.4f}")

    # by held-out year
    by_year = []
    for year in OPEN_YEARS:
        i = df.year == year
        for m in MODELS:
            by_year.append({"year": year, "positives": int(df.loc[i, LABEL].sum()), "model": m,
                            "auc": roc_auc_score(df.loc[i, LABEL], preds[m][i]),
                            "ap": average_precision_score(df.loc[i, LABEL], preds[m][i])})
    pd.DataFrame(by_year).to_csv(OUT / "other_models_by_year.csv", index=False)

    # figure: AP and AUC per model
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    order = cv.sort_values("ap").model.tolist()
    for ax, metric, title in [(a1, "ap", "Average precision (PR AUC)"), (a2, "auc", "ROC AUC")]:
        v = cv.set_index("model").loc[order, metric]
        ax.barh(range(len(order)), v, color=[COLORS[m] for m in order])
        ax.set_yticks(range(len(order)), order, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e4e3df", linewidth=0.8); ax.set_axisbelow(True)
        for i, val in enumerate(v):
            ax.text(val, i, f" {val:.3f}", va="center", fontsize=8)
    fig.suptitle(f"WP6: other models vs the logistic fit, held-out years ({LABEL} = y_S30_any)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "fig_other_models.png", dpi=150); plt.close(fig)
    print(f"\nwrote {OUT/'other_models_cv.csv'}, other_models_by_year.csv, fig_other_models.png")


if __name__ == "__main__":
    main()
