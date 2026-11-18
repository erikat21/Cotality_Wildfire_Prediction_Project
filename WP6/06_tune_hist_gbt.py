"""
WP6 step 6 (exploratory): tune the gradient-boosted trees (hist_gbt).

05_other_models.py ran hist_gbt with hand-picked defaults. This script searches its
hyperparameters properly and checks whether tuning actually improves the held-out score.

Two stages:
  1. SEARCH -- RandomizedSearchCV over the hist_gbt hyperparameters, scored by average
     precision (AP, the rare-event metric), using GroupKFold grouped by YEAR so no fold
     ever trains and validates on the same year. Number of boosting rounds is handled by
     early stopping, so it is not in the grid. Prints the best config and the top configs.
  2. CONFIRM -- re-score the tuned config AND the current default config under the SAME
     leave-one-year-out protocol as 05 (train on 7 years, predict the held-out year,
     pool the predictions), so the comparison is apples-to-apples.

Label: the panel's `fire` field = WP2 y_S30_any (loads straight off the WP5 panel, so it
works without wp2/outputs/labels.parquet).

Honesty note: the search uses GroupKFold over ALL open years, so the tuned config has been
informed by every year. The stage-2 LOYO number is therefore a mildly optimistic estimate
of tuned performance; a fully nested search (re-tuning inside each outer fold) would remove
this, at ~8x the cost.

Inputs:  WP5/outputs/cell_month_panel.parquet  (dvc pull)
Outputs (outputs/tune_hist_gbt/):
    search_results.csv      every searched config with mean/std CV AP
    tuned_vs_default.csv     pooled LOYO AUC/AP for default vs tuned hist_gbt

Run:  python WP6/06_tune_hist_gbt.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
PANEL = HERE.parent / "WP5" / "outputs" / "cell_month_panel.parquet"
OUT = HERE / "outputs" / "tune_hist_gbt"
OPEN_YEARS = list(range(2016, 2024))
LABEL = "fire"            # = WP2 y_S30_any
EPS = 1e-6
SEED = 180

FEATURES = [
    "tmax", "vpdmax", "tdmean", "ppt_log",
    "ndvi_lag1",
    "elev_mean", "elev_std", "slope_mean", "slope_p90", "aspect_south_share",
    "artificial_share", "agriculture_share", "forest_share",
    "shrub_herbaceous_share", "bare_sparse_share", "wetland_share",
]
COLS = ["month", "region"] + FEATURES

# 05's current hist_gbt settings -- the "default" we are trying to beat.
DEFAULT = dict(learning_rate=0.08, max_iter=400, max_leaf_nodes=31,
               l2_regularization=1.0, min_samples_leaf=100,
               early_stopping=False, random_state=SEED)


def load():
    df = pd.read_parquet(PANEL)
    df["ppt_log"] = np.log1p(pd.to_numeric(df["ppt"], errors="coerce"))
    df["region"] = df["nuts2"].astype(str)
    df = df[df["year"].isin(OPEN_YEARS)].reset_index(drop=True)
    miss = [c for c in COLS + [LABEL, "year"] if c not in df.columns]
    if miss:
        raise SystemExit(f"panel missing columns: {miss}")
    return df


def pipe(estimator):
    pre = ColumnTransformer([
        ("reg", OneHotEncoder(handle_unknown="ignore"), ["region"]),
        ("pass", "passthrough", ["month"] + FEATURES),
    ], sparse_threshold=0)
    return Pipeline([("pre", pre), ("clf", estimator)])


def loyo_scores(df, params):
    """Pooled leave-one-year-out AUC/AP for one hist_gbt config."""
    pred = np.empty(len(df))
    for year in OPEN_YEARS:
        tr, te = df[df.year != year], df[df.year == year]
        m = pipe(HistGradientBoostingClassifier(**params)).fit(tr[COLS], tr[LABEL])
        pred[te.index.to_numpy()] = m.predict_proba(te[COLS])[:, 1]
    p = np.clip(pred, EPS, 1 - EPS)
    return {"auc": roc_auc_score(df[LABEL], p), "ap": average_precision_score(df[LABEL], p)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"{len(df):,} cell-months, {int(df[LABEL].sum()):,} positives "
          f"({df[LABEL].mean():.3%}), label={LABEL} (= y_S30_any)\n")

    # ---------- stage 1: search ----------
    # Early stopping decides the number of trees, so max_iter is just a high cap.
    base = HistGradientBoostingClassifier(
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, max_iter=1000, random_state=SEED)

    grid = {
        "clf__learning_rate": [0.02, 0.05, 0.1, 0.2],
        "clf__max_leaf_nodes": [15, 31, 63, 127],
        "clf__min_samples_leaf": [20, 50, 100, 200],
        "clf__l2_regularization": [0.0, 1.0, 10.0],
        "clf__max_features": [0.5, 0.7, 1.0],
        "clf__class_weight": [None, "balanced"],
    }

    search = RandomizedSearchCV(
        pipe(base), grid, n_iter=25, scoring="average_precision",
        cv=GroupKFold(n_splits=4), n_jobs=4, random_state=SEED,
        refit=True, verbose=1, return_train_score=False)

    t0 = time.time()
    search.fit(df[COLS], df[LABEL], groups=df["year"])
    print(f"\nsearch done in {time.time()-t0:.0f}s over {len(search.cv_results_['params'])} configs")

    res = pd.DataFrame(search.cv_results_)
    keep = ["mean_test_score", "std_test_score", "rank_test_score"] + \
           [c for c in res.columns if c.startswith("param_clf__")]
    res = res[keep].sort_values("rank_test_score")
    res.to_csv(OUT / "search_results.csv", index=False)

    print("\ntop 5 configs by year-grouped CV average precision:")
    print(res.head(5).to_string(index=False))

    best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    print(f"\nbest CV AP = {search.best_score_:.4f}")
    print(f"best params: {best}")

    # ---------- stage 2: confirm under the same LOYO as 05 ----------
    tuned = dict(best, early_stopping=True, validation_fraction=0.1,
                 n_iter_no_change=20, max_iter=1000, random_state=SEED)

    print("\nre-scoring default vs tuned under full leave-one-year-out ...")
    d_sc = loyo_scores(df, DEFAULT)
    t_sc = loyo_scores(df, tuned)

    comp = pd.DataFrame([
        {"config": "default (05)", **d_sc, **{f"p_{k}": v for k, v in DEFAULT.items()}},
        {"config": "tuned",        **t_sc, **{f"p_{k}": v for k, v in tuned.items()}},
    ])
    comp.to_csv(OUT / "tuned_vs_default.csv", index=False)

    print(f"\n  default   AUC {d_sc['auc']:.4f}  AP {d_sc['ap']:.4f}")
    print(f"  tuned     AUC {t_sc['auc']:.4f}  AP {t_sc['ap']:.4f}")
    print(f"  AP change {t_sc['ap'] - d_sc['ap']:+.4f}   AUC change {t_sc['auc'] - d_sc['auc']:+.4f}")
    print(f"\nwrote {OUT/'search_results.csv'} and tuned_vs_default.csv")
    print("note: search saw all open years, so the tuned LOYO numbers are mildly optimistic.")


if __name__ == "__main__":
    main()
