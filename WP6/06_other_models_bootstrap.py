from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

HERE = Path(__file__).resolve().parent

LABEL = "y_S30_any"

OUT = HERE / "outputs" / "other_models" / LABEL

OOF_PATH = OUT / "other_models_oof.parquet"

N_BOOT = 300
RANDOM_SEED = 180

MODELS = ["logit_full", "random_forest", "hist_gbt"]

COMPARISONS = [
    ("random_forest", "logit_full"),
    ("hist_gbt", "logit_full"),
    ("hist_gbt", "random_forest"),
]

# Reuse the existing WP6 bootstrap implementation.
spec = importlib.util.spec_from_file_location(
    "wp6_logistic",
    HERE / "03_logistic.py"
)
step3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step3)


def main():
    if not OOF_PATH.exists():
        raise FileNotFoundError(OOF_PATH)

    df = pd.read_parquet(OOF_PATH)

    required = [
        "cell_id", "year", "month", LABEL
    ] + MODELS

    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("OOF data contains missing values")

    if df.duplicated(["cell_id", "year", "month"]).any():
        raise ValueError("Duplicate cell-month keys")

    if not df[LABEL].isin([0, 1]).all():
        raise ValueError("Fire label must be binary")

    if not set(df["year"].unique()).issubset(
        set(range(2016, 2024))
    ):
        raise ValueError("Unexpected year in Open Period OOF")

    for model in MODELS:
        if not df[model].between(0, 1).all():
            raise ValueError(f"Invalid probabilities: {model}")

    df["ym"] = df["year"] * 100 + df["month"]

    print(f"Observations: {len(df):,}")
    print(f"Positive cases: {int(df[LABEL].sum()):,}")
    print(f"Year-month blocks: {df['ym'].nunique()}")
    print(f"Bootstrap samples: {N_BOOT}")

    pred = df[MODELS].copy()

    # Original model scores
    print("\nOriginal scores:")

    for model in MODELS:
        score = step3.scores(df[LABEL], pred[model])

        print(
            f"{model:15s} "
            f"AUC={score['auc']:.4f} "
            f"AP={score['ap']:.4f}"
        )

    # Paired year-month block bootstrap
    rng = np.random.default_rng(RANDOM_SEED)

    bs = step3.bootstrap(
        df,
        LABEL,
        pred,
        MODELS,
        rng,
        n=N_BOOT
    )

    if bs["draw"].nunique() != N_BOOT:
        raise ValueError("Some bootstrap draws were skipped")

    bs.to_csv(
        OUT / "other_models_bootstrap_draws.csv",
        index=False
    )

    pivot = bs.pivot(
        index="draw",
        columns="model",
        values=["auc", "ap"]
    )

    # Model comparisons
    rows = []

    print("\nBootstrap comparisons:")

    for model_a, model_b in COMPARISONS:
        for metric in ["auc", "ap"]:
            diff = (
                pivot[metric][model_a]
                - pivot[metric][model_b]
            )

            score_a = step3.scores(
                df[LABEL], pred[model_a]
            )[metric]

            score_b = step3.scores(
                df[LABEL], pred[model_b]
            )[metric]

            point_diff = score_a - score_b

            ci_low = diff.quantile(0.025)
            ci_high = diff.quantile(0.975)
            win_rate = (diff > 0).mean()

            rows.append({
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                "difference": point_diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "win_rate": win_rate,
            })

            print(
                f"{model_a} - {model_b} | "
                f"{metric.upper()} "
                f"diff={point_diff:+.5f}, "
                f"95% CI=[{ci_low:+.5f}, "
                f"{ci_high:+.5f}], "
                f"win rate={win_rate:.2%}"
            )

    result = pd.DataFrame(rows)

    result.to_csv(
        OUT / "other_models_bootstrap.csv",
        index=False
    )

    print("\nBootstrap analysis completed.")
    print(f"Results saved to: {OUT}")


if __name__ == "__main__":
    main()
