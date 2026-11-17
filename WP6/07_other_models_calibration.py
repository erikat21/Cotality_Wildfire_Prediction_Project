from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

HERE = Path(__file__).resolve().parent
BASE = HERE / "outputs" / "other_models"

LABELS = ["y_S30_any", "y_S100_any", "y_S383_any"]
MODELS = ["logit_full", "random_forest", "hist_gbt"]

COLORS = {
    "logit_full": "#eb6834",
    "random_forest": "#2e9e6b",
    "hist_gbt": "#9b59b6",
}


def main():
    rows = []

    for label in LABELS:
        folder = BASE / label
        path = folder / "other_models_oof.parquet"

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_parquet(path)
        y = df[label].to_numpy()

        fig, ax = plt.subplots(figsize=(7, 6))

        max_rate = 0.0

        for model in MODELS:
            p = df[model].to_numpy()

            observed, predicted = calibration_curve(
                y,
                p,
                n_bins=10,
                strategy="quantile"
            )

            ax.plot(
                predicted * 1000,
                observed * 1000,
                marker="o",
                label=model,
                color=COLORS[model]
            )

            max_rate = max(
                max_rate,
                predicted.max() * 1000,
                observed.max() * 1000
            )

            rows.append({
                "label": label,
                "model": model,
                "brier": brier_score_loss(y, p),
                "logloss": log_loss(y, np.clip(p, 1e-6, 1-1e-6))
            })

        ax.plot(
            [0, max_rate],
            [0, max_rate],
            "--",
            color="gray",
            label="Perfect calibration"
        )

        ax.set_xlabel("Predicted fire rate per 1,000 cell-months")
        ax.set_ylabel("Observed fire rate per 1,000 cell-months")
        ax.set_title(f"Model Calibration — {label}")
        ax.legend()
        ax.grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(
            folder / "fig_other_models_calibration.png",
            dpi=160
        )
        plt.close(fig)

        print(f"Completed calibration: {label}")

    results = pd.DataFrame(rows)
    results.to_csv(
        BASE / "calibration_summary.csv",
        index=False
    )

    print("\nCalibration Summary:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
