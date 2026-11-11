"""
WP6 step 4: score the chosen models on the locked test years 2024 and 2025, ONCE.

Run this only after the label (S, x) and the models are fixed from steps 1-3 (brief rule 5).
The models are trained on all open years (2016-2023) and scored on 2024 and 2025. 2026 is the
final test and is not read. The script writes outputs/test_scores.done and refuses to run again,
so the test years are not used for tuning by accident.

Needs the WP5 panel built with test years:  cd ../WP5 && python cell_month_panel.py --locked

Inputs:  WP5 panel (with --locked) + WP2 labels
Outputs (outputs/):
    test_scores.csv     per test year and pooled: AUC, AP, Brier, log loss, with 95% intervals
                        from a bootstrap over year-months
    test_scores.done    marker that the test years have been scored

Run:  python 04_test_scores.py --label y_S383_any --yes-score-test-years
"""
import argparse
import importlib.util
from datetime import datetime

import numpy as np
import pandas as pd

from wp6_common import HERE, OPEN_YEARS, OUT, TEST_YEARS, available, load_panel

spec = importlib.util.spec_from_file_location("step3", HERE / "03_logistic.py")
step3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="the label chosen in steps 1-3, e.g. y_S383_any")
    ap.add_argument("--yes-score-test-years", action="store_true", help="confirm the models are final")
    ap.add_argument("--force", action="store_true", help="run again even though the test years were scored")
    args = ap.parse_args()
    marker = OUT / "test_scores.done"
    if not args.yes_score_test_years:
        raise SystemExit("this scores the locked test years; pass --yes-score-test-years once the models are final")
    if marker.exists() and not args.force:
        raise SystemExit(f"test years already scored ({marker.read_text().strip()}); see outputs/test_scores.csv")

    df = load_panel(OPEN_YEARS + TEST_YEARS).reset_index(drop=True)
    feats = available(df)
    y = args.label
    train, test = df[df.year.isin(OPEN_YEARS)], df[df.year.isin(TEST_YEARS)].reset_index(drop=True)
    print(f"train {len(train):,} rows ({int(train[y].sum())} positives), "
          f"test {len(test):,} rows ({int(test[y].sum())} positives), features: {feats}")

    pred = pd.DataFrame({m: step3.fit_predict(m, train, test, y, feats) for m in step3.MODELS})
    rng = np.random.default_rng(2024)
    rows = []
    for name, idx in [("2024", test.year == 2024), ("2025", test.year == 2025), ("2024+2025", test.year > 0)]:
        t, p = test[idx], pred[idx.to_numpy()].reset_index(drop=True)
        bs = step3.bootstrap(t.reset_index(drop=True), y, p, step3.MODELS, rng)
        for m in step3.MODELS:
            b = bs[bs.model == m]
            rows.append({"years": name, "label": y, "model": m, "positives": int(t[y].sum()),
                         **step3.scores(t[y], p[m]),
                         "auc_lo": b.auc.quantile(.025), "auc_hi": b.auc.quantile(.975),
                         "ap_lo": b.ap.quantile(.025), "ap_hi": b.ap.quantile(.975)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "test_scores.csv", index=False)
    marker.write_text(f"{y} scored {datetime.now():%Y-%m-%d %H:%M}\n")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
