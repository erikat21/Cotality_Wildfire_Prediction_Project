# WP6: Statistical Analysis and Logistic Fit

Uses the WP5 panel to answer the three vn0 questions: how the chance of a fire depends on the features, how that changes with the label definition (S and x), and whether a logistic model beats simple baselines on years it has not seen.

## Setup
```
pip install pandas numpy pyarrow scikit-learn matplotlib
```
Data (not committed):
- `../WP5/outputs/cell_month_panel.parquet`: output of WP5 (open years; rebuild with `--locked` only for step 4)
- `../wp2/outputs/labels.parquet` and `../wp2/outputs/cells.parquet`: outputs of WP2

## Run, in order
```
python 01_fire_stats.py      # label counts for every S and x, by year, month and region (no panel needed)
python 02_dependence.py      # fire rate across bins of each feature
python 03_logistic.py        # leave-one-year-out: baselines vs logistic, every S and x (~7 min)
python 04_test_scores.py --label y_S383_any --yes-score-test-years   # ONCE, at the very end
```
`03_logistic.py --only-any` runs only x = any (about 2 min).

## Choices made
- **Labels:** WP5's `fire` column is a single definition (any fire of 30 ha or more). WP6 joins all 15 WP2 labels (`y_S{30,100,383}_{any,x01,x02,x05,x10}`) onto the panel, so every step runs for every S and x.
- **Years:** steps 1 to 3 read only 2016 to 2023. Step 4 is the only script that reads 2024 and 2025, and it refuses to run a second time. 2026 is never read.
- **Validation:** leave one year out over 2016 to 2023. Rates, scaling and coefficients are learned from the training years of each fold only.
- **Models:** `season` (rate per calendar month), `season+place` (each cell's own rate for that month, pulled towards its NUTS2 region's rate when it has little history), `logit_base` (logistic on month and region), and `logit_full` (logit_base plus all features). Features are scaled to mean 0 and SD 1, and the logistic uses a light L2 penalty (C = 1).
- **Features:** `tmax`, `vpdmax`, `tdmean`, log(1 + `ppt`), `ndvi_lag1`, terrain, and land-cover shares (`water_share` is left out because the shares sum to 1). Any missing column is skipped, so the scripts run without weather.
- **Scores:** ROC AUC and average precision (PR AUC, the more useful one when only about 2 in 1,000 rows are fires), plus Brier score and log loss. The 95% intervals come from a bootstrap over year-months, not rows.
- **Small-fire check:** the main label is refitted without the cell-months that only had fires smaller than S.

## Outputs (outputs/)
| file | what |
|---|---|
| label_sweep.csv, label_by_year/month/region.csv | positives for each S and x |
| fig_label_sweep, fig_seasonality, fig_by_year, fig_fire_map .png | step 1 figures |
| dependence_bins.csv, fig_dependence(_anom).png | fire rate per feature bin, raw and anomaly |
| cv_sweep.csv, fig_sweep.png | held-out scores of each model for every label |
| cv_bootstrap.csv | main label and S sweep with 95% intervals, logit_full minus best baseline |
| cv_by_year.csv | main label, scores for each held-out year |
| small_fire_check.csv | main label with and without small-fire-only cell-months |
| coefficients.csv, fig_coefficients.png | odds ratio per 1 SD for each feature (main label) |
| fig_calibration.png | predicted vs observed rate on held-out years |
| test_scores.csv | step 4 only: scores on 2024 and 2025 |

## Results
See `REPORT.md`. With weather, the logistic model clearly beats the baselines (S = 383 ha: AUC 0.88 vs 0.80 for the cell-history baseline).
