# WP6: Statistical Analysis and Logistic Fit

WP6 uses the WP5 cell-month panel and WP2 fire labels to address the three **vn0** questions: (1) how monthly fire occurrence is associated with weather, vegetation, land cover, and terrain beyond season and place; (2) how findings change with the fire-size threshold **S** and burnt-cell fraction **x**; and (3) whether the resulting panel and modeling pipeline provide a sound foundation for later models.

This is an **open-period, retrospective monthly association and classification analysis**. Same-month weather aggregates are part of the vn0 specification; reported cross-validation scores should **not** be interpreted as performance of a forecast issued at the start of the month. The 2024–2025 locked test has **not** been scored in this update.

## Setup

From the repository root:

```powershell
pip install pandas numpy pyarrow scipy scikit-learn matplotlib
```

Required processed data:

- `WP5/outputs/cell_month_panel.parquet`: **Open Period**, May 2016–December 2023 (6,555 cells × 92 months = 603,060 rows).
- `wp2/outputs/labels.parquet`: the 15 official fire labels (`y_S{30,100,383}_{any,x01,x02,x05,x10}`), and fire counts for each size threshold.
- `wp2/outputs/cells.parquet`: WP2 grid/mask and region identifiers for fire-only statistics.
- `WP5/outputs/cell_month_panel_locked.parquet`: optional **locked-inclusive** panel (2016–2026), separate from the Open Panel. Only required for the authorized final test workflow. This file has not been validated or used here.

The revised WP5 panel uses `y_S30_any` for `fire` and WP2's union-based `burn_frac_S30` for `burn_frac`. Its feature-wise spatial imputation preserves observed values, uses forward fill rather than backward fill, and removes the original full-panel median fallback. WP5 validation found zero missing values and zero disagreements against the official WP2 S30 labels in the Open Period.

## Run, in order

Run from the **repository root**:

```powershell
python -u WP6/01_fire_stats.py      # Open-period label sweep, calendar/region statistics
python -u WP6/02_dependence.py      # Descriptive associations by feature bins
python -u WP6/03_logistic.py        # LOYO CV for all 15 S × x labels, plus bootstrap
```

For a quick diagnostic of the three `x = any` labels, use `python -u WP6/03_logistic.py --only-any`. This rewrites the same output filenames and is **not** a substitute for the full 15-label experiment when preparing final results.

**Locked evaluation — not executed in this update:** `04_test_scores.py` is reserved for the once-only evaluation of 2024–2025, after the team has fixed the outcome definition, modeling choices, and testing protocol. The corrected `wp6_common.py` selects `cell_month_panel_locked.parquet` when test years are explicitly requested; simply running WP5 with `--locked` does not overwrite the Open Panel. Do **not** run the locked evaluation during routine validation.

## Choices made

- **Labels:** Analyze the full 3 × 5 sweep with minimum whole-fire size `S ∈ {30,100,383}` hectares and minimum burnt-cell fraction `x ∈ {any,1%,2%,5%,10%}`. WP6 joins all 15 WP2 labels by (`cell_id`, `year`, `month`); missing sparse label rows are treated as negatives. The WP5 `fire` field agrees exactly with `y_S30_any`.
- **Period and validation:** Open Period is May 2016–December 2023. Each held-out year is evaluated using a **leave-one-year-out (LOYO)** model trained on the other Open years, not a random row split. This tests across-year generalization but is not a purely forward-chaining forecast simulation.
- **Models:** `season` (calendar-month fire rate), `season+place` (cell × month rate shrunk toward NUTS2 × month and month rates), `logit_base` (calendar month + NUTS2), and `logit_full` (same categorical predictors plus environmental features).
- **Logistic optimization:** `LogisticRegression(C=1.0, max_iter=3000, tol=1e-8)` with L2 regularization. A tighter tolerance was introduced after default-tolerance sensitivity checks showed materially different fitted coefficients and out-of-fold predictions. Standardization and category encoding are fitted within the training years of each LOYO fold. Convergence iteration-count diagnostics were added for prediction fits and main-label coefficient fits.
- **Features (16):** `tmax`, `vpdmax`, `tdmean`, `ppt_log = log(1+ppt)`, `ndvi_lag1`, five terrain features, and six CORINE land-cover shares. `water_share` is omitted to avoid the exact sum-to-one constraint with the other shares. The shared loader retains feature-availability fallback for deliberate no-weather comparisons; official full-feature runs should check the printed feature list.
- **Metrics and uncertainty:** ROC AUC, average precision (AP; a summary of the precision–recall curve), Brier score, and log loss. The 95% intervals use **300 bootstrap resamples of whole year-month blocks** of the out-of-fold predictions. For comparisons with the best baseline selected by point-estimate AP, `auc_share_of_draws_better` and `ap_share_of_draws_better` are now calculated **separately**.
- **Small-fire sensitivity:** For the main label `y_S383_any`, refit after excluding cell-months containing only fires below 383 ha.
- **Interpretation:** Monthly weather is observed contemporaneously with the fire label; these results describe retrospective association/classification, not advance warnings. Correlated `tmax`, `vpdmax` and `tdmean` produce sensitive conditional odds ratios. Feature-bin plots and coefficient estimates must not be read as causal effects.

## Outputs (`WP6/outputs/`)

| Files | Content |
|---|---|
| `label_sweep.csv`, `label_by_year.csv`, `label_by_month.csv`, `label_by_region.csv` | Fire-label counts/rates by threshold, year, month, region |
| `fig_label_sweep.png`, `fig_seasonality.png`, `fig_by_year.png`, `fig_fire_map.png` | Fire-only analysis figures |
| `dependence_bins.csv`, `fig_dependence.png`, `fig_dependence_anom.png` | Rates across raw-feature and anomaly bins |
| `cv_sweep.csv`, `fig_sweep.png` | Out-of-fold scores for all 15 labels and four models |
| `cv_bootstrap.csv` | Main label and size-threshold comparisons with 95% intervals and **separate AUC/AP bootstrap win shares** |
| `cv_by_year.csv`, `oof_main.parquet` | Main-label held-out-year scores and predictions |
| `small_fire_check.csv` | Main-label sensitivity to excluding smaller-fire-only rows |
| `coefficients.csv`, `fig_coefficients.png` | Main-label full-fit OR per SD and range across LOYO fits |
| `fig_calibration.png` | Predicted versus observed rate in out-of-fold bins |
| `test_scores.csv`, `test_scores.done` | Reserved for the once-only locked evaluation; **not generated in this update** |

## Validated Open-Period results

All three Open Period scripts were rerun against the revised WP5 pipeline on November 10, 2026. `03_logistic.py` completed all **15** labels with `tol=1e-8`; the additional `01_fire_stats.py` and `02_dependence.py` runs regenerated their outputs.

- **Panel:** 603,060 cell-months, 6,555 grid cells, 92 months, 24 NUTS2 regions, 16 predictors; zero missing predictor/label values after joining.
- **Positive labels (`x=any`):** S30 = **4,402**; S100 = **2,538**; S383 = **1,295** (S383 base rate **2.147 per 1,000** cell-months).
- **Seasonality:** 74.5% of S383 positives occur June–September (versus 51.6% for S30).
- **Descriptive associations, S383:** lowest-to-highest raw decile rates per 1,000 cell-months: `vpdmax` 0.17 → 5.86, `tmax` 0.30 → 6.13, `ppt_log` 4.57 → 0.22, and `shrub_herbaceous_share` 0.13 → 6.44. These are marginal associations, not adjusted effects.
- **Full logistic, S383 any:** ROC AUC **0.8969** (95% block-bootstrap interval **0.8688–0.9231**); AP **0.0243** (**0.0128–0.0405**). Against `season+place`: ΔAUC **+0.0962** (95% interval **+0.0440–+0.1519**), ΔAP **+0.0119** (**+0.0055–+0.0202**).
- **All-label AP sweep:** `logit_full` had the highest AP point estimate among the four models for all 15 labels. This does **not** establish a statistically reliable AP advantage for every label: e.g., S30 any ΔAP 95% interval **−0.0082 to +0.0253**, with 76.67% of paired bootstrap draws favoring the full logistic model.
- **Solver diagnostic:** Main-label full-data coefficient fit used 257/3000 iterations; eight leave-one-year-out coefficient fits used 214–268/3000. Hitting no iteration limit alone is not proof of numerical optimality; the tight-tolerance sensitivity check motivated the change.

See [`REPORT.md`](REPORT.md) for tables, interpretation, and limitations. Results were obtained using the **Open Period only**; there are no 2024–2025 test scores in this update.
