# vn0 analysis (WP6) — Updated Open-Period Results

**Scope.** Monthly fire-occurrence associations and baseline/logistic classification for Iberia, as described in the vn0 student briefing. This analysis investigates the dependence of monthly burnt-cell outcomes on contemporaneous weather, previous-month NDVI, terrain and land cover, and sensitivity to fire-label definitions. **It does not evaluate an advance month-start wildfire forecast.**

**Data and evaluation.** Revised WP5 Open Panel joined to WP2's official fire labels: **6,555 cells × 92 months (May 2016–December 2023) = 603,060 cell-months**. There are **24 regions, 16 model predictors, and zero missing values** in the joined data. WP5's binary fire labels and union-based burnt fractions exactly matched the corresponding WP2 S30 reference in the preceding integration check. Logistic/baseline comparisons use eight-fold **leave-one-year-out (LOYO)** cross-validation. The reported 95% intervals use **300 bootstrap resamples of year-month blocks** of out-of-fold predictions. The **2024–2025 locked test was not accessed or scored**; 2026 remains reserved as a final test year.

**Update notes.** The logistic fitting tolerance was tightened to **`tol=1e-8`**, with `max_iter=3000`, `C=1.0` and the existing L2 regularization. The default-tolerance solver had produced notably different fitted coefficients and somewhat lower cross-validation scores. Bootstrap comparisons now correctly report **distinct AUC and AP win shares** instead of merging the two. The results below replace the earlier report's default-tolerance model estimates. Fire-only and feature-dependence outputs were also regenerated with the current processed data.

## 1. Labels and where fires occur (step 1)

Positive cell-months for each minimum whole-fire size **S** and minimum within-cell burnt fraction **x**:

| Minimum fire size | Any overlap | ≥1% | ≥2% | ≥5% | ≥10% |
|---|---:|---:|---:|---:|---:|
| 30 ha | 4,402 | 2,158 | 1,395 | 734 | 443 |
| 100 ha | 2,538 | 1,931 | 1,300 | 717 | 435 |
| 383 ha | 1,295 | 1,053 | 951 | 692 | 418 |

- At **S383 any**, the positive rate is **2.147 per 1,000 cell-months** (0.2147%), compared with **7.299** at S30 any.
- The share of positives in **June–September** increases from **51.6%** for S30 any to **63.5%** for S100 any and **74.5%** for S383 any.
- Large-fire counts vary across years: for the complete calendar years, the S383-any count ranges from **25** to **439** positives. These values come from the current `label_sweep.csv`; the earlier report identifies 2018 and 2017, respectively.
- **3,107** S30-positive cell-months contain no qualifying S383 fire and therefore become negatives under S383. These are the basis of the small-fire sensitivity test.
- At S383 any, the highest reported regional rates are **PT19: 13.544** and **PT11: 10.823** positive cell-months per 1,000; **ES12: 4.364** and **ES11: 4.258** are among the higher Spanish regions. Region rates are descriptive and should be interpreted alongside the number of cells per region.

See `fig_label_sweep.png`, `fig_seasonality.png`, `fig_by_year.png`, `fig_fire_map.png`, and the corresponding label CSV files.

## 2. How fire occurrence varies with environmental features (step 2)

Observed S383-any positives **per 1,000 cell-months** in the lowest versus highest quantile bin of each feature (approximately ten bins, fewer when values are tied):

| Feature | Raw feature: low → high | Cell × calendar-month anomaly: low → high |
|---|---:|---:|
| Maximum vapor-pressure deficit (`vpdmax`) | 0.17 → 5.86 | 0.80 → 9.83 |
| Maximum temperature (`tmax`) | 0.30 → 6.13 | 0.08 → 7.38 |
| Log rainfall (`ppt_log`) | 4.57 → 0.22 | 2.90 → 1.13 |
| Dewpoint (`tdmean`) | 0.46 → 2.78 | 2.82 → 0.05 |
| Previous-month NDVI (`ndvi_lag1`) | 0.25 → 2.04 | 2.84 → 1.13 |
| Shrub/herbaceous land-cover share | 0.13 → 6.44 | — |
| Forest share | 0.28 → 1.77 | — |
| Agricultural share | 2.70 → 0.00 | — |
| Mean slope | 0.23 → 3.67 | — |
| Elevation spread | 0.43 → 3.93 | — |

Within this retrospective sample, cell-months with higher raw VPD and temperature and lower rainfall tend to have higher observed large-fire rates. High shrub/herbaceous share and slope likewise coincide with elevated rates; agricultural share shows the opposite descriptive pattern. Anomaly bins compare each monthly value to the cell's average for the same calendar month **within the available Open Period**; they do **not** prove causal effects or independently adjust for all possible confounding. Anomalies are used for descriptive plots, not as predictors in the current LOYO logistic model.

See `dependence_bins.csv`, `fig_dependence.png`, and `fig_dependence_anom.png`.

## 3. Logistic regression versus simple baselines (step 3)

### Main label: S383, any overlap

| Model | Pooled LOYO ROC AUC (95% block-bootstrap interval) | Pooled LOYO AP (95% interval) |
|---|---:|---:|
| `season` | 0.6887 (0.5523–0.7927) | 0.0044 (0.0026–0.0076) |
| `season+place` | 0.8008 (0.7333–0.8711) | 0.0124 (0.0063–0.0236) |
| `logit_base` | 0.7832 (0.7096–0.8545) | 0.0118 (0.0054–0.0253) |
| **`logit_full`** | **0.8969 (0.8688–0.9231)** | **0.0243 (0.0128–0.0405)** |
| **`logit_full` − `season+place`** | **+0.0962 (+0.0440–+0.1519)** | **+0.0119 (+0.0055–+0.0202)** |

For the paired year-month bootstrap, **100% of 300 draws** showed improvement in S383 AUC and AP against `season+place`. This describes the empirical fraction of bootstrap draws, **not** a posterior probability of model superiority or an independent test-set result.

### All 15 definitions: average precision

| Label | Positive cell-months | `season` AP | `season+place` AP | `logit_base` AP | `logit_full` AP |
|---|---:|---:|---:|---:|---:|
| S30 any | 4,402 | 0.0104 | 0.0766 | 0.0396 | **0.0832** |
| S30 ≥1% | 2,158 | 0.0056 | 0.0315 | 0.0185 | **0.0437** |
| S30 ≥2% | 1,395 | 0.0042 | 0.0200 | 0.0134 | **0.0291** |
| S30 ≥5% | 734 | 0.0024 | 0.0106 | 0.0063 | **0.0148** |
| S30 ≥10% | 443 | 0.0013 | 0.0032 | 0.0032 | **0.0085** |
| S100 any | 2,538 | 0.0066 | 0.0311 | 0.0216 | **0.0458** |
| S100 ≥1% | 1,931 | 0.0054 | 0.0252 | 0.0174 | **0.0372** |
| S100 ≥2% | 1,300 | 0.0041 | 0.0177 | 0.0126 | **0.0278** |
| S100 ≥5% | 717 | 0.0025 | 0.0104 | 0.0064 | **0.0149** |
| S100 ≥10% | 435 | 0.0013 | 0.0032 | 0.0032 | **0.0087** |
| S383 any | 1,295 | 0.0044 | 0.0124 | 0.0118 | **0.0243** |
| S383 ≥1% | 1,053 | 0.0037 | 0.0122 | 0.0104 | **0.0219** |
| S383 ≥2% | 951 | 0.0032 | 0.0112 | 0.0093 | **0.0201** |
| S383 ≥5% | 692 | 0.0023 | 0.0086 | 0.0061 | **0.0144** |
| S383 ≥10% | 418 | 0.0012 | 0.0030 | 0.0031 | **0.0085** |

`logit_full` has the highest **AP point estimate** for every label. AP is prevalence-sensitive, so absolute values should not be compared across differently defined labels without accounting for their distinct positive rates. These 15 rows are point estimates; the current code provides the paired bootstrap comparisons for the three `x=any` labels, not for every fraction threshold.

### Bootstrap improvement by fire-size threshold (`x = any`)

| Label | ΔAUC (95% interval) | ΔAP (95% interval) | Fraction of draws with ΔAUC > 0 | Fraction with ΔAP > 0 |
|---|---|---|---:|---:|
| S30 any | +0.0405 (+0.0248–+0.0588) | +0.0067 (**−0.0082–+0.0253**) | 100% | 76.67% |
| S100 any | +0.0682 (+0.0411–+0.0980) | +0.0147 (+0.0037–+0.0245) | 100% | 100% |
| S383 any | +0.0962 (+0.0440–+0.1519) | +0.0119 (+0.0055–+0.0202) | 100% | 100% |

In particular, the S30 AP interval includes zero: the full model has a higher AP point estimate, but the 95% paired interval does **not** establish a clear AP improvement for that label.

### Small-fire sensitivity and model coefficients

Removing **3,107** cell-months with smaller fires but no qualifying S383 fire changes `logit_full` ROC AUC from **0.8969 to 0.8983** and AP from **0.0243 to 0.0252**. The corresponding `season+place` AP changes from **0.0124 to 0.0129**. This is a modest sensitivity in the reported metrics.

The main-label full-data logistic coefficients are conditional **odds ratios per one training-data standard deviation**, with calendar month and NUTS2 held fixed. Selected estimates:

| Feature | Full-data odds ratio | LOYO fold OR range | Same sign in all folds? |
|---|---:|---:|---|
| `tmax` | 91.751 | 8.125–128.586 | Yes |
| `vpdmax` | 0.256 | 0.197–0.956 | Yes |
| `tdmean` | 0.141 | 0.102–0.419 | Yes |
| `ppt_log` | 1.087 | 0.924–1.142 | **No** |
| `ndvi_lag1` | 1.313 | 1.149–1.481 | Yes |
| `elev_mean` | 0.537 | 0.456–0.615 | Yes |
| `slope_mean` | 1.120 | 1.016–1.166 | Yes |
| `slope_p90` | 0.942 | 0.901–1.030 | **No** |
| `shrub_herbaceous_share` | 1.583 | 1.472–1.795 | Yes |

**Do not interpret `tmax` OR ≈ 92 as the effect of temperature alone.** The weather predictors are strongly correlated, and their jointly adjusted coefficients are sensitive to the optimization tolerance and included features. The original default-tolerance fit gave `tmax` OR ≈ 5.862, compared with 91.751 under strict tolerance; this prompted the re-fit. Nine coefficient fits used **214–268 iterations**, below `max_iter=3000`; the convergence diagnostics do not independently certify exact optimality. For physical interpretation, emphasize descriptive relationships, uncertainty, and correlated predictor groups rather than a causal reading of isolated odds ratios.

## 4. Is this pipeline sound enough for later models?

**The Open-Period data integration and statistical workflow are functioning, with identifiable limitations.**

- **Join and coverage checks:** The revised WP5 produced 603,060 unique Open Period cell-months (6,555 × 92) with zero missing values. WP6's shared loader validates panel/label keys, month coverage, and S30 label agreement. A prior direct comparison also found **0** WP5-vs-WP2 binary label disagreements and **0** burnt-fraction disagreements.
- **Imputation is already repaired:** The earlier report stated WP5 used backward filling and a median estimated over all years. **That statement is obsolete.** WP5 now imputes physical features independently using month-wise nearest valid spatial neighbors, has a past-only temporal forward-fill fallback, and no longer uses backward filling or a whole-panel median. This removes those specific pathways but does not prove every preprocessing operation is valid for every forecasting use case.
- **Other outcome definitions:** WP5 exports S30 any as its primary binary label, while WP6 joins the other 14 definitions directly from WP2's sparse `labels.parquet`. Future models should retain the exact label definition and join integrity checks.
- **Same-month weather:** Monthly `tmax`, `vpdmax`, `tdmean`, and `ppt` are contemporaneous aggregates specified for vn0. They are appropriate for **retrospective association/classification**, but are not known at the beginning of the month. A true advance forecast would require a new feature-availability design, such as lagged observations or operational weather forecasts.
- **Rare events and uncertainty:** S383 any positives occur in only 0.2147% of Open Period rows. Report AP/precision–recall performance alongside ROC AUC; do not rely on accuracy alone. Block-bootstrap intervals reflect resampling of fixed out-of-fold predictions, not every source of training or data uncertainty.
- **Validation interpretation:** LOYO trains on all other Open Period years, including years later than the held-out year. It provides across-year robustness estimates, **not** a chronological expanding-window backtest. Scaling is fold-fitted, but descriptive anomaly baselines are computed across Open Period for the plots and should not be carried as predictive features without fitting climatologies within each training fold.
- **Land-cover chronology:** CORINE 2012 is mapped to 2016–2017, and CORINE 2018 from 2018 onward in the current retrospective data convention. Revisit map-publication timing before any real-time prediction use.
- **Locked-test discipline:** Do not score 2024–2025 until the team has selected the final label and modeling protocol. The locked-inclusive WP5 panel is written to a separate filename and was **not** used in the reported experiments. The 2026 final year remains excluded.

**Conclusion.** Under the vn0's retrospective monthly analysis setup, the updated full-feature logistic model consistently improves AP point estimates across the 15 label definitions and has clearly positive paired bootstrap AP differences for S100 any and S383 any. The data pipeline is sufficiently integrated for follow-on modeling experiments, provided their prediction-time assumptions, label definitions, rare-event evaluation, and locked-test governance are made explicit.
