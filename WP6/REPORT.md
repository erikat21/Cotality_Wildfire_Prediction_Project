# vn0 analysis (WP6)

Data: the WP5 panel (DVC, md5 c249f963…) joined with the WP2 labels. That's 6,555 cells x 92 months (May 2016 to Dec 2023) = 603,060 cell-months, with no missing values. Validation is leave one year out, and intervals are 95% from a bootstrap over year-months. The test years (2024, 2025) have not been looked at. The results without weather are kept in `outputs/no_weather/` for comparison.

## 1. Labels (step 1)
| positives | any | 1% | 2% | 5% | 10% |
|---|---|---|---|---|---|
| S = 30 ha | 4,402 | 2,158 | 1,395 | 734 | 443 |
| S = 100 ha | 2,538 | 1,931 | 1,300 | 717 | 435 |
| S = 383 ha | 1,295 | 1,053 | 951 | 692 | 418 |

- At S = 383 ha, any overlap, about 2 in every 1,000 cell-months is a fire.
- A larger S makes the label more of a summer label. Positives in June to September are 52% of the total at S = 30, 64% at S = 100 and 75% at S = 383. The S = 30 label has a second peak in February and March.
- Years differ a lot. At S = 383, 2017 has 439 positives and 2018 has 25.
- At S = 383, 3,107 cell-months had only smaller fires and are labelled 0.
- Large fires concentrate in northern and central Portugal and in Galicia.

## 2. How the fire rate depends on the features (step 2)
Fire rate per 1,000 cell-months, lowest vs highest of about 10 bins, at S = 383 ha:

| feature | raw value | anomaly (vs the cell's usual value that month) |
|---|---|---|
| max VPD (dryness) | 0.17 → 5.9 | 0.8 → 9.8 |
| max temperature | 0.30 → 6.1 | 0.08 → 7.4 |
| rain (log) | 4.6 → 0.22 | 2.9 → 1.1 |
| dewpoint | 0.46 → 2.8 | 2.8 → 0.05 |
| NDVI, previous month | 0.25 → 2.0 | 2.8 → 1.1 |
| shrub/grass share | 0.13 → 6.4 | |
| slope | 0.23 → 3.7 | |
| farmland share | 2.7 → 0.0 | |

- Weather has a strong effect **beyond season and place**. In the 10% of cell-months that are driest compared with what is usual for that cell and month, the fire rate is about 4.5 times the overall average (2.1). In the 10% that are coolest compared with usual, large fires almost never happen (0.08).
- Fires also follow the vegetation and terrain: shrub, grass and forest burn, steep and rugged cells burn, and farmland hardly burns.

## 3. Does the logistic model beat the baselines? (step 3)
Main label S = 383 ha, any overlap:

| model | AUC | AP |
|---|---|---|
| season (month rate) | 0.69 (0.55 to 0.79) | 0.004 |
| season+place (cell's own month rate) | 0.80 (0.73 to 0.87) | 0.012 |
| logit_base (month + region) | 0.77 (0.69 to 0.84) | 0.010 |
| logit_full without weather | 0.82 (0.75 to 0.89) | 0.015 |
| **logit_full** (all features) | **0.88 (0.85 to 0.91)** | **0.022 (0.012 to 0.037)** |
| logit_full minus season+place | **+0.084 (+0.034 to +0.130)** | **+0.009 (+0.004 to +0.017)** |

- **Yes.** With weather, the logistic model beats every baseline, and the interval is well clear of zero. It wins in all 300 bootstrap draws. Weather is what makes the difference: without it the gain over the cell-history baseline was not significant.
- This holds for all 15 label definitions. logit_full has the best AUC (0.88 to 0.90) and the best AP in every one. The gain is largest at x = 10%, where AP more than doubles.
- By AUC, the model is better in 6 of the 8 held-out years. The cell-history baseline is level in 2019 (0.873 vs 0.870) and ahead in 2023 (0.79 vs 0.74). 2023 is the hardest year for every model.
- Dropping the small-fire-only cell-months changes almost nothing (AUC 0.884 → 0.886).
- Coefficients (odds ratio per 1 SD, with month and region held fixed, and the same sign in all 8 folds unless noted):
  - hotter is higher risk (tmax). Dewpoint, which stands for humidity at a given temperature, goes the other way: odds ratio 0.49.
  - rain 0.89, elevation 0.67, farmland 0.68
  - shrub/grass 1.5, elevation spread 1.3, NDVI previous month 1.6 (more fuel)
  - tmax, VPD and dewpoint are strongly correlated (tmax vs VPD r = 0.92), so the weight is shared among them in different ways from fold to fold. Read them as one "hot and dry" signal rather than three separate effects. Slope and the south-facing share flip sign between folds and carry little.
- The probabilities are reasonably calibrated (`fig_calibration.png`). A plain logistic model with no reweighting keeps the real base rate.

## 4. Is the pipeline sound enough for later models?
Mostly yes. Every table joins on `cell_id`, `year` and `month` without duplicates or gaps. The WP5 `fire` column matches the WP2 label for 30 ha, any overlap, exactly (4,402 positives). The base rate at 383 ha is 0.21%, a little under the slides' estimate of 0.26%. Some things to tidy up before random forest or XGBoost:
- WP5 fills the remaining gaps with a median taken over all years in the table. When it is rebuilt with `--locked`, that median includes the test years, and slide rule 2 says it should use 2016 to 2023 only. Only a few values are affected, but the fix is easy.
- WP5 adds only one label. The other 14 come from WP2's `labels.parquet`, which WP6 joins in. Later models should do the same.
- tmax, VPD and dewpoint are strongly correlated, which matters for reading coefficients but not for tree models.
