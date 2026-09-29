# House Prices – Advanced Regression Techniques

Kaggle "Getting Started" competition: predicting Ames, Iowa house sale prices
from 79 features. Evaluated on RMSLE (log-space RMSE).

**Best public leaderboard RMSLE: 0.12196**
*(rank 473 of 3,769 participants at the time of submission)*

## A note on the leaderboard

The Ames dataset has been publicly available for many years, so leaderboard
scores are not always directly comparable as demonstrations of modelling
quality alone. This project deliberately uses only the competition features:
no external price lookup, target matching, or manual adjustment of individual
test predictions.

## Approach

Three notebooks, run in order:

1. `01_data_audit_and_baselines.ipynb` — data contract check, train/holdout
   split, dummy and two-feature baselines, target distribution, known
   Ames outliers (Id 524/1299).
2. `02_model_comparison_and_tuning.ipynb` — feature engineering (reused from
   a prior classification project on the same dataset), ordinal/nominal
   encoding, model comparison (Lasso, HistGradientBoostingRegressor,
   RandomForest), hyperparameter tuning, paired CV comparisons with a fixed
   decision gate (≥0.003 RMSLE improvement AND ≥4/5 folds same direction),
   ablations (outlier removal, skewed-feature log-transform, blending).
3. `03_final_submission_generation.ipynb` — final fit on all 1460 training
   rows, predictions on the competition test set, submission files.

## Key findings

| Experiment | Result |
|---|---|
| Training on `log1p(SalePrice)` | Metric-aligned target transformation; materially improved RMSLE |
| Dropping known GrLivArea outliers (Id 524/1299) | Gate failed on paired CV — kept in training (test set contains a similar case, Id 2550) |
| Log-transforming skewed input features | Clear win (Δ0.0145 RMSLE, 5/5 folds) — the single biggest lever found |
| Blending Lasso + HistGB | Passed gate cleanly on the improved features; beat both individual models on the public leaderboard |

## Submissions

| # | Model | CV RMSLE | Holdout RMSLE | Leaderboard RMSLE |
|---|---|---|---|---|
| 1 | Lasso | 0.1495 | 0.1025 | 0.13134 |
| 2 | HistGradientBoostingRegressor (tuned) | 0.1335 | 0.1088 | 0.13107 |
| 3 | Blend 0.7×HistGB + 0.3×Lasso | — | — | 0.12526 |
| 4 | Lasso + log-transformed skewed features | 0.1351 (Δ0.0145, 5/5 folds) | — | 0.12373 |
| 5 | 0.5×HistGB + 0.5×Lasso on the improved feature pipeline | Δ0.0054, 4/5 folds | — | **0.12196** |

## Data

Not included in this repo per Kaggle's competition rules (no redistribution
outside the platform). Download `train.csv`, `test.csv`,
`sample_submission.csv`, and `data_description.txt` from the
[competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
and place them in this directory.

## Reproducibility

- Python: 3.11.15
- scikit-learn: 1.9.1
- Fixed random seed: 123
- Run notebooks in numerical order.
- The competition data is intentionally excluded from version control.
- Install dependencies: `pip install -r requirements.txt`

## Notes

Built with input from Claude and ChatGPT as independent reviewers; each
caught real methodological bugs the other missed during the session
(e.g. a scoring metric mismatch, an unpaired ablation test, a feature
engineering edge case). Feature engineering foundation carried over from
an earlier classification project on the same Ames dataset.