# Model comparison

Selection metric: cross-validated roc_auc on the training split.
Best model: **xgboost**.
CV columns are mean +/- std over the folds; test columns are on the held-out test split.

| Model | CV AUC (default) | CV AUC (tuned) | CV F1 | Test acc. | Test prec. | Test recall | Test F1 | Test AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.902 +/- 0.069 | 0.903 +/- 0.069 | 0.808 +/- 0.085 | 0.850 | 0.852 | 0.821 | 0.836 | 0.938 |
| random_forest | 0.900 +/- 0.060 | 0.904 +/- 0.055 | 0.790 +/- 0.083 | 0.833 | 0.875 | 0.750 | 0.808 | 0.948 |
| xgboost (best) | 0.872 +/- 0.058 | 0.910 +/- 0.052 | 0.798 +/- 0.063 | 0.867 | 0.885 | 0.821 | 0.852 | 0.929 |

## Best parameters

- **logistic_regression**: `{"C": 1.5966238408663485, "class_weight": "balanced"}`
- **random_forest**: `{"n_estimators": 250, "max_depth": 10, "min_samples_leaf": 3, "max_features": "log2"}`
- **xgboost**: `{"n_estimators": 250, "max_depth": 3, "learning_rate": 0.06874099203168502, "subsample": 0.6063382131333426, "colsample_bytree": 0.6386247785524612, "min_child_weight": 4}`

Note: the tuned CV score is slightly optimistic because the parameters were chosen on
the same folds. The test columns are the unbiased estimate.
