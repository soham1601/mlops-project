# Feature experiments

Data: heart_clean.csv, 237 training rows (the held-out test rows were not used).
Protocol: 5-fold stratified CV repeated 5 times = 25 paired scores per variant, identical folds for all variants.
Primary metric: roc_auc. Models use default parameters (not tuned).
Differences: corrected resampled t-test, then Holm adjustment. A variant counts as better or worse only when the adjusted p < 0.05.

## logistic_regression

### Feature sets

| variant | features | CV roc_auc | change | 95% interval | p (Holm) | reading |
| --- | --- | --- | --- | --- | --- | --- |
| all_13 | 13 | 0.902 +/- 0.043 | +0.000 | [+0.000, +0.000] | 1.000 | reference |
| top_10 | 10 | 0.903 +/- 0.040 | +0.000 | [-0.011, +0.012] | 1.000 | no detectable difference (simpler) |
| top_8 | 8 | 0.900 +/- 0.043 | -0.002 | [-0.017, +0.013] | 1.000 | no detectable difference (simpler) |
| top_6 | 6 | 0.896 +/- 0.038 | -0.006 | [-0.029, +0.017] | 1.000 | no detectable difference (simpler) |
| plus_hr_ratio | 14 | 0.902 +/- 0.043 | +0.000 | [-0.001, +0.001] | 1.000 | no detectable difference |

### Dropping one feature at a time

| feature dropped | CV roc_auc | change | 95% interval | p (Holm) | reading |
| --- | --- | --- | --- | --- | --- |
| ca | 0.875 | -0.027 | [-0.048, -0.005] | 0.279 | no detectable difference (simpler) |
| cp | 0.878 | -0.024 | [-0.045, -0.002] | 0.523 | no detectable difference (simpler) |
| thal | 0.888 | -0.014 | [-0.037, +0.010] | 1.000 | no detectable difference (simpler) |
| sex | 0.897 | -0.005 | [-0.014, +0.004] | 1.000 | no detectable difference (simpler) |
| trestbps | 0.900 | -0.002 | [-0.011, +0.006] | 1.000 | no detectable difference (simpler) |
| slope | 0.901 | -0.002 | [-0.016, +0.013] | 1.000 | no detectable difference (simpler) |
| oldpeak | 0.901 | -0.001 | [-0.012, +0.010] | 1.000 | no detectable difference (simpler) |
| exang | 0.902 | +0.000 | [-0.008, +0.007] | 1.000 | no detectable difference (simpler) |
| chol | 0.903 | +0.001 | [-0.014, +0.015] | 1.000 | no detectable difference (simpler) |
| fbs | 0.903 | +0.001 | [-0.005, +0.007] | 1.000 | no detectable difference (simpler) |
| restecg | 0.903 | +0.001 | [-0.005, +0.007] | 1.000 | no detectable difference (simpler) |
| age | 0.904 | +0.002 | [-0.002, +0.005] | 1.000 | no detectable difference (simpler) |
| thalach | 0.905 | +0.003 | [-0.003, +0.009] | 1.000 | no detectable difference (simpler) |

## random_forest

### Feature sets

| variant | features | CV roc_auc | change | 95% interval | p (Holm) | reading |
| --- | --- | --- | --- | --- | --- | --- |
| all_13 | 13 | 0.898 +/- 0.035 | +0.000 | [+0.000, +0.000] | 1.000 | reference |
| top_10 | 10 | 0.894 +/- 0.031 | -0.004 | [-0.021, +0.013] | 1.000 | no detectable difference (simpler) |
| top_8 | 8 | 0.891 +/- 0.030 | -0.007 | [-0.023, +0.010] | 1.000 | no detectable difference (simpler) |
| top_6 | 6 | 0.887 +/- 0.031 | -0.011 | [-0.036, +0.013] | 1.000 | no detectable difference (simpler) |
| plus_hr_ratio | 14 | 0.892 +/- 0.033 | -0.006 | [-0.016, +0.004] | 1.000 | no detectable difference |

### Dropping one feature at a time

| feature dropped | CV roc_auc | change | 95% interval | p (Holm) | reading |
| --- | --- | --- | --- | --- | --- |
| ca | 0.874 | -0.024 | [-0.039, -0.009] | 0.048 | worse |
| cp | 0.879 | -0.019 | [-0.040, +0.001] | 0.997 | no detectable difference (simpler) |
| thal | 0.880 | -0.018 | [-0.047, +0.011] | 1.000 | no detectable difference (simpler) |
| sex | 0.894 | -0.004 | [-0.015, +0.008] | 1.000 | no detectable difference (simpler) |
| restecg | 0.894 | -0.004 | [-0.012, +0.005] | 1.000 | no detectable difference (simpler) |
| oldpeak | 0.894 | -0.004 | [-0.020, +0.013] | 1.000 | no detectable difference (simpler) |
| age | 0.896 | -0.002 | [-0.019, +0.015] | 1.000 | no detectable difference (simpler) |
| slope | 0.896 | -0.002 | [-0.023, +0.019] | 1.000 | no detectable difference (simpler) |
| chol | 0.896 | -0.002 | [-0.014, +0.010] | 1.000 | no detectable difference (simpler) |
| fbs | 0.897 | -0.001 | [-0.009, +0.008] | 1.000 | no detectable difference (simpler) |
| exang | 0.901 | +0.003 | [-0.010, +0.015] | 1.000 | no detectable difference (simpler) |
| trestbps | 0.901 | +0.003 | [-0.008, +0.015] | 1.000 | no detectable difference (simpler) |
| thalach | 0.905 | +0.007 | [-0.003, +0.017] | 1.000 | no detectable difference (simpler) |

## xgboost

### Feature sets

| variant | features | CV roc_auc | change | 95% interval | p (Holm) | reading |
| --- | --- | --- | --- | --- | --- | --- |
| all_13 | 13 | 0.873 +/- 0.034 | +0.000 | [+0.000, +0.000] | 1.000 | reference |
| top_10 | 10 | 0.862 +/- 0.037 | -0.011 | [-0.038, +0.017] | 1.000 | no detectable difference (simpler) |
| top_8 | 8 | 0.853 +/- 0.040 | -0.020 | [-0.049, +0.010] | 1.000 | no detectable difference (simpler) |
| top_6 | 6 | 0.856 +/- 0.043 | -0.016 | [-0.053, +0.021] | 1.000 | no detectable difference (simpler) |
| plus_hr_ratio | 14 | 0.875 +/- 0.033 | +0.002 | [-0.015, +0.019] | 1.000 | no detectable difference |

### Dropping one feature at a time

| feature dropped | CV roc_auc | change | 95% interval | p (Holm) | reading |
| --- | --- | --- | --- | --- | --- |
| ca | 0.832 | -0.041 | [-0.067, -0.015] | 0.053 | no detectable difference (simpler) |
| age | 0.847 | -0.026 | [-0.048, -0.003] | 0.408 | no detectable difference (simpler) |
| thal | 0.847 | -0.025 | [-0.051, +0.000] | 0.754 | no detectable difference (simpler) |
| cp | 0.852 | -0.021 | [-0.049, +0.007] | 1.000 | no detectable difference (simpler) |
| sex | 0.867 | -0.006 | [-0.020, +0.009] | 1.000 | no detectable difference (simpler) |
| oldpeak | 0.867 | -0.005 | [-0.028, +0.017] | 1.000 | no detectable difference (simpler) |
| restecg | 0.870 | -0.003 | [-0.016, +0.010] | 1.000 | no detectable difference (simpler) |
| chol | 0.871 | -0.002 | [-0.019, +0.015] | 1.000 | no detectable difference (simpler) |
| fbs | 0.872 | -0.001 | [-0.010, +0.008] | 1.000 | no detectable difference (simpler) |
| slope | 0.874 | +0.001 | [-0.016, +0.018] | 1.000 | no detectable difference (simpler) |
| exang | 0.874 | +0.001 | [-0.007, +0.010] | 1.000 | no detectable difference (simpler) |
| thalach | 0.885 | +0.012 | [-0.010, +0.035] | 1.000 | no detectable difference (simpler) |
| trestbps | 0.886 | +0.014 | [-0.005, +0.032] | 1.000 | no detectable difference (simpler) |

## Which features did selection keep?

Share of the folds in which each feature was among the top k (1.00 = always kept, 0.00 = never kept):

| feature | top_10 | top_8 | top_6 |
| --- | --- | --- | --- |
| ca | 1.00 | 1.00 | 1.00 |
| cp | 1.00 | 1.00 | 1.00 |
| oldpeak | 1.00 | 1.00 | 1.00 |
| thal | 1.00 | 1.00 | 1.00 |
| exang | 1.00 | 1.00 | 0.92 |
| slope | 1.00 | 1.00 | 0.56 |
| thalach | 1.00 | 0.88 | 0.48 |
| sex | 0.96 | 0.36 | 0.00 |
| restecg | 0.68 | 0.12 | 0.00 |
| chol | 0.52 | 0.24 | 0.00 |
| trestbps | 0.40 | 0.20 | 0.04 |
| age | 0.40 | 0.20 | 0.00 |
| fbs | 0.04 | 0.00 | 0.00 |

## How to read this

- **change** is the variant's score minus the all-features score, so a negative number means the variant did worse. The interval is for that change; if it contains 0 the data cannot separate the two.
- **no detectable difference** is not proof of equality. With a few hundred patients the test has little power, so small real effects can be missed.
- When two feature sets score the same, the simpler one is usually preferred: fewer inputs to collect, validate and explain.
- Dropping a feature that is correlated with another (for example oldpeak and slope) often changes nothing, because the other one carries the same information.
- Many variants are compared at once, which is why the p-values are Holm-adjusted. Treat any single unadjusted p-value with suspicion.
- The final model is still trained on the full feature set unless you change the pipeline yourself. These results inform that choice; they do not make it.
