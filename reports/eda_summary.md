# EDA summary

- Rows: 297, columns: 14
- Missing values: 0

## Class balance
- No disease: 160 (53.9%)
- Disease: 137 (46.1%)

## Correlation with the target (Pearson, strongest first)
- thal: +0.527
- ca: +0.463
- oldpeak: +0.424
- thalach: -0.424
- exang: +0.421
- cp: +0.409
- slope: +0.333
- sex: +0.278
- age: +0.227
- restecg: +0.166
- trestbps: +0.153
- chol: +0.080
- fbs: +0.003

## Outliers by the 1.5 x IQR rule (numeric features)
- age: 0
- trestbps: 9
- chol: 5
- thalach: 1
- oldpeak: 5

## Numeric summary
```
          age  trestbps    chol  thalach  oldpeak
count  297.00    297.00  297.00   297.00   297.00
mean    54.54    131.69  247.35   149.60     1.06
std      9.05     17.76   52.00    22.94     1.17
min     29.00     94.00  126.00    71.00     0.00
25%     48.00    120.00  211.00   133.00     0.00
50%     56.00    130.00  243.00   153.00     0.80
75%     61.00    140.00  276.00   166.00     1.60
max     77.00    200.00  564.00   202.00     6.20
```
