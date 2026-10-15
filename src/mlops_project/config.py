"""Project-wide constants: paths, column names, valid value ranges and plot labels."""

from __future__ import annotations

import os
from pathlib import Path

# Repository root. Override with MLOPS_ROOT when the package is installed elsewhere (Docker).
ROOT = Path(os.environ.get("MLOPS_ROOT", Path(__file__).resolve().parents[2]))

RAW_PATH = ROOT / "data" / "raw" / "heart_disease_raw.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "heart_clean.csv"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Columns -----------------------------------------------------------------------------
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
            "oldpeak", "slope", "ca", "thal"]  # fmt: skip

RAW_TARGET = "num"  # 0 = no disease, 1-4 = disease severity
TARGET = "target"  # binary: 0 = no disease, 1 = disease
TARGET_LABELS = {0: "No disease", 1: "Disease"}

# --- Data dictionary (from the UCI documentation) ---------------------------------------
DESCRIPTIONS = {
    "age": "Age (years)",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg": "Resting ECG result (0 normal, 1 ST-T abnormality, 2 LV hypertrophy)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak exercise ST segment (1 up, 2 flat, 3 down)",
    "ca": "Number of major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thalassemia (3 normal, 6 fixed defect, 7 reversible defect)",
}

# --- Validation rules (used by preprocess.validate) -------------------------------------
VALID_RANGES = {
    "age": (18, 100),
    "trestbps": (60, 250),
    "chol": (80, 700),
    "thalach": (50, 250),
    "oldpeak": (0, 10),
}
VALID_CATEGORIES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}
