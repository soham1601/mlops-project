import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mlops_project.config import FEATURES
from mlops_project.features import RawCleaner, build_pipeline, clean_features


@pytest.fixture
def features(learnable_df) -> pd.DataFrame:
    return learnable_df[FEATURES].copy()


def test_valid_values_pass_through_unchanged(features):
    cleaned, issues = clean_features(features)
    pd.testing.assert_frame_equal(cleaned, features.astype("float64"))
    assert all(sum(c.values()) == 0 for c in issues.values())


def test_missing_markers_become_nan_and_are_counted(features):
    messy = features.astype(object)
    messy.loc[0, "chol"] = "?"
    messy.loc[1, "chol"] = " NA "
    messy.loc[2, "chol"] = ""
    cleaned, issues = clean_features(messy)
    assert cleaned["chol"].isna().sum() == 3
    assert issues["chol"]["marker"] == 3


def test_non_numeric_text_becomes_nan_and_is_counted(features):
    messy = features.astype(object)
    messy.loc[4, "age"] = "forty"
    cleaned, issues = clean_features(messy)
    assert np.isnan(cleaned.loc[4, "age"])
    assert issues["age"]["not_numeric"] == 1


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [("age", 250), ("chol", 0), ("trestbps", -1), ("sex", 2), ("cp", 9), ("thal", 5), ("ca", 7)],
)
def test_impossible_values_become_nan(features, column, bad_value):
    features.loc[0, column] = bad_value
    cleaned, issues = clean_features(features)
    assert np.isnan(cleaned.loc[0, column])
    assert issues[column]["invalid"] == 1
    assert cleaned[column].iloc[1:].notna().all()  # neighbours untouched


def test_missing_required_column_is_an_error_that_names_it(features):
    with pytest.raises(ValueError, match="chol"):
        clean_features(features.drop(columns="chol"))


def test_extra_columns_order_and_header_case_do_not_matter(features):
    scrambled = features[FEATURES[::-1]].copy()
    scrambled.columns = [f" {c.upper()} " for c in scrambled.columns]
    scrambled["site"] = "hungary"
    cleaned, _ = clean_features(scrambled)
    assert list(cleaned.columns) == FEATURES
    pd.testing.assert_frame_equal(cleaned, features.astype("float64"))


def test_text_and_numeric_inputs_give_identical_results(features):
    """Training reads CSV text; the API sends numbers. Both must clean to the same values."""
    from_text, _ = clean_features(features.astype(str))
    from_numbers, _ = clean_features(features)
    pd.testing.assert_frame_equal(from_text, from_numbers)


def test_cleaning_does_not_modify_the_input(features):
    messy = features.astype(object)
    messy.loc[0, "chol"] = "?"
    before = messy.copy()
    clean_features(messy)
    pd.testing.assert_frame_equal(before, messy)


def test_the_cleaner_stage_works_inside_sklearn(features):
    cleaner = RawCleaner().fit(features)
    assert list(cleaner.get_feature_names_out()) == FEATURES
    assert cleaner.transform(features).shape == features.shape


# --- the complete pipeline ---------------------------------------------------------------


@pytest.fixture
def fitted(learnable_df):
    train = learnable_df.iloc[:160]
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    return pipe.fit(train[FEATURES], train["target"])


def test_scaling_statistics_come_from_the_training_rows_only(learnable_df, fitted):
    scaler = fitted.named_steps["preprocess"].named_transformers_["num"]["scale"]
    numeric_cols = fitted.named_steps["preprocess"].transformers_[0][2]
    age = numeric_cols.index("age")
    assert scaler.mean_[age] == pytest.approx(learnable_df["age"].iloc[:160].mean())
    assert scaler.mean_[age] != pytest.approx(learnable_df["age"].mean())


def test_unseen_category_codes_do_not_crash_prediction(features, fitted):
    odd = features.iloc[:5].copy()
    odd["cp"] = 9
    odd["thal"] = 5
    proba = fitted.predict_proba(odd)[:, 1]
    assert np.isfinite(proba).all()


def test_a_row_with_every_value_missing_still_gets_a_prediction(features, fitted):
    empty = pd.DataFrame([dict.fromkeys(FEATURES, "?")])
    proba = fitted.predict_proba(empty)[:, 1]
    assert 0 <= proba[0] <= 1


def test_text_input_predicts_the_same_as_numeric_input(features, fitted):
    rows = features.iloc[:20]
    np.testing.assert_allclose(
        fitted.predict_proba(rows.astype(str)), fitted.predict_proba(rows), rtol=1e-9
    )
