"""Unit tests for frozen TASK-008 QC baseline behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shiftqc.qc.baselines import (
    QCBaselineError,
    build_ridge_search,
    fit_baselines,
    predict_all,
    regression_metrics,
    validate_feature_matrix,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (REPOSITORY / "configs/templates/qc_baselines.json").read_text()
)


def _synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(41)
    splits = ["QC_TRAIN"] * 30 + ["CALIBRATION"] * 8 + ["ID_TEST"] * 8 + ["OOD_DEV"] * 8
    records = []
    for index, split in enumerate(splits):
        features = rng.normal(size=15)
        target = 0.55 + 0.04 * features[0] - 0.02 * features[5]
        records.append(
            {
                "patient_id": f"patient_{index:03d}",
                "domain": "source" if split != "OOD_DEV" else "target",
                "split": split,
                **dict(zip(CONFIG["feature_columns"], features, strict=True)),
                "dice_macro": target,
            }
        )
    return pd.DataFrame.from_records(records)


def _synthetic_config() -> dict:
    config = json.loads(json.dumps(CONFIG))
    config["expected_counts"] = {
        "QC_TRAIN": 30,
        "CALIBRATION": 8,
        "ID_TEST": 8,
        "OOD_DEV": 8,
    }
    return config


def test_ridge_search_uses_scaler_inside_pipeline_and_exact_kfold() -> None:
    search = build_ridge_search(CONFIG)
    assert isinstance(search, GridSearchCV)
    assert isinstance(search.estimator, Pipeline)
    assert list(search.estimator.named_steps) == ["standard_scaler", "ridge"]
    assert isinstance(search.estimator.named_steps["standard_scaler"], StandardScaler)
    assert isinstance(search.cv, KFold)
    assert search.cv.n_splits == 5
    assert search.cv.shuffle is True
    assert search.cv.random_state == 2026
    assert search.scoring == "neg_mean_absolute_error"
    assert search.param_grid["ridge__alpha"] == CONFIG["qc1_ridge"]["alpha_grid"]


def test_feature_matrix_is_exactly_15_features_and_excludes_target() -> None:
    frame = _synthetic_frame()
    matrix = validate_feature_matrix(
        frame,
        CONFIG["feature_columns"],
        target="dice_macro",
    )
    assert matrix.shape == (54, 15)
    assert "dice_macro" not in CONFIG["feature_columns"]
    assert "patient_id" not in CONFIG["feature_columns"]
    assert "domain" not in CONFIG["feature_columns"]
    assert "split" not in CONFIG["feature_columns"]


@pytest.mark.parametrize(
    "forbidden",
    ["dice_macro", "hd95", "patient_id", "domain", "split"],
)
def test_forbidden_feature_columns_are_rejected(forbidden: str) -> None:
    frame = _synthetic_frame()
    columns = list(CONFIG["feature_columns"])
    columns[-1] = forbidden
    if forbidden not in frame:
        frame[forbidden] = 0.0
    with pytest.raises(QCBaselineError, match="forbidden"):
        validate_feature_matrix(frame, columns, target="dice_macro")


def test_fit_patient_ids_are_only_qc_train() -> None:
    frame = _synthetic_frame()
    fitted = fit_baselines(frame, _synthetic_config())
    expected = set(frame.loc[frame["split"].eq("QC_TRAIN"), "patient_id"])
    assert set(fitted.fit_patient_ids) == expected
    assert len(fitted.fit_patient_ids) == 30
    assert fitted.ridge_cv["n_splits"] == 5
    assert len(fitted.ridge_cv["fold_cv_maes"]) == 5
    assert fitted.ridge_cv["scaler_fit_inside_pipeline"] is True


@pytest.mark.parametrize("held_out_split", ["CALIBRATION", "ID_TEST", "OOD_DEV"])
def test_held_out_labels_do_not_influence_any_fitted_model(
    held_out_split: str,
) -> None:
    original = _synthetic_frame()
    altered = original.copy()
    altered.loc[altered["split"].eq(held_out_split), "dice_macro"] += 1000.0
    config = _synthetic_config()
    original_fit = fit_baselines(original, config)
    altered_fit = fit_baselines(altered, config)
    original_predictions = predict_all(original, original_fit, config)
    altered_predictions = predict_all(original, altered_fit, config)
    assert original_fit.ridge_cv == altered_fit.ridge_cv
    for column in ("qc0_pred", "qc1_ridge_pred", "qc2_hgb_pred"):
        np.testing.assert_allclose(
            original_predictions[column],
            altered_predictions[column],
            rtol=0,
            atol=0,
        )


def test_all_baselines_are_deterministic_under_seed_2026() -> None:
    frame = _synthetic_frame()
    config = _synthetic_config()
    first_fit = fit_baselines(frame, config)
    second_fit = fit_baselines(frame, config)
    first = predict_all(frame, first_fit, config)
    second = predict_all(frame, second_fit, config)
    assert first_fit.ridge_cv == second_fit.ridge_cv
    for column in ("qc0_pred", "qc1_ridge_pred", "qc2_hgb_pred"):
        np.testing.assert_array_equal(first[column], second[column])


def test_qc0_prediction_is_qc_train_target_mean_everywhere() -> None:
    frame = _synthetic_frame()
    config = _synthetic_config()
    fitted = fit_baselines(frame, config)
    predictions = predict_all(frame, fitted, config)
    expected = frame.loc[frame["split"].eq("QC_TRAIN"), "dice_macro"].mean()
    np.testing.assert_allclose(predictions["qc0_pred"], expected)


def test_constant_prediction_correlations_are_explicitly_undefined() -> None:
    target = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    prediction = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    metrics = regression_metrics(target, prediction)
    assert metrics["spearman"] == {
        "value": None,
        "defined": False,
        "reason": "constant_prediction",
    }
    assert metrics["pearson"] == {
        "value": None,
        "defined": False,
        "reason": "constant_prediction",
    }
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["rmse"])
