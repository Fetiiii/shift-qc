"""Frozen QC-0, QC-1, and QC-2 baseline definitions for TASK-008."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class QCBaselineError(ValueError):
    """Raised when frozen TASK-008 fitting or evaluation contracts fail."""


@dataclass(frozen=True)
class FittedBaselines:
    """Fitted baseline estimators plus frozen Ridge CV provenance."""

    qc0: DummyRegressor
    qc1_ridge: Pipeline
    qc2_hgb: HistGradientBoostingRegressor
    ridge_cv: dict[str, Any]
    fit_patient_ids: tuple[str, ...]


def validate_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    target: str,
) -> np.ndarray:
    """Select exactly the frozen feature list and exclude identifiers/targets."""
    if len(feature_columns) != 15 or len(set(feature_columns)) != 15:
        raise QCBaselineError("TASK-008 requires exactly 15 unique feature columns")
    forbidden = {
        target,
        "patient_id",
        "domain",
        "split",
        "dice_macro",
        "hd95",
    }
    overlap = sorted(set(feature_columns) & forbidden)
    if overlap:
        raise QCBaselineError(f"forbidden model input columns: {overlap}")
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise QCBaselineError(f"missing feature columns: {missing}")
    matrix = frame.loc[:, feature_columns].to_numpy(dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 15:
        raise QCBaselineError(f"unexpected feature matrix shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise QCBaselineError("feature matrix contains NaN or infinity")
    return matrix


def build_ridge_search(config: dict[str, Any]) -> GridSearchCV:
    """Build the exact frozen Pipeline and QC_TRAIN-only CV search."""
    ridge_config = config["qc1_ridge"]
    cv_config = ridge_config["cv"]
    pipeline = Pipeline(
        steps=[
            ("standard_scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )
    cross_validation = KFold(
        n_splits=int(cv_config["n_splits"]),
        shuffle=bool(cv_config["shuffle"]),
        random_state=int(cv_config["random_state"]),
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid={"ridge__alpha": [float(value) for value in ridge_config["alpha_grid"]]},
        scoring=str(cv_config["scoring"]),
        cv=cross_validation,
        refit=True,
        return_train_score=False,
        n_jobs=1,
        error_score="raise",
    )


def _ridge_cv_provenance(search: GridSearchCV) -> dict[str, Any]:
    best_index = int(search.best_index_)
    fold_maes = [
        float(-search.cv_results_[f"split{fold}_test_score"][best_index])
        for fold in range(search.n_splits_)
    ]
    grid_results = []
    for index, parameters in enumerate(search.cv_results_["params"]):
        grid_results.append(
            {
                "alpha": float(parameters["ridge__alpha"]),
                "mean_cv_mae": float(-search.cv_results_["mean_test_score"][index]),
                "standard_deviation_cv_mae": float(
                    search.cv_results_["std_test_score"][index]
                ),
                "rank": int(search.cv_results_["rank_test_score"][index]),
            }
        )
    return {
        "selected_alpha": float(search.best_params_["ridge__alpha"]),
        "mean_cv_mae": float(np.mean(fold_maes, dtype=np.float64)),
        "fold_cv_maes": fold_maes,
        "n_splits": int(search.n_splits_),
        "shuffle": True,
        "random_state": 2026,
        "scoring": "negative mean absolute error",
        "grid_results": grid_results,
        "scaler_fit_inside_pipeline": True,
    }


def fit_baselines(joined: pd.DataFrame, config: dict[str, Any]) -> FittedBaselines:
    """Fit every QC baseline using QC_TRAIN rows only."""
    target = str(config["target"])
    fit_split = str(config["fit_split"])
    train = joined.loc[joined["split"].eq(fit_split)].copy()
    expected_train_count = int(config["expected_counts"][fit_split])
    if len(train) != expected_train_count:
        raise QCBaselineError(
            f"QC_TRAIN count mismatch: expected={expected_train_count}, observed={len(train)}"
        )
    if target not in train.columns:
        raise QCBaselineError(f"missing regression target: {target}")
    target_values = train[target].to_numpy(dtype=np.float64)
    if not np.isfinite(target_values).all():
        raise QCBaselineError("QC_TRAIN target contains NaN or infinity")
    features = validate_feature_matrix(
        train,
        list(config["feature_columns"]),
        target=target,
    )

    qc0 = DummyRegressor(strategy="mean")
    qc0.fit(features, target_values)

    ridge_search = build_ridge_search(config)
    ridge_search.fit(features, target_values)
    qc1_ridge = ridge_search.best_estimator_

    hgb_config = config["qc2_hgb"]
    qc2_hgb = HistGradientBoostingRegressor(
        loss=str(hgb_config["loss"]),
        learning_rate=float(hgb_config["learning_rate"]),
        max_iter=int(hgb_config["max_iter"]),
        max_leaf_nodes=int(hgb_config["max_leaf_nodes"]),
        min_samples_leaf=int(hgb_config["min_samples_leaf"]),
        l2_regularization=float(hgb_config["l2_regularization"]),
        random_state=int(hgb_config["random_state"]),
    )
    qc2_hgb.fit(features, target_values)

    return FittedBaselines(
        qc0=qc0,
        qc1_ridge=qc1_ridge,
        qc2_hgb=qc2_hgb,
        ridge_cv=_ridge_cv_provenance(ridge_search),
        fit_patient_ids=tuple(str(value) for value in train["patient_id"]),
    )


def _correlation(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    rank: bool,
) -> dict[str, Any]:
    if np.all(target == target[0]):
        return {
            "value": None,
            "defined": False,
            "reason": "constant_target",
        }
    if np.all(prediction == prediction[0]):
        return {
            "value": None,
            "defined": False,
            "reason": "constant_prediction",
        }
    first = rankdata(target) if rank else target
    second = rankdata(prediction) if rank else prediction
    centered_first = first - np.mean(first)
    centered_second = second - np.mean(second)
    denominator = float(
        np.sqrt(
            np.sum(centered_first * centered_first)
            * np.sum(centered_second * centered_second)
        )
    )
    if denominator == 0.0:
        return {
            "value": None,
            "defined": False,
            "reason": "zero_variance_after_transformation",
        }
    value = float(np.sum(centered_first * centered_second) / denominator)
    return {"value": value, "defined": True, "reason": None}


def regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    """Compute descriptive regression metrics without hypothesis tests."""
    target_array = np.asarray(target, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    if target_array.shape != prediction_array.shape or target_array.ndim != 1:
        raise QCBaselineError(
            f"metric input shape mismatch: target={target_array.shape}, "
            f"prediction={prediction_array.shape}"
        )
    if not np.isfinite(target_array).all() or not np.isfinite(prediction_array).all():
        raise QCBaselineError("regression metrics require finite inputs")
    residual = prediction_array - target_array
    return {
        "n": int(target_array.size),
        "mae": float(np.mean(np.abs(residual), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(residual * residual, dtype=np.float64))),
        "spearman": _correlation(target_array, prediction_array, rank=True),
        "pearson": _correlation(target_array, prediction_array, rank=False),
    }


def predict_all(
    joined: pd.DataFrame,
    fitted: FittedBaselines,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Generate raw, unclipped predictions for every authorized patient."""
    features = validate_feature_matrix(
        joined,
        list(config["feature_columns"]),
        target=str(config["target"]),
    )
    predictions = joined[["patient_id", "domain", "split", config["target"]]].copy()
    predictions["qc0_pred"] = fitted.qc0.predict(features)
    predictions["qc1_ridge_pred"] = fitted.qc1_ridge.predict(features)
    predictions["qc2_hgb_pred"] = fitted.qc2_hgb.predict(features)
    predictions["qc0_abs_error"] = np.abs(
        predictions["qc0_pred"] - predictions[config["target"]]
    )
    predictions["qc1_ridge_abs_error"] = np.abs(
        predictions["qc1_ridge_pred"] - predictions[config["target"]]
    )
    predictions["qc2_hgb_abs_error"] = np.abs(
        predictions["qc2_hgb_pred"] - predictions[config["target"]]
    )
    numeric = predictions.iloc[:, 3:].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise QCBaselineError("QC prediction artifact contains NaN or infinity")
    return predictions.sort_values(
        ["split", "patient_id"], kind="stable", ignore_index=True
    )
