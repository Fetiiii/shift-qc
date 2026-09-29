"""TASK-022/023 — frozen M&Ms QC model, q-hat, and deployed calibration.

Construction only. Nothing here evaluates anything: no skill, no correlation, no
coverage, no cohort comparison. It builds the exact frozen predictor and the exact
deployed calibration procedure so that the confirmatory readout, when it is opened,
reads artefacts that were fixed before any result was seen.

Firewalls are structural rather than conventional, and the negative tests in
``tests/test_task022_023_m3_deployment.py`` attack them deliberately.

* :func:`fit_qc_model` takes a feature frame and a target vector that the caller
  has already restricted to ``QC_TRAIN``. It re-asserts that restriction and
  refuses anything else, so an evaluation label cannot reach the fit even by a
  caller mistake.
* :func:`predict_q_hat` takes features only. There is no parameter through which
  ``q_true`` could enter, so a prediction cannot depend on a label.
* :func:`fit_deployed_conformal` takes calibration residuals derived from
  ``CALIBRATION`` alone and re-asserts the population, so evaluation residuals
  cannot move the threshold.

Every rule is read from the frozen configuration; none is defined here. The
estimator comes from ``shiftqc.qc.baselines`` (TASK-008) and the calibration from
``shiftqc.conformal.split`` (TASK-009).
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from shiftqc.conformal.split import (
    absolute_residuals,
    conformal_residual_threshold,
    construct_intervals,
    finite_sample_rank,
)
from shiftqc.qc.baselines import build_ridge_search, validate_feature_matrix
from shiftqc.qc.m3_inputs import (
    EXPECTED_POPULATION_PATIENTS,
    FROZEN_FEATURE_ORDER,
    QC_POPULATIONS,
)

TRAINING_POPULATION = "QC_TRAIN"
CALIBRATION_POPULATION = "CALIBRATION"


class Task022Failure(RuntimeError):
    """Raised when a TASK-022/023 input violates the frozen design. Never repaired."""

    def __init__(self, code: str, details: Any = None) -> None:
        super().__init__(f"{code}: {details}" if details is not None else code)
        self.code = code
        self.details = details


def feature_columns_sha256(columns: Sequence[str]) -> str:
    """Hash the canonical feature order so a silent reorder cannot pass unnoticed."""
    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()


def _require_single_population(frame: pd.DataFrame, population: str, code: str) -> None:
    if "population" not in frame.columns:
        raise Task022Failure("population_column_missing", code)
    observed = sorted(set(frame["population"].astype(str)))
    if observed != [population]:
        raise Task022Failure(code, {"expected": [population], "observed": observed})
    if bool(frame["patient_id"].duplicated().any()):
        raise Task022Failure("duplicate_patient_id",
                             sorted(frame.loc[frame["patient_id"].duplicated(), "patient_id"]))
    expected = EXPECTED_POPULATION_PATIENTS[population]
    if len(frame) != expected:
        raise Task022Failure("unexpected_population_patient_count",
                             {"population": population, "expected": expected,
                              "observed": int(len(frame))})


def _feature_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Enforce the canonical order exactly; never reorder silently."""
    listed = list(columns)
    if listed != list(FROZEN_FEATURE_ORDER):
        raise Task022Failure("feature_order_mismatch",
                             {"expected": list(FROZEN_FEATURE_ORDER), "observed": listed})
    present = [c for c in frame.columns if c in set(listed)]
    if present != listed:
        raise Task022Failure("feature_columns_out_of_canonical_order",
                             {"expected": listed, "observed": present})
    return validate_feature_matrix(frame, listed, target="dice_macro")


# ------------------------------------------------------------------ TASK-022

def fit_qc_model(
    features_qc_train: pd.DataFrame,
    q_true_qc_train: pd.Series,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit the frozen StandardScaler-to-Ridge on QC_TRAIN alone.

    The caller supplies the QC_TRAIN subset; this function re-asserts it. Passing
    any other population, or a frame carrying more than one, is a hard failure.
    """
    _require_single_population(features_qc_train, TRAINING_POPULATION,
                               "fit_population_not_qc_train")
    if list(q_true_qc_train.index) != list(features_qc_train.index):
        raise Task022Failure("feature_target_index_mismatch", None)
    target = np.asarray(q_true_qc_train, dtype=np.float64)
    if target.ndim != 1 or not np.isfinite(target).all():
        raise Task022Failure("non_finite_training_target", None)
    if target.size != EXPECTED_POPULATION_PATIENTS[TRAINING_POPULATION]:
        raise Task022Failure("unexpected_training_target_count", int(target.size))

    matrix = _feature_matrix(features_qc_train, FROZEN_FEATURE_ORDER)
    search = build_ridge_search(dict(config))
    search.fit(matrix, target)

    pipeline = search.best_estimator_
    scaler = pipeline.named_steps["standard_scaler"]
    ridge = pipeline.named_steps["ridge"]
    return {
        "estimator": pipeline,
        "selected_alpha": float(search.best_params_["ridge__alpha"]),
        "feature_columns": list(FROZEN_FEATURE_ORDER),
        "feature_columns_sha256": feature_columns_sha256(FROZEN_FEATURE_ORDER),
        "training_population": TRAINING_POPULATION,
        "n_train": int(target.size),
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "ridge_coefficients": [float(v) for v in np.ravel(ridge.coef_)],
        "ridge_intercept": float(np.ravel(ridge.intercept_)[0]),
        "cv": dict(config["qc1_ridge"]["cv"]),
        "alpha_grid": [float(a) for a in config["qc1_ridge"]["alpha_grid"]],
    }


def predict_q_hat(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Predict from features alone. No parameter admits a label.

    The frozen TASK-008 configuration sets ``prediction_clipping`` false, so the
    raw Ridge output is returned unclipped.
    """
    matrix = _feature_matrix(features, FROZEN_FEATURE_ORDER)
    predictions = np.asarray(model.predict(matrix), dtype=np.float64)
    if predictions.ndim != 1 or predictions.size != len(features):
        raise Task022Failure("unexpected_prediction_shape", predictions.shape)
    if not np.isfinite(predictions).all():
        raise Task022Failure("non_finite_prediction", None)
    return predictions


# ------------------------------------------------------------------ TASK-023

def fit_deployed_conformal(
    calibration: pd.DataFrame,
    *,
    alpha_levels: Mapping[str, float],
) -> dict[str, Any]:
    """Fit the deployed absolute-residual threshold on CALIBRATION alone.

    ``calibration`` must carry ``patient_id``, ``population``, ``q_true`` and
    ``q_hat`` for the CALIBRATION cohort and nothing else. Evaluation residuals
    cannot reach the threshold, which the negative tests assert directly.
    """
    _require_single_population(calibration, CALIBRATION_POPULATION,
                               "calibration_population_not_calibration")
    residuals = absolute_residuals(
        np.asarray(calibration["q_true"], dtype=np.float64),
        np.asarray(calibration["q_hat"], dtype=np.float64),
    )
    levels = {}
    for name, alpha in alpha_levels.items():
        threshold, rank = conformal_residual_threshold(residuals, float(alpha))
        levels[name] = {"alpha": float(alpha), "finite_sample_rank": int(rank),
                        "residual_threshold": float(threshold)}
    return {
        "calibration_population": CALIBRATION_POPULATION,
        "n_calibration": int(len(calibration)),
        "residual_definition": "abs(q_true - q_hat)",
        "quantile_convention": "one-based order statistic, ceil((n+1)(1-alpha)), no interpolation",
        "levels": levels,
        "residual_sha256": hashlib.sha256(
            np.ascontiguousarray(np.sort(residuals)).tobytes()).hexdigest(),
    }


def deployed_intervals(q_hat: np.ndarray, residual_threshold: float) -> dict[str, np.ndarray]:
    """Frozen interval construction: q_hat +/- threshold, then physical clipping."""
    return construct_intervals(np.asarray(q_hat, dtype=np.float64), float(residual_threshold))


def unconditional_source_interval(
    calibration_q_true: np.ndarray,
    *,
    alpha_levels: Mapping[str, float],
) -> dict[str, Any]:
    """The source-calibrated comparator: a CALIBRATION true-Dice quantile interval.

    Built here so that the R6 Case A comparator is fixed before any endpoint runs.
    It is transferred unchanged to the evaluation populations; nothing is evaluated
    with it in this task.
    """
    values = np.asarray(calibration_q_true, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise Task022Failure("non_finite_calibration_target", None)
    levels = {}
    for name, alpha in alpha_levels.items():
        a = float(alpha)
        lower = float(np.quantile(values, a / 2.0, method="linear"))
        upper = float(np.quantile(values, 1.0 - a / 2.0, method="linear"))
        levels[name] = {
            "alpha": a,
            "lower_unclipped": lower, "upper_unclipped": upper,
            "lower_clipped": float(np.clip(lower, 0.0, 1.0)),
            "upper_clipped": float(np.clip(upper, 0.0, 1.0)),
        }
    return {
        "source_population": CALIBRATION_POPULATION,
        "n_calibration": int(values.size),
        "definition": "CALIBRATION true-Dice quantile interval, transferred unchanged",
        "quantile_method": "numpy linear interpolation",
        "levels": levels,
    }
