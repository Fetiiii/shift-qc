"""Frozen StandardScaler-to-Ridge QC model selection contract."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import FEATURE_COLUMNS, ContractError, validate_feature_columns


RIDGE_ALPHA_GRID = (
    0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0,
    3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0,
)
QC_CV_FOLDS = 5
QC_CV_SEED = 2026


def _feature_matrix(values: np.ndarray, columns: Sequence[str]) -> np.ndarray:
    validate_feature_columns(columns)
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_COLUMNS) or matrix.shape[0] == 0:
        raise ContractError("QC features must be a non-empty n x 18 matrix")
    if not np.isfinite(matrix).all():
        raise ContractError("QC features contain NaN or infinity")
    return matrix


def fit_qc_ridge(
    features: np.ndarray,
    q_true: np.ndarray,
    patient_ids: Sequence[str],
    roles: Sequence[str],
    *,
    columns: Sequence[str] = FEATURE_COLUMNS,
) -> GridSearchCV:
    """Select alpha by QC_TRAIN-only five-fold MAE CV and refit deterministically."""

    matrix = _feature_matrix(features, columns)
    target = np.asarray(q_true, dtype=np.float64)
    ids = tuple(str(value) for value in patient_ids)
    role_values = tuple(str(value) for value in roles)
    if target.ndim != 1 or target.size != matrix.shape[0] or not np.isfinite(target).all():
        raise ContractError("q_true must be a finite vector aligned to QC features")
    if np.any((target < 0.0) | (target > 1.0)):
        raise ContractError("q_true must lie in [0, 1]")
    if len(ids) != matrix.shape[0] or len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ContractError("QC_TRAIN patient IDs must be unique and aligned")
    if len(role_values) != matrix.shape[0] or set(role_values) != {"QC_TRAIN"}:
        raise ContractError("QC model fitting accepts QC_TRAIN rows only")
    if matrix.shape[0] < QC_CV_FOLDS:
        raise ContractError("QC_TRAIN has fewer patients than CV folds")

    pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    cv = KFold(n_splits=QC_CV_FOLDS, shuffle=True, random_state=QC_CV_SEED)
    search = GridSearchCV(
        pipeline,
        {"ridge__alpha": list(RIDGE_ALPHA_GRID)},
        scoring="neg_mean_absolute_error",
        cv=cv,
        refit=True,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(matrix, target)
    if float(search.best_params_["ridge__alpha"]) not in RIDGE_ALPHA_GRID:
        raise RuntimeError("GridSearchCV selected an alpha outside the frozen grid")
    return search


def predict_qc(
    fitted: GridSearchCV,
    features: np.ndarray,
    *,
    columns: Sequence[str] = FEATURE_COLUMNS,
) -> np.ndarray:
    """Return finite, deliberately unclipped QC predictions."""

    matrix = _feature_matrix(features, columns)
    prediction = np.asarray(fitted.predict(matrix), dtype=np.float64)
    if prediction.shape != (matrix.shape[0],) or not np.isfinite(prediction).all():
        raise ContractError("QC model produced invalid predictions")
    return prediction
