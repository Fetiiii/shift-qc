"""Frozen cross-fitted recalibrators for R8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .common import (
    IRLS_EPSILON,
    IRLS_ITERATIONS,
    ConfirmatoryInputError,
    finite_vector,
)


RecalibratorName = Literal["affine_L1", "affine_OLS", "isotonic"]


@dataclass(frozen=True)
class AffineFit:
    intercept: float
    slope: float
    objective_l1: float

    def predict(self, q_hat: np.ndarray) -> np.ndarray:
        values = finite_vector(q_hat, "q_hat")
        return self.intercept + self.slope * values


def fit_affine_l1_irls(
    q_hat: np.ndarray,
    q_true: np.ndarray,
    *,
    iterations: int = IRLS_ITERATIONS,
    eps: float = IRLS_EPSILON,
) -> AffineFit:
    """Fit affine LAD by the frozen deterministic 80-step IRLS procedure."""
    prediction = finite_vector(q_hat, "q_hat")
    target = finite_vector(q_true, "q_true", expected_size=prediction.size)
    if prediction.size < 2:
        raise ConfirmatoryInputError("affine L1 requires at least two training patients")
    if iterations <= 0 or not np.isfinite(eps) or eps <= 0.0:
        raise ConfirmatoryInputError("IRLS iterations and eps must be positive")

    design = np.column_stack((np.ones(prediction.size), prediction))
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    for _ in range(iterations):
        residual = target - design @ coefficients
        weights = 1.0 / np.maximum(np.abs(residual), eps)
        sqrt_weights = np.sqrt(weights)
        coefficients = np.linalg.lstsq(
            design * sqrt_weights[:, None], target * sqrt_weights, rcond=None
        )[0]

    fitted = design @ coefficients
    objective = float(np.sum(np.abs(target - fitted), dtype=np.float64))
    if not np.isfinite(coefficients).all() or not np.isfinite(objective):
        raise ConfirmatoryInputError("affine L1 fit is non-finite")
    return AffineFit(float(coefficients[0]), float(coefficients[1]), objective)


def fit_affine_ols(q_hat: np.ndarray, q_true: np.ndarray) -> AffineFit:
    """Fit the frozen affine-OLS sensitivity model."""
    prediction = finite_vector(q_hat, "q_hat")
    target = finite_vector(q_true, "q_true", expected_size=prediction.size)
    if prediction.size < 2:
        raise ConfirmatoryInputError("affine OLS requires at least two training patients")
    design = np.column_stack((np.ones(prediction.size), prediction))
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    fitted = design @ coefficients
    objective = float(np.sum(np.abs(target - fitted), dtype=np.float64))
    return AffineFit(float(coefficients[0]), float(coefficients[1]), objective)


def cross_fitted_recalibrated_predictions(
    q_hat: np.ndarray,
    q_true: np.ndarray,
    fold_assignments: np.ndarray,
    *,
    recalibrator: RecalibratorName,
) -> np.ndarray:
    """Fit training-fold-only recalibrators and return a full OOF vector."""
    prediction = finite_vector(q_hat, "q_hat")
    target = finite_vector(q_true, "q_true", expected_size=prediction.size)
    folds = np.asarray(fold_assignments, dtype=np.int64)
    if folds.ndim != 1 or folds.size != prediction.size:
        raise ConfirmatoryInputError("fold_assignments must match q_hat")
    unique_folds = np.unique(folds)
    if unique_folds.size < 2 or np.any(unique_folds < 0):
        raise ConfirmatoryInputError("cross-fitting requires at least two valid folds")

    output = np.full(prediction.size, np.nan, dtype=np.float64)
    for fold in unique_folds:
        held_out = folds == fold
        training = ~held_out
        if not held_out.any() or np.count_nonzero(training) < 2:
            raise ConfirmatoryInputError("each recalibration fold needs held-out and training data")
        if recalibrator == "affine_L1":
            model = fit_affine_l1_irls(prediction[training], target[training])
            output[held_out] = model.predict(prediction[held_out])
        elif recalibrator == "affine_OLS":
            model = fit_affine_ols(prediction[training], target[training])
            output[held_out] = model.predict(prediction[held_out])
        elif recalibrator == "isotonic":
            # Boundary extrapolation is part of the isotonic predictor itself; no
            # post-hoc clipping is applied to affine or isotonic outputs.
            model = IsotonicRegression(increasing=True, out_of_bounds="clip")
            model.fit(prediction[training], target[training])
            output[held_out] = model.predict(prediction[held_out])
        else:
            raise ConfirmatoryInputError(f"unknown recalibrator: {recalibrator}")
    if not np.isfinite(output).all():
        raise ConfirmatoryInputError("recalibrator did not produce finite full OOF output")
    return output
