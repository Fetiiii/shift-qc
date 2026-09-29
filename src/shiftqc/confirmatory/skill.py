"""Frozen relative-MAE skill and cross-fitted oracle constant."""

from __future__ import annotations

import numpy as np

from .common import ConfirmatoryInputError, finite_vector


def cross_fitted_median_constant(
    q_true: np.ndarray,
    fold_assignments: np.ndarray,
) -> np.ndarray:
    """Predict each held-out copy with the training-fold-only target median."""
    target = finite_vector(q_true, "q_true")
    folds = np.asarray(fold_assignments, dtype=np.int64)
    if folds.ndim != 1 or folds.size != target.size:
        raise ConfirmatoryInputError("fold_assignments must match q_true")
    unique_folds = np.unique(folds)
    if unique_folds.size < 2 or np.any(unique_folds < 0):
        raise ConfirmatoryInputError("cross-fitting requires at least two valid folds")

    predictions = np.full(target.size, np.nan, dtype=np.float64)
    for fold in unique_folds:
        held_out = folds == fold
        training = ~held_out
        if not held_out.any() or not training.any():
            raise ConfirmatoryInputError("every fold needs held-out and training patients")
        predictions[held_out] = float(np.median(target[training]))
    if not np.isfinite(predictions).all():
        raise RuntimeError("cross-fitted median did not produce a full OOF vector")
    return predictions


def relative_mae_skill(
    model_prediction: np.ndarray,
    comparator_prediction: np.ndarray,
    q_true: np.ndarray,
) -> float:
    """Compute S = 1 - MAE(model)/MAE(comparator), refusing zero denominator."""
    target = finite_vector(q_true, "q_true")
    model = finite_vector(model_prediction, "model_prediction", expected_size=target.size)
    comparator = finite_vector(
        comparator_prediction, "comparator_prediction", expected_size=target.size
    )
    model_mae = float(np.mean(np.abs(model - target), dtype=np.float64))
    comparator_mae = float(np.mean(np.abs(comparator - target), dtype=np.float64))
    if comparator_mae <= 0.0:
        raise ConfirmatoryInputError("comparator MAE is zero; skill estimand is undefined")
    skill = 1.0 - model_mae / comparator_mae
    if not np.isfinite(skill):
        raise ConfirmatoryInputError("skill estimand is non-finite")
    return float(skill)
