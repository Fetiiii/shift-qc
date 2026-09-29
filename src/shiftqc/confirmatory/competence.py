"""Baseline ID ranking-competence precondition (D-020)."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .bootstrap import make_patient_bootstrap
from .common import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    ConfirmatoryInputError,
    PatientIds,
    align_named_vectors,
    percentile_interval,
)


def spearman_rho(q_hat: np.ndarray, q_true: np.ndarray) -> float:
    """Compute tied-rank Pearson correlation, hard-failing undefined cases."""
    prediction = np.asarray(q_hat, dtype=np.float64)
    target = np.asarray(q_true, dtype=np.float64)
    if prediction.ndim != 1 or target.shape != prediction.shape or prediction.size < 2:
        raise ConfirmatoryInputError("Spearman requires matching vectors with n >= 2")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ConfirmatoryInputError("Spearman inputs must be finite")
    if np.unique(prediction).size < 2:
        raise ConfirmatoryInputError("constant q_hat: Spearman estimand is undefined")
    if np.unique(target).size < 2:
        raise ConfirmatoryInputError("constant q_true: Spearman estimand is undefined")
    prediction_rank = rankdata(prediction, method="average")
    target_rank = rankdata(target, method="average")
    rho = float(np.corrcoef(prediction_rank, target_rank)[0, 1])
    if not np.isfinite(rho):
        raise ConfirmatoryInputError("Spearman estimand is undefined")
    return rho


def compute_baseline_competence(
    q_hat_patient_ids: PatientIds,
    q_hat: np.ndarray,
    q_true_patient_ids: PatientIds,
    q_true: np.ndarray,
    *,
    bootstrap_B: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Return the frozen ID-only Spearman estimate, percentile CI, and verdict."""
    patient_ids, aligned = align_named_vectors(
        {
            "q_hat": (q_hat_patient_ids, q_hat),
            "q_true": (q_true_patient_ids, q_true),
        }
    )
    rho = spearman_rho(aligned["q_hat"], aligned["q_true"])
    bootstrap = make_patient_bootstrap(
        patient_ids, seed=bootstrap_seed, resamples=bootstrap_B
    )
    values = np.empty(bootstrap_B, dtype=np.float64)
    for replicate, parents in enumerate(bootstrap.indices):
        try:
            values[replicate] = spearman_rho(
                aligned["q_hat"][parents], aligned["q_true"][parents]
            )
        except ConfirmatoryInputError as error:
            raise ConfirmatoryInputError(
                f"bootstrap replicate {replicate} has undefined Spearman estimand"
            ) from error
    ci_low, ci_high = percentile_interval(values, confidence_level=CONFIDENCE_LEVEL)
    return {
        "endpoint": "baseline_competence",
        "population": "M3_ID_EVAL",
        "rho": rho,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(patient_ids.size),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_B": int(bootstrap_B),
        "verdict": "RESOLVED" if ci_low > 0.0 else "UNRESOLVED",
    }
