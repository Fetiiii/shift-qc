"""R8 repeat-first oracle-recalibrated recoverable utility endpoint."""

from __future__ import annotations

import numpy as np

from .bootstrap import (
    count_fold_inheritance_violations,
    inherited_fold_assignments,
    make_repeated_patient_folds,
    make_two_population_bootstrap,
)
from .common import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    CROSSFIT_FOLDS,
    CROSSFIT_SEEDS,
    PatientIds,
    align_named_vectors,
    percentile_interval,
)
from .recalibration import RecalibratorName, cross_fitted_recalibrated_predictions
from .skill import cross_fitted_median_constant, relative_mae_skill


def _repeat_r8(
    q_hat: np.ndarray,
    q_true: np.ndarray,
    folds: np.ndarray,
    recalibrator: RecalibratorName,
) -> float:
    recalibrated = cross_fitted_recalibrated_predictions(
        q_hat, q_true, folds, recalibrator=recalibrator
    )
    comparator = cross_fitted_median_constant(q_true, folds)
    return relative_mae_skill(recalibrated, comparator, q_true)


def _repeat_values(
    q_hat: np.ndarray,
    q_true: np.ndarray,
    assignments: np.ndarray,
    recalibrator: RecalibratorName,
) -> np.ndarray:
    return np.asarray(
        [
            _repeat_r8(q_hat, q_true, assignments[repeat], recalibrator)
            for repeat in range(assignments.shape[0])
        ],
        dtype=np.float64,
    )


def compute_r8(
    id_q_hat_ids: PatientIds,
    id_q_hat: np.ndarray,
    id_q_true_ids: PatientIds,
    id_q_true: np.ndarray,
    ood_q_hat_ids: PatientIds,
    ood_q_hat: np.ndarray,
    ood_q_true_ids: PatientIds,
    ood_q_true: np.ndarray,
    *,
    recalibrator: RecalibratorName = "affine_L1",
    seeds: tuple[int, ...] = CROSSFIT_SEEDS,
    n_splits: int = CROSSFIT_FOLDS,
    bootstrap_B: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Compute the frozen R8 primary or a prespecified sensitivity analysis."""
    id_ids, id_values = align_named_vectors(
        {"q_hat": (id_q_hat_ids, id_q_hat), "q_true": (id_q_true_ids, id_q_true)}
    )
    ood_ids, ood_values = align_named_vectors(
        {"q_hat": (ood_q_hat_ids, ood_q_hat), "q_true": (ood_q_true_ids, ood_q_true)}
    )
    id_folds = make_repeated_patient_folds(id_ids, seeds, n_splits=n_splits)
    ood_folds = make_repeated_patient_folds(ood_ids, seeds, n_splits=n_splits)
    id_repeat = _repeat_values(
        id_values["q_hat"], id_values["q_true"], id_folds.assignments, recalibrator
    )
    ood_repeat = _repeat_values(
        ood_values["q_hat"], ood_values["q_true"], ood_folds.assignments, recalibrator
    )
    delta_repeat = ood_repeat - id_repeat

    bootstrap = make_two_population_bootstrap(
        id_ids, ood_ids, seed=bootstrap_seed, resamples=bootstrap_B
    )
    checked = min(500, bootstrap_B)
    violations = count_fold_inheritance_violations(
        id_folds.assignments, bootstrap.id_population.indices[:checked]
    ) + count_fold_inheritance_violations(
        ood_folds.assignments, bootstrap.ood_population.indices[:checked]
    )
    if violations:
        raise RuntimeError("bootstrap copies violated fixed parent fold identities")

    bootstrap_delta = np.empty(bootstrap_B, dtype=np.float64)
    for replicate in range(bootstrap_B):
        id_parent = bootstrap.id_population.indices[replicate]
        ood_parent = bootstrap.ood_population.indices[replicate]
        id_inherited = inherited_fold_assignments(id_folds.assignments, id_parent)
        ood_inherited = inherited_fold_assignments(ood_folds.assignments, ood_parent)
        id_boot = _repeat_values(
            id_values["q_hat"][id_parent],
            id_values["q_true"][id_parent],
            id_inherited,
            recalibrator,
        )
        ood_boot = _repeat_values(
            ood_values["q_hat"][ood_parent],
            ood_values["q_true"][ood_parent],
            ood_inherited,
            recalibrator,
        )
        bootstrap_delta[replicate] = float(np.mean(ood_boot - id_boot, dtype=np.float64))

    ci_low, ci_high = percentile_interval(bootstrap_delta, confidence_level=CONFIDENCE_LEVEL)
    primary = recalibrator == "affine_L1"
    return {
        "endpoint": "R8",
        "role": "PRIMARY" if primary else "SENSITIVITY",
        "recalibrator": recalibrator,
        "R": len(seeds),
        "folds": int(n_splits),
        "seeds": list(seeds),
        "S_ID": float(np.mean(id_repeat, dtype=np.float64)),
        "S_OOD": float(np.mean(ood_repeat, dtype=np.float64)),
        "Delta8": float(np.mean(delta_repeat, dtype=np.float64)),
        "CI_low": ci_low,
        "CI_high": ci_high,
        "bootstrap_B": int(bootstrap_B),
        "bootstrap_seed": int(bootstrap_seed),
        "fold_integrity_checked_replicates": checked,
        "fold_integrity_violations": violations,
        "verdict": ("REPLICATED" if ci_high < 0.0 else "UNRESOLVED") if primary else None,
        "primary_verdict_eligible": primary,
    }
