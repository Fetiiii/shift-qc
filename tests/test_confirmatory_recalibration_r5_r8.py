"""LP, leakage, and repeated-bootstrap validation for R5/R8."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linprog
from sklearn.model_selection import KFold

from shiftqc.confirmatory.bootstrap import inherited_fold_assignments, make_repeated_patient_folds
from shiftqc.confirmatory.r5 import compute_r5
from shiftqc.confirmatory.r8 import compute_r8
from shiftqc.confirmatory.recalibration import (
    cross_fitted_recalibrated_predictions,
    fit_affine_l1_irls,
)


def exact_lad_lp(q_hat: np.ndarray, q_true: np.ndarray) -> float:
    """Independent HiGHS formulation with free affine coefficients."""
    n = q_hat.size
    design = np.column_stack((np.ones(n), q_hat))
    objective = np.concatenate((np.zeros(2), np.ones(n)))
    lhs = np.vstack(
        (
            np.column_stack((design, -np.eye(n))),
            np.column_stack((-design, -np.eye(n))),
        )
    )
    rhs = np.concatenate((q_true, -q_true))
    result = linprog(
        objective,
        A_ub=lhs,
        b_ub=rhs,
        bounds=[(None, None), (None, None), *([(0.0, None)] * n)],
        method="highs",
    )
    assert result.success
    return float(result.fun)


@pytest.mark.parametrize(
    "name,q_hat,q_true",
    [
        ("perfect_affine", np.arange(6.0), 1.0 + 2.0 * np.arange(6.0)),
        ("positive_slope", np.arange(7.0), 0.2 + 0.7 * np.arange(7.0)),
        ("negative_slope", np.arange(7.0), 4.0 - 0.7 * np.arange(7.0)),
        ("intercept_shift", np.arange(6.0), 9.0 + np.arange(6.0)),
        (
            "ties",
            np.array([0, 0, 1, 1, 2, 2, 3, 3.0]),
            np.array([1, 1.2, 2, 2.1, 3, 3.2, 4, 7.0]),
        ),
        (
            "outlier",
            np.array([0, 0, 1, 1, 2, 2, 3, 3.0]),
            np.array([1, 1.2, 2, 2.1, 3, 3.2, 4, 70.0]),
        ),
        ("constant_q_hat", np.ones(7), np.array([1, 2, 3, 4, 5, 6, 20.0])),
        ("small_n", np.array([0.0, 1.0]), np.array([2.0, 5.0])),
        (
            "near_degenerate",
            np.array([0, 1e-12, 2e-12, 1, 2, 3.0]),
            0.2 + 1.3 * np.array([0, 1e-12, 2e-12, 1, 2, 3.0]),
        ),
    ],
)
def test_irls_objective_matches_exact_highs_below_frozen_tolerance(
    name, q_hat, q_true
) -> None:
    fit = fit_affine_l1_irls(q_hat, q_true)
    assert abs(fit.objective_l1 - exact_lad_lp(q_hat, q_true)) < 1e-11, name


def test_affine_recalibrator_is_not_clipped() -> None:
    fit = fit_affine_l1_irls(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
    assert fit.predict(np.array([-2.0, 4.0])).tolist() == pytest.approx([-1.0, 5.0])


def test_held_out_duplicate_labels_cannot_enter_own_fit_test_b() -> None:
    q_hat = np.linspace(0.0, 1.0, 20)
    q_true = 0.1 + 0.8 * q_hat
    folds = np.repeat(np.arange(5), 4)
    parents = np.array([0, 0, *range(1, 20)])
    inherited = folds[parents]
    before = cross_fitted_recalibrated_predictions(
        q_hat[parents], q_true[parents], inherited, recalibrator="affine_L1"
    )
    mutated = q_true[parents].copy()
    mutated[:2] = 1000.0
    after = cross_fitted_recalibrated_predictions(
        q_hat[parents], mutated, inherited, recalibrator="affine_L1"
    )
    assert after[:2] == pytest.approx(before[:2])


def test_correct_parent_folds_detect_deliberately_buggy_position_repartition_test_c() -> None:
    ids = [f"p{i:02d}" for i in range(20)]
    fixed = make_repeated_patient_folds(ids, seeds=(2026,)).assignments
    parents = np.array([0] * 10 + list(range(1, 11)))
    correct = inherited_fold_assignments(fixed, parents)[0]
    assert np.unique(correct[parents == 0]).size == 1

    buggy = np.full(parents.size, -1)
    for fold, (_, held_out) in enumerate(KFold(5, shuffle=False).split(parents)):
        buggy[held_out] = fold
    assert np.unique(buggy[parents == 0]).size > 1
    assert not np.array_equal(correct, buggy)


def synthetic_populations(n: int = 25):
    ids = np.asarray([f"p{i:02d}" for i in range(n)])
    q_hat = np.linspace(0.1, 0.9, n)
    q_true = 0.05 + 0.9 * q_hat + 0.01 * np.sin(np.arange(n))
    return ids, q_hat, q_true


def test_r5_repeat_first_row_invariant_and_deterministic() -> None:
    ids, q_hat, q_true = synthetic_populations()
    first = compute_r5(ids, q_hat, ids, q_true, ids, q_hat, ids, q_true, bootstrap_B=8)
    order = np.random.default_rng(3).permutation(ids.size)
    second = compute_r5(
        ids[order], q_hat[order], ids, q_true, ids, q_hat, ids, q_true, bootstrap_B=8
    )
    assert first == second
    assert first["Delta5"] == pytest.approx(0.0)
    assert first["fold_integrity_violations"] == 0


@pytest.mark.parametrize("recalibrator", ["affine_L1", "affine_OLS", "isotonic"])
def test_r8_primary_and_sensitivities_run_synthetic_only(recalibrator) -> None:
    ids, q_hat, q_true = synthetic_populations()
    result = compute_r8(
        ids,
        q_hat,
        ids,
        q_true,
        ids,
        q_hat,
        ids,
        q_true,
        recalibrator=recalibrator,
        seeds=(2026, 2027),
        bootstrap_B=3,
    )
    assert result["Delta8"] == pytest.approx(0.0)
    assert result["fold_integrity_violations"] == 0
    assert result["role"] == ("PRIMARY" if recalibrator == "affine_L1" else "SENSITIVITY")
    assert result["primary_verdict_eligible"] is (recalibrator == "affine_L1")


def test_r8_same_seed_is_numerically_identical_test_d() -> None:
    ids, q_hat, q_true = synthetic_populations()
    kwargs = dict(seeds=(2026, 2027), bootstrap_B=4, bootstrap_seed=2026)
    first = compute_r8(ids, q_hat, ids, q_true, ids, q_hat, ids, q_true, **kwargs)
    second = compute_r8(ids, q_hat, ids, q_true, ids, q_hat, ids, q_true, **kwargs)
    assert first == second
