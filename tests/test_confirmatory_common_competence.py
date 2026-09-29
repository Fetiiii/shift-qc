"""Analytic and contract tests for common TASK-024 primitives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shiftqc.confirmatory.bootstrap import (
    count_fold_inheritance_violations,
    inherited_fold_assignments,
    make_patient_bootstrap,
    make_repeated_patient_folds,
)
from shiftqc.confirmatory.common import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CROSSFIT_FOLDS,
    CROSSFIT_SEEDS,
    ConfirmatoryInputError,
    align_named_vectors,
)
from shiftqc.confirmatory.competence import compute_baseline_competence, spearman_rho
from shiftqc.confirmatory.skill import cross_fitted_median_constant, relative_mae_skill


GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures/confirmatory_golden.json").read_text(encoding="utf-8")
)


def test_frozen_resampling_constants() -> None:
    assert BOOTSTRAP_RESAMPLES == 2000
    assert BOOTSTRAP_SEED == 2026
    assert CROSSFIT_FOLDS == 5
    assert CROSSFIT_SEEDS == tuple(range(2026, 2036))


def test_patient_alignment_is_exact_and_lexical() -> None:
    ids, values = align_named_vectors(
        {
            "a": (["p2", "p1", "p3"], [2.0, 1.0, 3.0]),
            "b": (["p3", "p2", "p1"], [30.0, 20.0, 10.0]),
        }
    )
    assert ids.tolist() == ["p1", "p2", "p3"]
    assert values["a"].tolist() == [1.0, 2.0, 3.0]
    assert values["b"].tolist() == [10.0, 20.0, 30.0]


@pytest.mark.parametrize(
    "vectors",
    [
        {"a": (["p1", "p1"], [1.0, 2.0])},
        {"a": (["p1", "p2"], [1.0, 2.0]), "b": (["p1", "p3"], [1.0, 3.0])},
        {"a": (["p1", "p2"], [1.0, np.nan])},
    ],
)
def test_alignment_hard_fails_invalid_inputs(vectors) -> None:
    with pytest.raises(ConfirmatoryInputError):
        align_named_vectors(vectors)


def test_golden_spearman_median_and_skill() -> None:
    q_true = np.asarray(GOLDEN["q_true"])
    q_hat = np.asarray(GOLDEN["q_hat"])
    folds = np.asarray(GOLDEN["folds"])
    comparator = cross_fitted_median_constant(q_true, folds)
    assert spearman_rho(q_hat, q_true) == pytest.approx(GOLDEN["expected"]["spearman"])
    assert comparator == pytest.approx(GOLDEN["expected"]["cross_fitted_median"])
    assert np.mean(np.abs(comparator - q_true)) == pytest.approx(
        GOLDEN["expected"]["comparator_mae"]
    )
    assert relative_mae_skill(q_hat, comparator, q_true) == pytest.approx(
        GOLDEN["expected"]["skill"]
    )


def test_cross_fitted_median_own_label_mutation_cannot_change_own_prediction() -> None:
    q_true = np.asarray(GOLDEN["q_true"])
    folds = np.asarray(GOLDEN["folds"])
    before = cross_fitted_median_constant(q_true, folds)
    mutated = q_true.copy()
    mutated[0] = 1000.0
    after = cross_fitted_median_constant(mutated, folds)
    assert after[0] == before[0]
    assert after[1] == before[1]


def test_skill_zero_comparator_mae_is_undefined() -> None:
    with pytest.raises(ConfirmatoryInputError, match="comparator MAE is zero"):
        relative_mae_skill(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0]))


def test_spearman_ties_use_average_ranks() -> None:
    rho = spearman_rho(np.array([1.0, 1.0, 2.0, 3.0]), np.array([1.0, 2.0, 2.0, 4.0]))
    assert rho == pytest.approx(5.0 / 6.0)


@pytest.mark.parametrize(
    "q_hat,q_true",
    [
        ([1.0], [1.0]),
        ([1.0, 1.0], [1.0, 2.0]),
        ([1.0, 2.0], [1.0, 1.0]),
        ([1.0, np.nan], [1.0, 2.0]),
        ([1.0, 2.0], [1.0, np.inf]),
    ],
)
def test_spearman_undefined_cases_hard_fail(q_hat, q_true) -> None:
    with pytest.raises(ConfirmatoryInputError):
        spearman_rho(np.asarray(q_hat), np.asarray(q_true))


def test_competence_row_order_invariance_and_determinism() -> None:
    ids = np.asarray([f"p{i:02d}" for i in range(20)])
    q_true = np.linspace(0.1, 0.9, 20)
    q_hat = q_true**2
    first = compute_baseline_competence(ids, q_hat, ids, q_true, bootstrap_B=20)
    permutation = np.random.default_rng(11).permutation(ids.size)
    second = compute_baseline_competence(
        ids[permutation], q_hat[permutation], ids, q_true, bootstrap_B=20
    )
    assert first == second
    assert first["verdict"] == "RESOLVED"


def test_repeated_folds_are_complete_deterministic_and_inspectable() -> None:
    ids = [f"p{i:02d}" for i in range(20)]
    first = make_repeated_patient_folds(ids)
    second = make_repeated_patient_folds(list(reversed(ids)))
    assert np.array_equal(first.assignments, second.assignments)
    assert first.patient_ids.tolist() == sorted(ids)
    for repeat in first.assignments:
        assert sorted(np.unique(repeat).tolist()) == [0, 1, 2, 3, 4]
        assert repeat.size == 20


def test_duplicate_copies_inherit_one_parent_fold_test_a() -> None:
    ids = [f"p{i:02d}" for i in range(20)]
    folds = make_repeated_patient_folds(ids)
    parents = np.repeat(3, 10)
    inherited = inherited_fold_assignments(folds.assignments, parents)
    assert all(np.unique(repeat).size == 1 for repeat in inherited)


def test_500_replicate_fold_integrity_has_zero_violations() -> None:
    ids = [f"p{i:02d}" for i in range(20)]
    folds = make_repeated_patient_folds(ids)
    bootstrap = make_patient_bootstrap(ids, resamples=500)
    assert count_fold_inheritance_violations(folds.assignments, bootstrap.indices) == 0
