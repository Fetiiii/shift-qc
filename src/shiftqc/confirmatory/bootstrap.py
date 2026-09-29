"""Patient-identity-preserving bootstrap and repeated fold assignments."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import KFold

from .common import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CROSSFIT_FOLDS,
    CROSSFIT_SEEDS,
    ConfirmatoryInputError,
    PatientIds,
    canonical_patient_ids,
)


@dataclass(frozen=True)
class PatientBootstrap:
    """Parent-patient resamples; positions always index the unique ID vector."""

    patient_ids: np.ndarray
    indices: np.ndarray
    seed: int
    resamples: int


@dataclass(frozen=True)
class TwoPopulationBootstrap:
    """Independent ID and OOD patient resamples from one frozen RNG stream."""

    id_population: PatientBootstrap
    ood_population: PatientBootstrap
    seed: int
    resamples: int


@dataclass(frozen=True)
class RepeatedPatientFolds:
    """Fixed repeat-by-patient fold identities generated before bootstrap."""

    patient_ids: np.ndarray
    seeds: tuple[int, ...]
    n_splits: int
    assignments: np.ndarray


def make_patient_bootstrap(
    patient_ids: PatientIds,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    generator: np.random.Generator | None = None,
) -> PatientBootstrap:
    """Draw reproducible patient resamples from a unique canonical ID vector."""
    ids = canonical_patient_ids(patient_ids)
    if resamples <= 0:
        raise ConfirmatoryInputError("bootstrap resamples must be positive")
    rng = np.random.default_rng(seed) if generator is None else generator
    indices = rng.integers(0, ids.size, size=(resamples, ids.size), endpoint=False)
    return PatientBootstrap(ids, indices, int(seed), int(resamples))


def make_two_population_bootstrap(
    id_patient_ids: PatientIds,
    ood_patient_ids: PatientIds,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> TwoPopulationBootstrap:
    """Draw ID and OOD independently, preserving each population's sample size."""
    rng = np.random.default_rng(seed)
    id_bootstrap = make_patient_bootstrap(
        id_patient_ids, seed=seed, resamples=resamples, generator=rng
    )
    ood_bootstrap = make_patient_bootstrap(
        ood_patient_ids, seed=seed, resamples=resamples, generator=rng
    )
    return TwoPopulationBootstrap(id_bootstrap, ood_bootstrap, seed, resamples)


def make_repeated_patient_folds(
    patient_ids: PatientIds,
    seeds: Sequence[int] = CROSSFIT_SEEDS,
    *,
    n_splits: int = CROSSFIT_FOLDS,
) -> RepeatedPatientFolds:
    """Generate deterministic, inspectable, fully OOF patient partitions."""
    ids = canonical_patient_ids(patient_ids)
    ordered_ids = np.asarray(sorted(ids.tolist()), dtype=str)
    seed_tuple = tuple(int(seed) for seed in seeds)
    if not seed_tuple or len(set(seed_tuple)) != len(seed_tuple):
        raise ConfirmatoryInputError("fold seeds must be non-empty and unique")
    if n_splits < 2 or ordered_ids.size < n_splits:
        raise ConfirmatoryInputError("n_splits must be in [2, patient_count]")

    assignments = np.full((len(seed_tuple), ordered_ids.size), -1, dtype=np.int64)
    positions = np.arange(ordered_ids.size)
    for repeat, seed in enumerate(seed_tuple):
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train, held_out) in enumerate(splitter.split(positions)):
            if np.intersect1d(train, held_out).size:
                raise RuntimeError("KFold produced overlapping train and held-out positions")
            assignments[repeat, held_out] = fold
        if np.any(assignments[repeat] < 0):
            raise RuntimeError("repeat did not assign every patient exactly once")
        counts = np.bincount(assignments[repeat], minlength=n_splits)
        if np.any(counts == 0):
            raise RuntimeError("repeat contains an empty held-out fold")

    return RepeatedPatientFolds(ordered_ids, seed_tuple, n_splits, assignments)


def align_folds_to_patient_order(
    folds: RepeatedPatientFolds,
    patient_ids: PatientIds,
) -> np.ndarray:
    """Return fixed fold identities in an independently supplied unique ID order."""
    ids = canonical_patient_ids(patient_ids)
    if set(ids.tolist()) != set(folds.patient_ids.tolist()):
        raise ConfirmatoryInputError("fold and endpoint patient_id sets differ")
    positions = {patient_id: index for index, patient_id in enumerate(folds.patient_ids)}
    return folds.assignments[:, [positions[patient_id] for patient_id in ids.tolist()]]


def inherited_fold_assignments(
    assignments: np.ndarray,
    parent_indices: np.ndarray,
) -> np.ndarray:
    """Index predetermined folds by bootstrap parents; never repartition copies."""
    folds = np.asarray(assignments, dtype=np.int64)
    parents = np.asarray(parent_indices, dtype=np.int64)
    if folds.ndim != 2 or parents.ndim != 1:
        raise ConfirmatoryInputError("fold assignments must be 2D and parents 1D")
    if parents.size == 0 or np.any((parents < 0) | (parents >= folds.shape[1])):
        raise ConfirmatoryInputError("bootstrap parent index is out of range")
    return folds[:, parents]


def count_fold_inheritance_violations(
    assignments: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> int:
    """Count parents whose copies acquire more than one fold in any repeat."""
    folds = np.asarray(assignments, dtype=np.int64)
    resamples = np.asarray(bootstrap_indices, dtype=np.int64)
    if folds.ndim != 2 or resamples.ndim != 2:
        raise ConfirmatoryInputError("assignments and bootstrap_indices must be 2D")
    violations = 0
    for parents in resamples:
        inherited = inherited_fold_assignments(folds, parents)
        for repeat in range(folds.shape[0]):
            for parent in np.unique(parents):
                if np.unique(inherited[repeat, parents == parent]).size != 1:
                    violations += 1
    return violations
