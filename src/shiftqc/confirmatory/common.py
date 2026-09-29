"""Shared input, alignment, and uncertainty contracts for TASK-024."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np


BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 2026
CONFIDENCE_LEVEL = 0.95
CROSSFIT_FOLDS = 5
CROSSFIT_SEEDS = tuple(range(2026, 2036))
CONFORMAL_ALPHA = 0.10
IRLS_ITERATIONS = 80
IRLS_EPSILON = 1e-9

PatientIds: TypeAlias = Sequence[str] | np.ndarray
NumericValues: TypeAlias = Sequence[float] | np.ndarray


class ConfirmatoryInputError(ValueError):
    """Raised when an input violates a frozen confirmatory contract."""


def canonical_patient_ids(patient_ids: PatientIds, *, unique: bool = True) -> np.ndarray:
    """Return a validated one-dimensional canonical string-ID vector."""
    raw = np.asarray(patient_ids, dtype=object)
    if raw.ndim != 1 or raw.size == 0:
        raise ConfirmatoryInputError("patient_ids must be a non-empty vector")
    if any(not isinstance(value, str) or not value for value in raw.tolist()):
        raise ConfirmatoryInputError("patient_ids must contain non-empty strings")
    ids = raw.astype(str)
    if unique and np.unique(ids).size != ids.size:
        raise ConfirmatoryInputError("duplicated patient_id")
    return ids


def finite_vector(
    values: NumericValues,
    name: str,
    *,
    expected_size: int | None = None,
) -> np.ndarray:
    """Return a finite non-empty float64 vector, refusing silent repair."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ConfirmatoryInputError(f"{name} must be a non-empty vector")
    if expected_size is not None and array.size != expected_size:
        raise ConfirmatoryInputError(
            f"{name} length {array.size} does not match expected {expected_size}"
        )
    if not np.isfinite(array).all():
        raise ConfirmatoryInputError(f"{name} must contain only finite values")
    return array


def unit_interval_vector(values: NumericValues, name: str) -> np.ndarray:
    """Validate a finite vector on the physical Dice/coverage support."""
    array = finite_vector(values, name)
    if np.any((array < 0.0) | (array > 1.0)):
        raise ConfirmatoryInputError(f"{name} must lie in [0, 1]")
    return array


def align_named_vectors(
    vectors: Mapping[str, tuple[PatientIds, NumericValues]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Align independent vectors by exact patient-ID set and lexical order."""
    if not vectors:
        raise ConfirmatoryInputError("at least one named patient vector is required")

    value_maps: dict[str, dict[str, float]] = {}
    reference_set: set[str] | None = None
    for name, (patient_ids, values) in vectors.items():
        ids = canonical_patient_ids(patient_ids)
        array = finite_vector(values, name, expected_size=ids.size)
        current_set = set(ids.tolist())
        if reference_set is None:
            reference_set = current_set
        elif current_set != reference_set:
            missing = sorted(reference_set - current_set)
            extra = sorted(current_set - reference_set)
            raise ConfirmatoryInputError(
                f"patient_id set mismatch for {name}: missing={missing}, extra={extra}"
            )
        value_maps[name] = dict(zip(ids.tolist(), array.tolist(), strict=True))

    assert reference_set is not None
    ordered_ids = np.asarray(sorted(reference_set), dtype=str)
    aligned = {
        name: np.asarray([mapping[patient_id] for patient_id in ordered_ids], dtype=np.float64)
        for name, mapping in value_maps.items()
    }
    return ordered_ids, aligned


def percentile_interval(
    values: NumericValues,
    *,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """Return the frozen two-sided linear percentile interval."""
    array = finite_vector(values, "bootstrap values")
    if not 0.0 < confidence_level < 1.0:
        raise ConfirmatoryInputError("confidence_level must lie in (0, 1)")
    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(array, [tail, 1.0 - tail], method="linear")
    return float(low), float(high)


def copy_ordinals(patient_ids: PatientIds) -> np.ndarray:
    """Assign deterministic zero-based occurrence suffixes to bootstrap copies."""
    ids = canonical_patient_ids(patient_ids, unique=False)
    counts: dict[str, int] = {}
    ordinals = np.empty(ids.size, dtype=np.int64)
    for index, patient_id in enumerate(ids.tolist()):
        ordinal = counts.get(patient_id, 0)
        ordinals[index] = ordinal
        counts[patient_id] = ordinal + 1
    return ordinals
