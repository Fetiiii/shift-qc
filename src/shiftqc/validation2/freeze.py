"""Deterministic identity-only split and fold contracts for TASK-033."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from .contracts import ContractError


ROLE_ORDER = ("SEG_DEV", "QC_TRAIN", "CALIBRATION", "ID_EVAL")
SOURCE_STRATUM_ORDER = ("MSD Pancreas", "NIH Pancreas-CT")
SPLIT_SEED = 2026
SEGMENTATION_FOLDS = 5

SOURCE_ROLE_QUOTAS: dict[str, dict[str, int]] = {
    "MSD Pancreas": {
        "SEG_DEV": 140,
        "QC_TRAIN": 43,
        "CALIBRATION": 43,
        "ID_EVAL": 50,
    },
    "NIH Pancreas-CT": {
        "SEG_DEV": 40,
        "QC_TRAIN": 12,
        "CALIBRATION": 12,
        "ID_EVAL": 16,
    },
}

SEGMENTATION_PER_FOLD: dict[str, int] = {
    "MSD Pancreas": 28,
    "NIH Pancreas-CT": 8,
}


def _validated_rows(
    patient_ids: Sequence[str],
    source_case_ids: Sequence[str],
    provenance: Sequence[str],
) -> list[tuple[str, str, str]]:
    rows = [
        (str(patient), str(source_case), str(source))
        for patient, source_case, source in zip(
            patient_ids, source_case_ids, provenance, strict=True
        )
    ]
    if not rows:
        raise ContractError("at least one patient is required")
    if any(not all(row) for row in rows):
        raise ContractError("split identities and provenance must be non-empty")
    if len({row[0] for row in rows}) != len(rows):
        raise ContractError("duplicate patient ID")
    if len({(row[2], row[1]) for row in rows}) != len(rows):
        raise ContractError("duplicate provenance-scoped source case ID")
    return rows


def make_fixed_quota_split(
    patient_ids: Sequence[str],
    source_case_ids: Sequence[str],
    provenance: Sequence[str],
    *,
    quotas: Mapping[str, Mapping[str, int]] = SOURCE_ROLE_QUOTAS,
    seed: int = SPLIT_SEED,
    stratum_order: Sequence[str] = SOURCE_STRATUM_ORDER,
) -> dict[str, str]:
    """Allocate fixed per-provenance quotas using one sequential PCG64 stream.

    Within each stratum, rows are first sorted by authoritative source case ID.
    One ``Generator(PCG64(seed))`` is initialized, and its state advances first
    through MSD Pancreas and then NIH Pancreas-CT. Role slices follow
    ``ROLE_ORDER``. Input row order therefore cannot affect the manifest.
    """

    rows = _validated_rows(patient_ids, source_case_ids, provenance)
    order = tuple(str(value) for value in stratum_order)
    if set(order) != set(quotas) or len(order) != len(set(order)):
        raise ContractError("stratum order must identify every quota stratum once")
    if any(tuple(quota) != ROLE_ORDER for quota in quotas.values()):
        raise ContractError(f"every quota must use role order {ROLE_ORDER}")

    observed = Counter(row[2] for row in rows)
    expected = {source: sum(int(value) for value in quotas[source].values()) for source in order}
    if observed != Counter(expected):
        raise ContractError(f"source counts differ from fixed quotas: {observed} != {expected}")
    if any(value < 0 for quota in quotas.values() for value in quota.values()):
        raise ContractError("role quotas must be nonnegative")

    rng = np.random.Generator(np.random.PCG64(seed))
    assignments: dict[str, str] = {}
    for source in order:
        members = sorted(
            ((source_case, patient) for patient, source_case, value in rows if value == source),
            key=lambda row: (row[0], row[1]),
        )
        shuffled = [members[index] for index in rng.permutation(len(members))]
        start = 0
        for role in ROLE_ORDER:
            stop = start + int(quotas[source][role])
            for _, patient in shuffled[start:stop]:
                assignments[patient] = role
            start = stop
        if start != len(members):
            raise ContractError("fixed quota did not exhaust a provenance stratum")
    if len(assignments) != len(rows):
        raise ContractError("fixed split did not assign every source patient exactly once")
    return dict(sorted(assignments.items()))


def make_segmentation_fold_assignments(
    patient_ids: Sequence[str],
    source_case_ids: Sequence[str],
    provenance: Sequence[str],
    *,
    per_fold: Mapping[str, int] = SEGMENTATION_PER_FOLD,
    seed: int = SPLIT_SEED,
    stratum_order: Sequence[str] = SOURCE_STRATUM_ORDER,
    n_folds: int = SEGMENTATION_FOLDS,
) -> dict[str, int]:
    """Assign SEG_DEV patients to balanced folds with an independent PCG64 stream."""

    rows = _validated_rows(patient_ids, source_case_ids, provenance)
    order = tuple(str(value) for value in stratum_order)
    if n_folds < 2 or set(order) != set(per_fold):
        raise ContractError("fold strata or fold count are invalid")
    expected = {source: int(per_fold[source]) * n_folds for source in order}
    if Counter(row[2] for row in rows) != Counter(expected):
        raise ContractError("SEG_DEV provenance counts differ from fixed fold quotas")

    rng = np.random.Generator(np.random.PCG64(seed))
    assignments: dict[str, int] = {}
    for source in order:
        members = sorted(
            ((source_case, patient) for patient, source_case, value in rows if value == source),
            key=lambda row: (row[0], row[1]),
        )
        shuffled = [members[index] for index in rng.permutation(len(members))]
        width = int(per_fold[source])
        for fold in range(n_folds):
            for _, patient in shuffled[fold * width : (fold + 1) * width]:
                assignments[patient] = fold
    if len(assignments) != len(rows):
        raise ContractError("fold assignment did not cover every SEG_DEV patient")
    return dict(sorted(assignments.items()))
