"""Pure, synthetic-testable Validation-2 structural contracts.

No function in this module loads real data, predictions, or scientific results.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


class ContractError(ValueError):
    """A prospective Validation-2 contract was violated."""


REQUIRED_LABELS = {
    0: "background",
    1: "liver",
    2: "kidney",
    3: "spleen",
    4: "pancreas",
}

LABEL_INTEGER_TOLERANCE = 1e-5

FEATURE_COLUMNS = (
    "entropy_mean",
    "entropy_p90",
    "entropy_p95",
    "confidence_mean",
    "confidence_p10",
    "foreground_fraction",
    "class_fraction_liver",
    "class_fraction_kidney",
    "class_fraction_spleen",
    "class_fraction_pancreas",
    "components_liver",
    "components_kidney",
    "components_spleen",
    "components_pancreas",
    "largest_component_fraction_liver",
    "largest_component_fraction_kidney",
    "largest_component_fraction_spleen",
    "largest_component_fraction_pancreas",
)


def normalize_label_values(
    values: np.ndarray,
    tolerance: float = LABEL_INTEGER_TOLERANCE,
) -> np.ndarray:
    """Validate finite near-integer labels and return exact int64 labels."""

    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ContractError("labels must be finite")
    rounded = np.rint(array)
    if np.any(np.abs(array - rounded) > tolerance):
        raise ContractError("label differs from nearest integer beyond tolerance")
    normalized = rounded.astype(np.int64)
    unexpected = sorted(set(np.unique(normalized).tolist()) - set(REQUIRED_LABELS))
    if unexpected:
        raise ContractError(f"unsupported label IDs: {unexpected}")
    return normalized


def dice_for_required_organ(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    organ_label: int,
) -> float:
    """Prospective per-organ Dice used only by synthetic tests in TASK-031."""

    if organ_label not in REQUIRED_LABELS or organ_label == 0:
        raise ContractError("organ_label must be one of 1, 2, 3, 4")
    ground_truth = normalize_label_values(ground_truth)
    prediction = normalize_label_values(prediction)
    gt_mask = ground_truth == organ_label
    pred_mask = prediction == organ_label
    if not gt_mask.any():
        raise ContractError("required ground-truth organ is empty")
    denominator = int(gt_mask.sum()) + int(pred_mask.sum())
    if denominator == 0:
        raise ContractError("Dice denominator is zero")
    return float(2 * np.logical_and(gt_mask, pred_mask).sum() / denominator)


def validate_feature_columns(columns: Iterable[str]) -> tuple[str, ...]:
    """Require the exact ordered 18-feature, label-free contract."""

    observed = tuple(columns)
    if observed != FEATURE_COLUMNS:
        raise ContractError("QC feature columns differ from frozen order")
    forbidden_tokens = ("ground_truth", "q_true", "dice", "hd95", "mask_gt")
    if any(token in name.lower() for name in observed for token in forbidden_tokens):
        raise ContractError("ground-truth-derived QC feature is forbidden")
    return observed


def make_metadata_split(
    patient_ids: Sequence[str],
    provenance: Sequence[str],
    sizes: dict[str, int],
    *,
    seed: int,
) -> dict[str, str]:
    """Deterministic provenance-stratified allocation for synthetic validation.

    The real split may be created only after case-level provenance is resolved.
    Within each provenance stratum, canonical IDs are sorted and permuted with
    NumPy PCG64. Allocation order is SEG_DEV, QC_TRAIN, CALIBRATION, ID_EVAL.
    Each requested global role size is apportioned across strata by largest
    remainder with lexical provenance tie-breaking.
    """

    roles = ("SEG_DEV", "QC_TRAIN", "CALIBRATION", "ID_EVAL")
    if tuple(sizes) != roles:
        raise ContractError(f"sizes must use allocation order {roles}")
    ids = np.asarray([str(x) for x in patient_ids], dtype=object)
    strata = np.asarray([str(x) for x in provenance], dtype=object)
    if len(ids) != len(strata) or len(ids) == 0:
        raise ContractError("patient_ids and provenance must have equal nonzero length")
    if len(set(ids.tolist())) != len(ids):
        raise ContractError("duplicate patient ID")
    if any(not x or x == "PROVENANCE_UNRESOLVED" for x in strata.tolist()):
        raise ContractError("resolved provenance is required before splitting")
    if sum(sizes.values()) != len(ids) or any(v < 0 for v in sizes.values()):
        raise ContractError("split sizes must be nonnegative and exhaust patients")

    unique_strata = sorted(set(strata.tolist()))
    counts = {name: int(np.sum(strata == name)) for name in unique_strata}
    remaining = counts.copy()
    allocations: dict[str, dict[str, int]] = {role: {} for role in roles}
    remaining_total = len(ids)
    for role_index, role in enumerate(roles):
        target = sizes[role]
        if role_index == len(roles) - 1:
            allocation = remaining.copy()
        else:
            quotas = {
                name: target * remaining[name] / remaining_total
                for name in unique_strata
            }
            allocation = {name: int(np.floor(quotas[name])) for name in unique_strata}
            slots = target - sum(allocation.values())
            order = sorted(
                unique_strata,
                key=lambda name: (-(quotas[name] - allocation[name]), name),
            )
            for name in order[:slots]:
                allocation[name] += 1
        if sum(allocation.values()) != target:
            raise ContractError("stratified allocation failed")
        allocations[role] = allocation
        for name, count in allocation.items():
            remaining[name] -= count
        remaining_total -= target

    rng = np.random.Generator(np.random.PCG64(seed))
    result: dict[str, str] = {}
    for name in unique_strata:
        members = np.asarray(sorted(ids[strata == name].tolist()), dtype=object)
        members = members[rng.permutation(len(members))]
        start = 0
        for role in roles:
            stop = start + allocations[role][name]
            for patient_id in members[start:stop].tolist():
                result[str(patient_id)] = role
            start = stop
    if len(result) != len(ids):
        raise ContractError("split did not assign every patient exactly once")
    return dict(sorted(result.items()))
