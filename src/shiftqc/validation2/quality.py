"""Frozen four-organ quality target for Validation-2."""

from __future__ import annotations

import numpy as np

from .contracts import ContractError, REQUIRED_LABELS, normalize_label_values


ORGAN_ORDER = ("liver", "kidney", "spleen", "pancreas")


def four_organ_quality(ground_truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Return four binary organ Dice values and their patient arithmetic mean."""

    gt = normalize_label_values(ground_truth)
    pred = normalize_label_values(prediction)
    if gt.shape != pred.shape or gt.size == 0:
        raise ContractError("ground truth and prediction must have equal non-empty shapes")
    result: dict[str, float] = {}
    for label, organ in REQUIRED_LABELS.items():
        if label == 0:
            continue
        gt_mask = gt == label
        pred_mask = pred == label
        if not gt_mask.any():
            raise ContractError(f"required ground-truth organ is empty: {organ}")
        denominator = int(gt_mask.sum()) + int(pred_mask.sum())
        value = float(2 * np.logical_and(gt_mask, pred_mask).sum() / denominator)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ContractError(f"invalid Dice for {organ}")
        result[f"dice_{organ}"] = value
    result["q_true"] = float(np.mean([result[f"dice_{organ}"] for organ in ORGAN_ORDER]))
    return result
