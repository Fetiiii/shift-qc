"""Validation-2 wrappers around the result-blind frozen endpoint engine."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

from shiftqc.confirmatory.competence import compute_baseline_competence
from shiftqc.confirmatory.r5 import compute_r5
from shiftqc.confirmatory.r8 import compute_r8 as _compute_r8

from .contracts import ContractError


COMPETENCE_MACRO_THRESHOLD = 0.80
COMPETENCE_ORGAN_THRESHOLD = 0.60


def segmentation_competence(organ_dice: np.ndarray) -> dict[str, object]:
    """Apply the frozen aggregate and anti-collapse gates to OOF patient Dice."""

    values = np.asarray(organ_dice, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
        raise ContractError("organ_dice must be a non-empty n x 4 OOF matrix")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ContractError("organ Dice must be finite and lie in [0, 1]")
    organ_means = np.asarray(
        [math.fsum(values[:, column].tolist()) / values.shape[0] for column in range(4)],
        dtype=np.float64,
    )
    patient_macro = np.asarray(
        [math.fsum(row.tolist()) / 4 for row in values], dtype=np.float64
    )
    macro_mean = math.fsum(patient_macro.tolist()) / values.shape[0]
    passed = macro_mean >= COMPETENCE_MACRO_THRESHOLD and bool(
        np.all(organ_means >= COMPETENCE_ORGAN_THRESHOLD)
    )
    return {
        "macro_mean": macro_mean,
        "organ_means": organ_means.tolist(),
        "n": int(values.shape[0]),
        "verdict": "PASS" if passed else "FAIL",
    }


def compute_validation2_baseline(
    q_hat_patient_ids: Sequence[str],
    q_hat: np.ndarray,
    q_true_patient_ids: Sequence[str],
    q_true: np.ndarray,
    *,
    bootstrap_B: int = 2000,
    bootstrap_seed: int = 2026,
) -> dict[str, object]:
    """Compute the frozen ID_EVAL tied-rank Spearman prerequisite."""

    result = compute_baseline_competence(
        q_hat_patient_ids,
        q_hat,
        q_true_patient_ids,
        q_true,
        bootstrap_B=bootstrap_B,
        bootstrap_seed=bootstrap_seed,
    )
    result["population"] = "ID_EVAL"
    return result


def r8_decision(ci_low: float, ci_high: float) -> str:
    """Apply the directional primary decision without equivalence inference."""

    if not np.isfinite([ci_low, ci_high]).all() or ci_low > ci_high:
        raise ContractError("R8 confidence interval is invalid")
    return "POSITIVE" if ci_high < 0.0 else "UNRESOLVED"


def compute_r8(*args: object, **kwargs: object) -> dict[str, object]:
    """Run frozen R8 and use Validation-2's predeclared verdict labels."""

    result = _compute_r8(*args, **kwargs)
    if result.get("recalibrator") == "affine_L1":
        result["verdict"] = r8_decision(float(result["CI_low"]), float(result["CI_high"]))
    return result


__all__ = [
    "compute_r5",
    "compute_r8",
    "compute_validation2_baseline",
    "r8_decision",
    "segmentation_competence",
]
