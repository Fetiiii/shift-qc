"""Synthetic data-generating model and point estimators for EXP-0007.

Protocol: docs/EXP0007_R8_SYNTHETIC_PROTOCOL.md, Sections 3, 4 and 9. No patient data
is read anywhere in this module. The estimators call the frozen confirmatory
functions unchanged.
"""

from __future__ import annotations

import math

import numpy as np

from shiftqc.confirmatory.bootstrap import make_repeated_patient_folds
from shiftqc.confirmatory.common import CROSSFIT_FOLDS, CROSSFIT_SEEDS
from shiftqc.confirmatory.r5 import _repeat_values as _r5_repeat_values
from shiftqc.confirmatory.r8 import _repeat_values as _r8_repeat_values

# Beta(8.5, 1.5): the in-domain quality distribution fixed by the protocol.
BETA_ID = (8.5, 1.5)
BETA_DISPERSED = (4.0, 1.5)  # scenario F, out of domain
Q_MEAN = BETA_ID[0] / sum(BETA_ID)
Q_SD = math.sqrt(BETA_ID[0] * BETA_ID[1] / (sum(BETA_ID) ** 2 * (sum(BETA_ID) + 1.0)))
NONINFORMATIVE_SPREAD = 0.5  # protocol Section 9, clarification 2
SCENARIOS = ("A", "B", "C", "D", "E", "F")
RECALIBRATORS = ("affine_L1", "affine_OLS", "isotonic")


class SyntheticDesignError(ValueError):
    """The requested cell is outside the frozen design."""


def _score(q: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    z = (q - Q_MEAN) / Q_SD
    return rho * z + math.sqrt(1.0 - rho * rho) * rng.standard_normal(q.size)


def simulate_domain(n: int, rho: float, rng: np.random.Generator, *,
                    scenario: str, out_of_domain: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (q_hat, q_true) for one domain of one simulated dataset."""
    if scenario not in SCENARIOS:
        raise SyntheticDesignError(f"unknown scenario {scenario!r}")
    if not 0.0 <= rho < 1.0:
        raise SyntheticDesignError("rho must lie in [0, 1)")
    shifted = out_of_domain and scenario != "A"
    q = rng.beta(*(BETA_DISPERSED if shifted and scenario == "F" else BETA_ID), size=n)
    if rho == 0.0:
        if scenario != "A":
            raise SyntheticDesignError("rho = 0 is defined for scenario A only")
        return Q_MEAN + NONINFORMATIVE_SPREAD * Q_SD * rng.standard_normal(n), q
    if shifted and scenario == "E":
        return Q_MEAN + rho * Q_SD * rng.standard_normal(n), q
    x = _score(q, rho, rng)
    if shifted and scenario == "C":
        x = -x
    if shifted and scenario == "D":
        return Q_MEAN + Q_SD * np.tanh(2.0 * x), q
    q_hat = Q_MEAN + rho * Q_SD * x
    if shifted and scenario == "B":
        q_hat = 0.3 + 0.5 * q_hat
    return q_hat, q


def patient_ids(n: int, prefix: str) -> list[str]:
    return [f"{prefix}{index:06d}" for index in range(n)]


def frozen_folds(n: int, prefix: str) -> np.ndarray:
    """The registered repeated folds for ``n`` patients; identical in every dataset."""
    return make_repeated_patient_folds(
        patient_ids(n, prefix), CROSSFIT_SEEDS, n_splits=CROSSFIT_FOLDS).assignments


def point_estimates(id_q_hat: np.ndarray, id_q: np.ndarray, ood_q_hat: np.ndarray,
                    ood_q: np.ndarray, id_folds: np.ndarray,
                    ood_folds: np.ndarray) -> dict[str, float]:
    """S_ID, S_OOD and Delta for R5 and the three R8 recalibrators (repeat-first)."""
    out: dict[str, float] = {}
    id_r5 = _r5_repeat_values(id_q_hat, id_q, id_folds)
    ood_r5 = _r5_repeat_values(ood_q_hat, ood_q, ood_folds)
    out.update({"R5.S_ID": float(np.mean(id_r5)), "R5.S_OOD": float(np.mean(ood_r5)),
                "R5.Delta": float(np.mean(ood_r5 - id_r5))})
    for name in RECALIBRATORS:
        id_r8 = _r8_repeat_values(id_q_hat, id_q, id_folds, name)
        ood_r8 = _r8_repeat_values(ood_q_hat, ood_q, ood_folds, name)
        out.update({f"R8.{name}.S_ID": float(np.mean(id_r8)),
                    f"R8.{name}.S_OOD": float(np.mean(ood_r8)),
                    f"R8.{name}.Delta": float(np.mean(ood_r8 - id_r8))})
    return out
