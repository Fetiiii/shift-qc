"""Finite-sample absolute-residual split conformal primitives."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class ConformalInputError(ValueError):
    """Raised when split-conformal inputs violate the frozen contract."""


def finite_sample_rank(calibration_count: int, alpha: float) -> int:
    """Return one-based k = ceil((n_cal + 1) * (1 - alpha))."""
    if calibration_count <= 0:
        raise ConformalInputError("calibration_count must be positive")
    if not 0.0 < alpha < 1.0:
        raise ConformalInputError("alpha must lie strictly between zero and one")
    return int(math.ceil((calibration_count + 1) * (1.0 - alpha)))


def absolute_residuals(
    target: np.ndarray,
    point_prediction: np.ndarray,
) -> np.ndarray:
    """Compute finite absolute calibration residuals."""
    target_array = np.asarray(target, dtype=np.float64)
    prediction_array = np.asarray(point_prediction, dtype=np.float64)
    if target_array.ndim != 1 or target_array.shape != prediction_array.shape:
        raise ConformalInputError(
            f"residual input shape mismatch: target={target_array.shape}, "
            f"prediction={prediction_array.shape}"
        )
    if not np.isfinite(target_array).all() or not np.isfinite(prediction_array).all():
        raise ConformalInputError("residual inputs must be finite")
    return np.abs(target_array - prediction_array)


def conformal_residual_threshold(
    residuals: np.ndarray,
    alpha: float,
) -> tuple[float, int]:
    """Select the exact one-based finite-sample residual order statistic."""
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ConformalInputError("calibration residuals must be a non-empty vector")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ConformalInputError("calibration residuals must be finite and non-negative")
    rank = finite_sample_rank(int(values.size), alpha)
    if rank > values.size:
        raise ConformalInputError(
            f"finite-sample rank {rank} exceeds calibration count {values.size}"
        )
    threshold = float(np.partition(values, rank - 1)[rank - 1])
    return threshold, rank


def construct_intervals(
    point_predictions: np.ndarray,
    residual_threshold: float,
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> dict[str, np.ndarray]:
    """Construct raw and physically clipped intervals without clipping points."""
    predictions = np.asarray(point_predictions, dtype=np.float64)
    if predictions.ndim != 1 or not np.isfinite(predictions).all():
        raise ConformalInputError("point predictions must be a finite vector")
    if not np.isfinite(residual_threshold) or residual_threshold < 0:
        raise ConformalInputError("residual threshold must be finite and non-negative")
    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        raise ConformalInputError("interval bounds must be finite")
    if lower_bound >= upper_bound:
        raise ConformalInputError("lower interval bound must be below upper bound")

    lower_raw = predictions - residual_threshold
    upper_raw = predictions + residual_threshold
    # D-034: each endpoint is projected independently onto the physical support of
    # the target. Clipping is monotone, so lower_raw <= upper_raw implies
    # lower <= upper, and a raw interval lying entirely outside [0, 1] collapses to
    # the boundary it sits beyond: [0, 0] below, [1, 1] above. A degenerate boundary
    # interval is a valid bounded interval, not an empty one.
    #
    # The earlier one-sided form, max(lower_bound, lower_raw) with
    # min(upper_bound, upper_raw), left lower and upper on opposite sides when the
    # raw interval was disjoint from the support, and the resulting crossed pair was
    # rejected. It differs from this projection only when lower_raw > 1 or
    # upper_raw < 0, neither of which occurs in the frozen EXP-0002 artefacts, so
    # those remain bit-identical.
    #
    # This is not prediction clipping, interval widening, recalibration or a
    # scientific tolerance rule: point predictions are never clipped and the
    # calibration radius is untouched. It also cannot break the conformal coverage
    # event, since q lies in [0, 1] by construction, so q in [lower_raw, upper_raw]
    # implies q in [clip(lower_raw), clip(upper_raw)].
    lower = np.clip(lower_raw, lower_bound, upper_bound)
    upper = np.clip(upper_raw, lower_bound, upper_bound)
    if np.any(lower > upper):
        raise ConformalInputError(
            "projected interval bounds are crossed; raw lower exceeded raw upper"
        )
    width = upper - lower
    return {
        "lower_raw": lower_raw,
        "upper_raw": upper_raw,
        "lower": lower,
        "upper": upper,
        "width": width,
    }


def interval_hits(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Return inclusive interval membership for finite targets and bounds."""
    target_array = np.asarray(target, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    if (
        target_array.ndim != 1
        or target_array.shape != lower_array.shape
        or target_array.shape != upper_array.shape
    ):
        raise ConformalInputError(
            "target, lower, and upper must be matching one-dimensional arrays"
        )
    if not (
        np.isfinite(target_array).all()
        and np.isfinite(lower_array).all()
        and np.isfinite(upper_array).all()
    ):
        raise ConformalInputError("hit calculation requires finite values")
    if np.any(lower_array > upper_array):
        raise ConformalInputError("interval lower bound exceeds upper bound")
    return (target_array >= lower_array) & (target_array <= upper_array)


def interval_metrics(
    hits: np.ndarray,
    widths: np.ndarray,
    nominal_coverage: float,
) -> dict[str, Any]:
    """Compute the predeclared descriptive conformal evaluation metrics."""
    hit_array = np.asarray(hits)
    width_array = np.asarray(widths, dtype=np.float64)
    if hit_array.ndim != 1 or hit_array.shape != width_array.shape:
        raise ConformalInputError("hits and widths must be matching vectors")
    if hit_array.dtype != np.bool_:
        raise ConformalInputError("hits must be boolean")
    if not np.isfinite(width_array).all() or np.any(width_array < 0):
        raise ConformalInputError("interval widths must be finite and non-negative")
    if not 0.0 < nominal_coverage < 1.0:
        raise ConformalInputError("nominal coverage must lie in (0, 1)")
    coverage = float(np.mean(hit_array, dtype=np.float64))
    return {
        "n": int(hit_array.size),
        "coverage": coverage,
        "coverage_error": float(abs(coverage - nominal_coverage)),
        "mean_interval_width": float(np.mean(width_array, dtype=np.float64)),
        "median_interval_width": float(np.median(width_array)),
    }
