"""Conformal calibration and interval evaluation."""
"""Split conformal prediction for frozen QC outputs."""

from shiftqc.conformal.split import (
    ConformalInputError,
    absolute_residuals,
    conformal_residual_threshold,
    construct_intervals,
    finite_sample_rank,
    interval_hits,
    interval_metrics,
)

__all__ = [
    "ConformalInputError",
    "absolute_residuals",
    "conformal_residual_threshold",
    "construct_intervals",
    "finite_sample_rank",
    "interval_hits",
    "interval_metrics",
]
