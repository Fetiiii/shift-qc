"""Frozen TASK-006 segmentation quality metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree


class MetricInputError(ValueError):
    """Raised when metric inputs violate the frozen metric contract."""


def _matching_arrays(reference: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference_array = np.asarray(reference)
    prediction_array = np.asarray(prediction)
    if reference_array.shape != prediction_array.shape:
        raise MetricInputError(
            f"shape mismatch: reference={reference_array.shape}, prediction={prediction_array.shape}"
        )
    if reference_array.ndim == 0:
        raise MetricInputError("metric inputs must have at least one dimension")
    return reference_array, prediction_array


def dice_binary(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Compute binary Dice using the frozen empty-mask convention."""
    reference_array, prediction_array = _matching_arrays(reference, prediction)
    reference_mask = reference_array.astype(bool, copy=False)
    prediction_mask = prediction_array.astype(bool, copy=False)
    reference_count = int(np.count_nonzero(reference_mask))
    prediction_count = int(np.count_nonzero(prediction_mask))
    if reference_count == 0 and prediction_count == 0:
        return 1.0
    if reference_count == 0 or prediction_count == 0:
        return 0.0
    intersection = int(np.count_nonzero(reference_mask & prediction_mask))
    return float(2.0 * intersection / (reference_count + prediction_count))


def dice_per_class(
    reference: np.ndarray,
    prediction: np.ndarray,
    labels: Iterable[int],
) -> dict[int, float]:
    """Compute one-vs-rest Dice for each requested foreground label."""
    reference_array, prediction_array = _matching_arrays(reference, prediction)
    label_values = tuple(int(label) for label in labels)
    if not label_values or len(set(label_values)) != len(label_values):
        raise MetricInputError("labels must be a non-empty sequence of unique integers")
    return {
        label: dice_binary(reference_array == label, prediction_array == label)
        for label in label_values
    }


def macro_foreground_dice(
    reference: np.ndarray,
    prediction: np.ndarray,
    labels: Iterable[int] = (1, 2, 3),
) -> float:
    """Compute the arithmetic mean of frozen per-class foreground Dice."""
    values = dice_per_class(reference, prediction, labels)
    return float(np.mean(tuple(values.values()), dtype=np.float64))


def whole_tumor_dice(
    reference: np.ndarray,
    prediction: np.ndarray,
    labels: Iterable[int] = (1, 2, 3),
) -> float:
    """Compute Dice after merging all configured foreground classes."""
    reference_array, prediction_array = _matching_arrays(reference, prediction)
    label_values = tuple(int(label) for label in labels)
    if not label_values:
        raise MetricInputError("whole-tumor labels must not be empty")
    return dice_binary(
        np.isin(reference_array, label_values),
        np.isin(prediction_array, label_values),
    )


def _surface_points(mask: np.ndarray, spacing: np.ndarray, connectivity: int) -> np.ndarray:
    structure = generate_binary_structure(mask.ndim, connectivity)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    surface = mask & ~eroded
    return np.argwhere(surface).astype(np.float64, copy=False) * spacing


def hd95_binary(
    reference: np.ndarray,
    prediction: np.ndarray,
    spacing: Sequence[float],
    *,
    percentile: float = 95.0,
    connectivity: int = 1,
) -> float:
    """Compute symmetric surface HD95 in physical units.

    Undefined empty-mask cases are explicit: both empty returns NaN and exactly
    one empty returns positive infinity. Callers must retain these values.
    """
    reference_array, prediction_array = _matching_arrays(reference, prediction)
    reference_mask = reference_array.astype(bool, copy=False)
    prediction_mask = prediction_array.astype(bool, copy=False)
    spacing_array = np.asarray(spacing, dtype=np.float64)
    if spacing_array.shape != (reference_mask.ndim,):
        raise MetricInputError(
            f"spacing must contain {reference_mask.ndim} values, got {spacing_array.shape}"
        )
    if not np.isfinite(spacing_array).all() or np.any(spacing_array <= 0):
        raise MetricInputError("spacing values must be finite and strictly positive")
    if not 0.0 <= percentile <= 100.0:
        raise MetricInputError("percentile must be in [0, 100]")
    if not 1 <= connectivity <= reference_mask.ndim:
        raise MetricInputError(
            f"connectivity must be in [1, {reference_mask.ndim}]"
        )

    reference_empty = not bool(np.any(reference_mask))
    prediction_empty = not bool(np.any(prediction_mask))
    if reference_empty and prediction_empty:
        return float("nan")
    if reference_empty or prediction_empty:
        return float("inf")

    reference_surface = _surface_points(reference_mask, spacing_array, connectivity)
    prediction_surface = _surface_points(prediction_mask, spacing_array, connectivity)
    reference_tree = cKDTree(reference_surface)
    prediction_tree = cKDTree(prediction_surface)
    prediction_to_reference = reference_tree.query(
        prediction_surface, k=1, workers=1
    )[0]
    reference_to_prediction = prediction_tree.query(
        reference_surface, k=1, workers=1
    )[0]
    symmetric_distances = np.concatenate(
        (prediction_to_reference, reference_to_prediction)
    )
    result = float(np.percentile(symmetric_distances, percentile))
    if not np.isfinite(result) or result < 0:
        raise RuntimeError(f"finite non-negative HD95 expected, got {result}")
    return result


def patient_segmentation_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    spacing: Sequence[float],
    *,
    foreground_labels: Iterable[int] = (1, 2, 3),
    hd95_percentile: float = 95.0,
    surface_connectivity: int = 1,
) -> dict[str, float]:
    """Compute the complete frozen patient-level TASK-006 metric set."""
    labels = tuple(int(label) for label in foreground_labels)
    per_class = dice_per_class(reference, prediction, labels)
    if labels != (1, 2, 3):
        raise MetricInputError("TASK-006 requires foreground labels (1, 2, 3)")
    values = {
        "dice_class_1": per_class[1],
        "dice_class_2": per_class[2],
        "dice_class_3": per_class[3],
        "dice_macro": float(np.mean(tuple(per_class.values()), dtype=np.float64)),
        "dice_whole_tumor": whole_tumor_dice(reference, prediction, labels),
        "hd95": hd95_binary(
            np.isin(reference, labels),
            np.isin(prediction, labels),
            spacing,
            percentile=hd95_percentile,
            connectivity=surface_connectivity,
        ),
    }
    dice_values = np.asarray(
        [value for name, value in values.items() if name.startswith("dice_")],
        dtype=np.float64,
    )
    if not np.isfinite(dice_values).all() or np.any(dice_values < 0) or np.any(dice_values > 1):
        raise RuntimeError(f"Dice values outside [0, 1]: {dice_values.tolist()}")
    return values
