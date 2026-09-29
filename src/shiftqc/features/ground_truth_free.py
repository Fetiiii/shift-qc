"""Ground-truth-free feature primitives frozen for TASK-007."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import label as connected_components


class FeatureInputError(ValueError):
    """Raised when deployment-available feature inputs fail validation."""


def image_support(image: np.ndarray) -> np.ndarray:
    """Return the non-zero support of a finite skull-stripped input image."""
    array = np.asarray(image)
    if array.ndim != 3:
        raise FeatureInputError(f"input image must be 3D, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise FeatureInputError("input image contains NaN or infinity")
    support = array != 0
    if not np.any(support):
        raise FeatureInputError("input-image support is empty")
    return support


def validate_probabilities(
    probabilities: np.ndarray,
    expected_shape: tuple[int, ...],
    *,
    classes: int = 4,
    class_axis: int = 0,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-5,
) -> np.ndarray:
    """Validate frozen class probabilities without modifying or renormalizing."""
    array = np.asarray(probabilities)
    expected_probability_shape = list(expected_shape)
    expected_probability_shape.insert(class_axis, classes)
    if array.shape != tuple(expected_probability_shape):
        raise FeatureInputError(
            "probability shape mismatch: "
            f"expected={tuple(expected_probability_shape)}, observed={array.shape}"
        )
    if not np.isfinite(array).all():
        raise FeatureInputError("probability map contains NaN or infinity")
    if np.any(array < 0) or np.any(array > 1):
        observed_min = float(np.min(array))
        observed_max = float(np.max(array))
        raise FeatureInputError(
            f"probability values outside [0, 1]: min={observed_min}, max={observed_max}"
        )
    sums = np.sum(array, axis=class_axis, dtype=np.float64)
    if not np.allclose(
        sums,
        1.0,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    ):
        maximum_error = float(np.max(np.abs(sums - 1.0)))
        raise FeatureInputError(
            "class probabilities do not sum approximately to one; "
            f"maximum_absolute_error={maximum_error}"
        )
    return array


def validate_hard_prediction(
    prediction: np.ndarray,
    expected_shape: tuple[int, ...],
    *,
    allowed_labels: tuple[int, ...] = (0, 1, 2, 3),
) -> np.ndarray:
    """Validate hard-prediction geometry and the frozen label set."""
    array = np.asarray(prediction)
    if array.shape != expected_shape:
        raise FeatureInputError(
            f"hard-prediction shape mismatch: expected={expected_shape}, observed={array.shape}"
        )
    observed = {int(value) for value in np.unique(array)}
    unexpected = sorted(observed - set(allowed_labels))
    if unexpected:
        raise FeatureInputError(f"unexpected hard-prediction labels: {unexpected}")
    return array


def probability_features(
    probabilities: np.ndarray,
    support: np.ndarray,
    *,
    class_axis: int = 0,
) -> dict[str, float]:
    """Compute entropy and confidence summaries inside image-only support."""
    probability_array = np.asarray(probabilities)
    support_array = np.asarray(support, dtype=bool)
    spatial_shape = tuple(
        size for axis, size in enumerate(probability_array.shape) if axis != class_axis
    )
    if support_array.shape != spatial_shape:
        raise FeatureInputError(
            f"support shape mismatch: expected={spatial_shape}, observed={support_array.shape}"
        )
    if not np.any(support_array):
        raise FeatureInputError("image support is empty")

    entropy_terms = np.zeros_like(probability_array, dtype=np.float64)
    positive = probability_array > 0
    entropy_terms[positive] = (
        probability_array[positive] * np.log(probability_array[positive])
    )
    entropy = -np.sum(entropy_terms, axis=class_axis, dtype=np.float64)
    confidence = np.max(probability_array, axis=class_axis)
    supported_entropy = entropy[support_array]
    supported_confidence = confidence[support_array]
    values = {
        "entropy_mean": float(np.mean(supported_entropy, dtype=np.float64)),
        "entropy_p90": float(np.quantile(supported_entropy, 0.90)),
        "entropy_p95": float(np.quantile(supported_entropy, 0.95)),
        "confidence_mean": float(np.mean(supported_confidence, dtype=np.float64)),
        "confidence_p10": float(np.quantile(supported_confidence, 0.10)),
    }
    if not np.isfinite(tuple(values.values())).all():
        raise RuntimeError(f"non-finite probability features: {values}")
    if not 0.0 <= values["confidence_mean"] <= 1.0:
        raise RuntimeError(f"confidence_mean outside [0, 1]: {values}")
    if not 0.0 <= values["confidence_p10"] <= 1.0:
        raise RuntimeError(f"confidence_p10 outside [0, 1]: {values}")
    return values


def _component_features(class_mask: np.ndarray) -> tuple[int, float]:
    structure_26_connectivity = np.ones((3, 3, 3), dtype=bool)
    labeled, count = connected_components(
        class_mask,
        structure=structure_26_connectivity,
    )
    component_count = int(count)
    predicted_count = int(np.count_nonzero(class_mask))
    if predicted_count == 0:
        return 0, 0.0
    sizes = np.bincount(labeled.ravel())[1:]
    largest_fraction = float(np.max(sizes) / predicted_count)
    return component_count, largest_fraction


def morphology_features(
    prediction: np.ndarray,
    support: np.ndarray,
    *,
    foreground_labels: tuple[int, ...] = (1, 2, 3),
) -> dict[str, int | float]:
    """Compute frozen hard-prediction morphology features.

    Volume-fraction numerators are restricted to image support. Connected
    components are computed from the full hard prediction with 3D
    26-connectivity, independently of image support.
    """
    prediction_array = np.asarray(prediction)
    support_array = np.asarray(support, dtype=bool)
    if prediction_array.ndim != 3 or prediction_array.shape != support_array.shape:
        raise FeatureInputError(
            "prediction and support must be matching 3D arrays: "
            f"prediction={prediction_array.shape}, support={support_array.shape}"
        )
    support_count = int(np.count_nonzero(support_array))
    if support_count == 0:
        raise FeatureInputError("image support is empty")
    if tuple(foreground_labels) != (1, 2, 3):
        raise FeatureInputError("TASK-007 requires foreground labels (1, 2, 3)")

    features: dict[str, Any] = {}
    foreground = np.isin(prediction_array, foreground_labels)
    features["foreground_fraction"] = float(
        np.count_nonzero(foreground & support_array) / support_count
    )
    for class_label in foreground_labels:
        class_mask = prediction_array == class_label
        features[f"class{class_label}_fraction"] = float(
            np.count_nonzero(class_mask & support_array) / support_count
        )
        count, largest_fraction = _component_features(class_mask)
        features[f"components_class{class_label}"] = count
        features[f"largest_component_fraction_class{class_label}"] = largest_fraction

    fraction_values = np.asarray(
        [value for name, value in features.items() if "fraction" in name],
        dtype=np.float64,
    )
    if (
        not np.isfinite(fraction_values).all()
        or np.any(fraction_values < 0)
        or np.any(fraction_values > 1)
    ):
        raise RuntimeError(f"morphology fractions outside [0, 1]: {features}")
    component_values = [
        value for name, value in features.items() if name.startswith("components_")
    ]
    if any(not isinstance(value, int) or value < 0 for value in component_values):
        raise RuntimeError(f"invalid component counts: {features}")
    return features


def extract_ground_truth_free_features(
    image: np.ndarray,
    probabilities: np.ndarray,
    hard_prediction: np.ndarray,
    *,
    probability_config: dict[str, Any],
    foreground_labels: tuple[int, ...] = (1, 2, 3),
) -> dict[str, int | float]:
    """Extract TASK-007 features from deployment-available arrays only."""
    support = image_support(image)
    probability_array = validate_probabilities(
        probabilities,
        image.shape,
        classes=int(probability_config["classes"]),
        class_axis=int(probability_config["class_axis"]),
        absolute_tolerance=float(probability_config["sum_absolute_tolerance"]),
        relative_tolerance=float(probability_config["sum_relative_tolerance"]),
    )
    prediction_array = validate_hard_prediction(hard_prediction, image.shape)
    return {
        **probability_features(
            probability_array,
            support,
            class_axis=int(probability_config["class_axis"]),
        ),
        **morphology_features(
            prediction_array,
            support,
            foreground_labels=foreground_labels,
        ),
    }
