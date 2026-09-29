"""Label-free 18-feature QC extractor frozen for Validation-2."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .contracts import FEATURE_COLUMNS, ContractError


PROBABILITY_SUM_ATOL = 1e-6
PROBABILITY_SUM_RTOL = 1e-5
CLASS_NAMES = ("liver", "kidney", "spleen", "pancreas")

# Slices of the first spatial axis processed at a time. Bounds peak memory only; the
# feature values are unchanged, which is pinned by a bit-identity test against the
# whole-array reference implementation.
FEATURE_SLICE = 16


def extract_qc_features(probabilities: np.ndarray) -> dict[str, float]:
    """Compute the ordered label-free feature row from a 5×X×Y×Z softmax."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 4 or values.shape[0] != 5 or any(size <= 0 for size in values.shape[1:]):
        raise ContractError("probabilities must have shape 5 x X x Y x Z")

    # Validation and the voxelwise quantities are computed in slices along the first
    # spatial axis. Every operation here is independent per voxel and reduces over the
    # five classes in class order, so slicing cannot change a single result: the same
    # five terms are combined the same way, and the full-size entropy, confidence and
    # segmentation arrays that the reductions below consume are identical to the ones
    # the whole-array form produced. Only the peak memory changes. On the largest
    # Validation-2 case the whole-array form held values, log_values and the
    # values*log_values temporary at once, about 31 GB, and the kernel OOM killer
    # stopped the run; the sliced form holds one slice of each instead.
    #
    # Validation runs as its own pass first so that a case which violates a contract
    # still fails before any quantity is computed, exactly as before.
    depth = values.shape[1]
    for start in range(0, depth, FEATURE_SLICE):
        chunk = values[:, start:min(start + FEATURE_SLICE, depth)]
        if not np.isfinite(chunk).all() or np.any((chunk < 0.0) | (chunk > 1.0)):
            raise ContractError("probabilities must be finite and lie in [0, 1]")
    for start in range(0, depth, FEATURE_SLICE):
        chunk = values[:, start:min(start + FEATURE_SLICE, depth)]
        totals = np.sum(chunk, axis=0, dtype=np.float64)
        if not np.allclose(totals, 1.0, rtol=PROBABILITY_SUM_RTOL, atol=PROBABILITY_SUM_ATOL):
            raise ContractError("softmax probabilities do not sum to one")

    entropy = np.empty(values.shape[1:], dtype=np.float64)
    confidence = np.empty(values.shape[1:], dtype=np.float64)
    segmentation = np.empty(values.shape[1:], dtype=np.int64)
    for start in range(0, depth, FEATURE_SLICE):
        stop = min(start + FEATURE_SLICE, depth)
        chunk = values[:, start:stop]
        log_chunk = np.zeros_like(chunk)
        positive = chunk > 0.0
        log_chunk[positive] = np.log(chunk[positive])
        entropy[start:stop] = -np.sum(chunk * log_chunk, axis=0, dtype=np.float64)
        confidence[start:stop] = np.max(chunk, axis=0)
        segmentation[start:stop] = np.argmax(chunk, axis=0)
        del chunk, log_chunk, positive
    voxel_count = segmentation.size

    row: dict[str, float] = {
        "entropy_mean": float(np.mean(entropy, dtype=np.float64)),
        "entropy_p90": float(np.percentile(entropy, 90, method="linear")),
        "entropy_p95": float(np.percentile(entropy, 95, method="linear")),
        "confidence_mean": float(np.mean(confidence, dtype=np.float64)),
        "confidence_p10": float(np.percentile(confidence, 10, method="linear")),
        "foreground_fraction": float(np.count_nonzero(segmentation) / voxel_count),
    }
    class_fractions: dict[str, float] = {}
    component_counts: dict[str, float] = {}
    largest_fractions: dict[str, float] = {}
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for label, organ in enumerate(CLASS_NAMES, start=1):
        mask = segmentation == label
        count = int(np.count_nonzero(mask))
        class_fractions[organ] = float(count / voxel_count)
        if count == 0:
            component_counts[organ] = 0.0
            largest_fractions[organ] = 0.0
            continue
        components, number = ndimage.label(mask, structure=structure)
        sizes = np.bincount(components.ravel())[1:]
        component_counts[organ] = float(number)
        largest_fractions[organ] = float(np.max(sizes) / count)

    for organ in CLASS_NAMES:
        row[f"class_fraction_{organ}"] = class_fractions[organ]
    for organ in CLASS_NAMES:
        row[f"components_{organ}"] = component_counts[organ]
    for organ in CLASS_NAMES:
        row[f"largest_component_fraction_{organ}"] = largest_fractions[organ]

    if tuple(row) != FEATURE_COLUMNS or not np.isfinite(list(row.values())).all():
        raise ContractError("feature extractor violated the frozen schema")
    return row
