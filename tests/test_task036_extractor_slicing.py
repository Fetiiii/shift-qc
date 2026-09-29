"""Bit-identity of the sliced feature extractor against the whole-array reference.

The frozen extractor held the full probability tensor, a full log tensor and their
full product at once. On the largest Validation-2 case that is about 31 GB and the
kernel OOM killer stopped the run. Slicing bounds the temporaries. Every operation is
independent per voxel and reduces over the five classes in class order, so the values
cannot change; these tests pin that rather than assert it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shiftqc.validation2 import extract_qc_features
from shiftqc.validation2.contracts import FEATURE_COLUMNS
from shiftqc.validation2.features import (
    CLASS_NAMES,
    FEATURE_SLICE,
    PROBABILITY_SUM_ATOL,
    PROBABILITY_SUM_RTOL,
)


def reference_extract(probabilities: np.ndarray) -> dict[str, float]:
    """The whole-array implementation as it stood before slicing, verbatim."""
    values = np.asarray(probabilities, dtype=np.float64)
    totals = np.sum(values, axis=0, dtype=np.float64)
    assert np.allclose(totals, 1.0, rtol=PROBABILITY_SUM_RTOL, atol=PROBABILITY_SUM_ATOL)
    log_values = np.zeros_like(values)
    positive = values > 0.0
    log_values[positive] = np.log(values[positive])
    entropy = -np.sum(values * log_values, axis=0, dtype=np.float64)
    confidence = np.max(values, axis=0)
    segmentation = np.argmax(values, axis=0)
    voxel_count = segmentation.size
    row: dict[str, float] = {
        "entropy_mean": float(np.mean(entropy, dtype=np.float64)),
        "entropy_p90": float(np.percentile(entropy, 90, method="linear")),
        "entropy_p95": float(np.percentile(entropy, 95, method="linear")),
        "confidence_mean": float(np.mean(confidence, dtype=np.float64)),
        "confidence_p10": float(np.percentile(confidence, 10, method="linear")),
        "foreground_fraction": float(np.count_nonzero(segmentation) / voxel_count),
    }
    fractions, components, largest = {}, {}, {}
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for label, organ in enumerate(CLASS_NAMES, start=1):
        mask = segmentation == label
        count = int(np.count_nonzero(mask))
        fractions[organ] = float(count / voxel_count)
        if count == 0:
            components[organ] = 0.0
            largest[organ] = 0.0
            continue
        labelled, number = ndimage.label(mask, structure=structure)
        sizes = np.bincount(labelled.ravel())[1:]
        components[organ] = float(number)
        largest[organ] = float(np.max(sizes) / count)
    for organ in CLASS_NAMES:
        row[f"class_fraction_{organ}"] = fractions[organ]
    for organ in CLASS_NAMES:
        row[f"components_{organ}"] = components[organ]
    for organ in CLASS_NAMES:
        row[f"largest_component_fraction_{organ}"] = largest[organ]
    return row


def _softmax(shape, seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(5, *shape))
    exponent = np.exp(logits - logits.max(axis=0, keepdims=True))
    return exponent / exponent.sum(axis=0, keepdims=True)


@pytest.mark.parametrize("shape", [
    (FEATURE_SLICE, 6, 7),          # exactly one slice
    (FEATURE_SLICE * 3, 5, 5),      # exact multiple
    (FEATURE_SLICE * 2 + 7, 6, 6),  # ragged tail
    (1, 8, 8),                      # shorter than one slice
    (FEATURE_SLICE + 1, 4, 4),      # tail of one
])
def test_sliced_extractor_is_bit_identical(shape) -> None:
    values = _softmax(shape, seed=hash(shape) % 10_000)
    produced = extract_qc_features(values)
    expected = reference_extract(values)
    assert tuple(produced) == FEATURE_COLUMNS
    for name in FEATURE_COLUMNS:
        assert produced[name] == expected[name], name


def test_bit_identical_on_degenerate_probability_patterns() -> None:
    """Zeros, ones and ties are where a rewrite would most plausibly diverge."""
    shape = (FEATURE_SLICE * 2 + 3, 5, 5)
    one_hot = np.zeros((5, *shape), dtype=np.float64)
    one_hot[1] = 1.0                                  # exact zeros and ones
    tied = np.full((5, *shape), 0.2, dtype=np.float64)  # every class tied
    mixed = _softmax(shape, 7)
    mixed[:, 0] = 0.0
    mixed[3, 0] = 1.0
    for values in (one_hot, tied, mixed):
        produced = extract_qc_features(values)
        expected = reference_extract(values)
        for name in FEATURE_COLUMNS:
            assert produced[name] == expected[name], name


def test_contract_violations_still_fail_before_any_quantity_is_computed() -> None:
    from shiftqc.validation2.contracts import ContractError
    shape = (FEATURE_SLICE * 2, 4, 4)
    bad = _softmax(shape, 3).copy()
    bad[0, -1, 0, 0] = np.nan                     # violation in the final slice
    with pytest.raises(ContractError, match="finite"):
        extract_qc_features(bad)
    unnormalized = _softmax(shape, 4) * 0.5
    with pytest.raises(ContractError, match="sum to one"):
        extract_qc_features(unnormalized)
    with pytest.raises(ContractError, match="5 x X x Y x Z"):
        extract_qc_features(np.zeros((4, 3, 3, 3)))


def test_slicing_bounds_the_temporaries() -> None:
    source = (ROOT / "src/shiftqc/validation2/features.py").read_text(encoding="utf-8")
    body = source[source.index("def extract_qc_features"):source.index("    row: dict")]
    assert "log_values = np.zeros_like(values)" not in body    # the 10 GB allocation
    assert "np.sum(values * log_values" not in body            # the 10 GB temporary
    assert "log_chunk = np.zeros_like(chunk)" in body
    assert "FEATURE_SLICE" in body
    assert FEATURE_SLICE >= 1
