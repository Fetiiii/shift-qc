"""TASK-031 uses synthetic/structural fixtures only."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from shiftqc.validation2.contracts import (
    FEATURE_COLUMNS,
    LABEL_INTEGER_TOLERANCE,
    ContractError,
    dice_for_required_organ,
    make_metadata_split,
    normalize_label_values,
    validate_feature_columns,
)


def test_feature_contract_has_exactly_18_label_free_features() -> None:
    assert len(FEATURE_COLUMNS) == 18
    assert validate_feature_columns(FEATURE_COLUMNS) == FEATURE_COLUMNS
    assert not any("dice" in name or "q_true" in name for name in FEATURE_COLUMNS)


def test_feature_validator_rejects_reorder_or_ground_truth_feature() -> None:
    with pytest.raises(ContractError):
        validate_feature_columns(tuple(reversed(FEATURE_COLUMNS)))
    with pytest.raises(ContractError):
        validate_feature_columns((*FEATURE_COLUMNS[:-1], "dice_liver"))


def test_label_normalization_and_invalid_values() -> None:
    values = np.array([0.0, 1.0 + 1e-7, 2.0 - 1e-7, 3.0, 4.0])
    np.testing.assert_array_equal(normalize_label_values(values), [0, 1, 2, 3, 4])
    with pytest.raises(ContractError):
        normalize_label_values(np.array([1.0 + 1.1e-5]))
    with pytest.raises(ContractError):
        normalize_label_values(np.array([np.nan]))
    with pytest.raises(ContractError):
        normalize_label_values(np.array([5.0]))


def test_label_tolerance_boundary_is_closed_and_outside_fails() -> None:
    np.testing.assert_array_equal(
        normalize_label_values(np.array([LABEL_INTEGER_TOLERANCE])),
        [0],
    )
    with pytest.raises(ContractError, match="beyond tolerance"):
        normalize_label_values(
            np.array([np.nextafter(LABEL_INTEGER_TOLERANCE, np.inf)])
        )


def test_required_organ_dice_synthetic_edges() -> None:
    ground_truth = np.array([1, 1, 0, 0])
    assert dice_for_required_organ(ground_truth, ground_truth, 1) == 1.0
    assert dice_for_required_organ(ground_truth, np.array([0, 0, 1, 1]), 1) == 0.0
    assert dice_for_required_organ(ground_truth, np.array([1, 0, 1, 0]), 1) == pytest.approx(0.5)
    with pytest.raises(ContractError, match="ground-truth organ is empty"):
        dice_for_required_organ(np.zeros(4), np.zeros(4), 1)


def test_metadata_split_is_deterministic_disjoint_and_exhaustive() -> None:
    ids = [f"p{index:02d}" for index in range(20)]
    provenance = ["MSD"] * 12 + ["NIH"] * 8
    sizes = {"SEG_DEV": 8, "QC_TRAIN": 4, "CALIBRATION": 4, "ID_EVAL": 4}
    first = make_metadata_split(ids, provenance, sizes, seed=2026)
    second = make_metadata_split(list(reversed(ids)), list(reversed(provenance)), sizes, seed=2026)
    assert first == second
    assert set(first) == set(ids)
    assert {role: list(first.values()).count(role) for role in sizes} == sizes


def test_metadata_split_rejects_duplicates_or_unresolved_provenance() -> None:
    sizes = {"SEG_DEV": 1, "QC_TRAIN": 1, "CALIBRATION": 1, "ID_EVAL": 1}
    with pytest.raises(ContractError, match="duplicate"):
        make_metadata_split(["p0", "p0", "p2", "p3"], ["A"] * 4, sizes, seed=2026)
    with pytest.raises(ContractError, match="resolved provenance"):
        make_metadata_split(
            ["p0", "p1", "p2", "p3"],
            ["A", "A", "A", "PROVENANCE_UNRESOLVED"],
            sizes,
            seed=2026,
        )


def test_structural_api_does_not_accept_scientific_outputs() -> None:
    for function in (validate_feature_columns, make_metadata_split):
        parameters = set(inspect.signature(function).parameters)
        assert not parameters & {"q_true", "q_hat", "dice", "prediction", "ground_truth"}
