"""Synthetic tests for the frozen TASK-021 QC inputs.

No real M&Ms prediction, ground truth or cohort summary is read. These fix the
codepaths and the firewall between them before any real artefact is produced.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shiftqc.qc.m3_inputs import (
    EXPECTED_POPULATION_PATIENTS,
    FROZEN_FEATURE_ORDER,
    PHASES,
    QC_POPULATIONS,
    Task021Failure,
    aggregate_to_patient,
    case_q_true,
    case_qc_features,
    frozen_label_map,
)

nib = pytest.importorskip("nibabel")
PROB_CONFIG = {"array_key": "probabilities", "class_axis": 0, "classes": 4,
               "sum_absolute_tolerance": 1e-5, "sum_relative_tolerance": 1e-5,
               "entropy_logarithm": "natural"}
SHAPE = (4, 6, 5)


def _save4d(path: Path, volume: np.ndarray, frames: int = 3, frame: int = 1) -> None:
    series = np.zeros((*volume.shape, frames), dtype=np.float32)
    series[..., frame] = volume
    nib.save(nib.Nifti1Image(series, np.eye(4)), path)


def _save3d(path: Path, volume: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(volume.astype(np.uint8), np.eye(4)), path)


# ------------------------------------------------------------------ q_true

def test_perfect_prediction_gives_dice_one(tmp_path: Path) -> None:
    gt = np.zeros(SHAPE, dtype=np.uint8)
    gt[0, :3, :3], gt[1, :3, :3], gt[2, :3, :3] = 1, 2, 3
    _save4d(tmp_path/"gt.nii.gz", gt); _save3d(tmp_path/"pred.nii.gz", gt)
    v = case_q_true(tmp_path/"gt.nii.gz", 1, tmp_path/"pred.nii.gz",
                    label_map={"LV":1,"MYO":2,"RV":3})
    assert v == pytest.approx({"dice_lv":1.0,"dice_myo":1.0,"dice_rv":1.0,"dice_macro":1.0})


def test_both_empty_class_gives_one_and_one_empty_gives_zero(tmp_path: Path) -> None:
    gt = np.zeros(SHAPE, dtype=np.uint8); gt[0, :3, :3] = 1        # RV absent in both
    pred = np.zeros(SHAPE, dtype=np.uint8); pred[0, :3, :3] = 1
    pred[1, :2, :2] = 2                                            # MYO only in prediction
    _save4d(tmp_path/"gt.nii.gz", gt); _save3d(tmp_path/"pred.nii.gz", pred)
    v = case_q_true(tmp_path/"gt.nii.gz", 1, tmp_path/"pred.nii.gz",
                    label_map={"LV":1,"MYO":2,"RV":3})
    assert v["dice_lv"] == pytest.approx(1.0)
    assert v["dice_myo"] == pytest.approx(0.0)     # exactly one empty
    assert v["dice_rv"] == pytest.approx(1.0)      # both empty
    assert v["dice_macro"] == pytest.approx(2.0/3.0)


def test_known_overlap_matches_analytic_dice(tmp_path: Path) -> None:
    gt = np.zeros(SHAPE, dtype=np.uint8); gt[0, 0, :4] = 1
    pred = np.zeros(SHAPE, dtype=np.uint8); pred[0, 0, 2:6] = 1     # |A|=4 |B|=3 overlap 2
    _save4d(tmp_path/"gt.nii.gz", gt); _save3d(tmp_path/"pred.nii.gz", pred)
    v = case_q_true(tmp_path/"gt.nii.gz", 1, tmp_path/"pred.nii.gz",
                    label_map={"LV":1,"MYO":2,"RV":3})
    assert v["dice_lv"] == pytest.approx(2*2/(4+3))


def test_class_macro_is_unweighted_mean(tmp_path: Path) -> None:
    gt = np.zeros(SHAPE, dtype=np.uint8)
    gt[0, :4, 0], gt[1, :4, 1], gt[2, :4, 2] = 1, 2, 3
    pred = gt.copy(); pred[0, 2:4, 0] = 0                           # LV half missed
    _save4d(tmp_path/"gt.nii.gz", gt); _save3d(tmp_path/"pred.nii.gz", pred)
    v = case_q_true(tmp_path/"gt.nii.gz", 1, tmp_path/"pred.nii.gz",
                    label_map={"LV":1,"MYO":2,"RV":3})
    assert v["dice_macro"] == pytest.approx((v["dice_lv"]+v["dice_myo"]+v["dice_rv"])/3)


def test_shape_mismatch_and_bad_labels_are_hard_failures(tmp_path: Path) -> None:
    gt = np.zeros(SHAPE, dtype=np.uint8); gt[0, 0, 0] = 1
    _save4d(tmp_path/"gt.nii.gz", gt)
    _save3d(tmp_path/"small.nii.gz", np.zeros((2,2,2), dtype=np.uint8))
    with pytest.raises(Task021Failure) as e:
        case_q_true(tmp_path/"gt.nii.gz", 1, tmp_path/"small.nii.gz",
                    label_map={"LV":1,"MYO":2,"RV":3})
    assert e.value.code == "prediction_reference_shape_mismatch"

    bad = np.zeros(SHAPE, dtype=np.uint8); bad[0, 0, 0] = 7
    _save4d(tmp_path/"bad.nii.gz", bad); _save3d(tmp_path/"pred.nii.gz", np.zeros(SHAPE, np.uint8))
    with pytest.raises(Task021Failure) as e:
        case_q_true(tmp_path/"bad.nii.gz", 1, tmp_path/"pred.nii.gz",
                    label_map={"LV":1,"MYO":2,"RV":3})
    assert e.value.code == "unexpected_ground_truth_labels"


def test_phase_frame_out_of_range_fails(tmp_path: Path) -> None:
    _save4d(tmp_path/"gt.nii.gz", np.zeros(SHAPE, np.uint8))
    _save3d(tmp_path/"pred.nii.gz", np.zeros(SHAPE, np.uint8))
    with pytest.raises(Task021Failure) as e:
        case_q_true(tmp_path/"gt.nii.gz", 99, tmp_path/"pred.nii.gz",
                    label_map={"LV":1,"MYO":2,"RV":3})
    assert e.value.code == "phase_frame_out_of_range"


# ----------------------------------------------------------------- features

def _feature_case(tmp_path: Path, prob: np.ndarray, pred: np.ndarray,
                  image: np.ndarray | None = None) -> dict[str, float]:
    """Arrays are given in SimpleITK order, which is how the feature path reads them.

    nibabel writes an array as (x, y, z) and SimpleITK reads it back as (z, y, x), so
    the files are written transposed and SimpleITK then sees exactly ``SHAPE``. The
    probability archive needs no transpose: nnU-Net writes it in SimpleITK order.
    """
    img = np.ones(SHAPE, dtype=np.float32) if image is None else image
    _save3d(tmp_path/"pred.nii.gz", pred.T)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(img.T), np.eye(4)), tmp_path/"img.nii.gz")
    np.savez_compressed(tmp_path/"prob.npz", probabilities=prob)
    return case_qc_features(tmp_path/"img.nii.gz", tmp_path/"prob.npz",
                            tmp_path/"pred.nii.gz", probability_config=PROB_CONFIG)


def test_uniform_probability_gives_maximum_entropy(tmp_path: Path) -> None:
    prob = np.full((4, *SHAPE), 0.25, dtype=np.float32)
    f = _feature_case(tmp_path, prob, np.zeros(SHAPE, np.uint8))
    assert f["entropy_mean"] == pytest.approx(np.log(4), abs=1e-5)
    assert f["confidence_mean"] == pytest.approx(0.25, abs=1e-6)


def test_one_hot_probability_gives_zero_entropy(tmp_path: Path) -> None:
    prob = np.zeros((4, *SHAPE), dtype=np.float32); prob[0] = 1.0
    f = _feature_case(tmp_path, prob, np.zeros(SHAPE, np.uint8))
    assert f["entropy_mean"] == pytest.approx(0.0, abs=1e-9)
    assert f["confidence_mean"] == pytest.approx(1.0, abs=1e-9)
    assert f["confidence_p10"] == pytest.approx(1.0, abs=1e-9)


def test_known_fractions_and_components(tmp_path: Path) -> None:
    prob = np.zeros((4, *SHAPE), dtype=np.float32); prob[0] = 1.0
    pred = np.zeros(SHAPE, dtype=np.uint8)
    pred[0, 0, 0] = 1; pred[0, 2, 0] = 1                # two separated LV components
    pred[1, 0, :2] = 2                                  # one MYO component, 2 voxels
    f = _feature_case(tmp_path, prob, pred)
    n = float(np.prod(SHAPE))                           # image all ones -> full support
    assert f["class1_fraction"] == pytest.approx(2/n)
    assert f["class2_fraction"] == pytest.approx(2/n)
    assert f["class3_fraction"] == pytest.approx(0.0)
    assert f["foreground_fraction"] == pytest.approx(4/n)
    assert f["components_class1"] == 2
    assert f["components_class2"] == 1
    assert f["components_class3"] == 0
    assert f["largest_component_fraction_class1"] == pytest.approx(0.5)
    assert f["largest_component_fraction_class2"] == pytest.approx(1.0)
    assert f["largest_component_fraction_class3"] == pytest.approx(0.0)


def test_feature_order_is_canonical_and_complete(tmp_path: Path) -> None:
    prob = np.full((4, *SHAPE), 0.25, dtype=np.float32)
    f = _feature_case(tmp_path, prob, np.zeros(SHAPE, np.uint8))
    assert tuple(f) == FROZEN_FEATURE_ORDER
    assert len(FROZEN_FEATURE_ORDER) == 15


def test_non_finite_and_wrong_class_count_fail(tmp_path: Path) -> None:
    bad = np.full((4, *SHAPE), 0.25, dtype=np.float32); bad[0, 0, 0, 0] = np.nan
    with pytest.raises(Exception):
        _feature_case(tmp_path, bad, np.zeros(SHAPE, np.uint8))
    with pytest.raises(Exception):
        _feature_case(tmp_path, np.full((3, *SHAPE), 1/3, np.float32), np.zeros(SHAPE, np.uint8))


def test_feature_extraction_is_deterministic(tmp_path: Path) -> None:
    prob = np.random.default_rng(2026).dirichlet(np.ones(4), size=SHAPE).transpose(3,0,1,2).astype(np.float32)
    pred = np.zeros(SHAPE, dtype=np.uint8); pred[0, :2, :2] = 1
    a = _feature_case(tmp_path, prob, pred)
    b = _feature_case(tmp_path, prob, pred)
    assert a == b


# ----------------------------------------------------------------- FIREWALL

def test_feature_signature_cannot_accept_ground_truth() -> None:
    """Structural firewall: no parameter through which a label could enter."""
    params = set(inspect.signature(case_qc_features).parameters)
    for forbidden in ("ground_truth", "gt", "reference", "q_true", "dice", "label", "target"):
        assert not any(forbidden in p for p in params), (forbidden, params)
    assert params == {"input_image_path", "probability_path", "hard_prediction_path",
                      "probability_config"}


def test_features_are_invariant_to_ground_truth(tmp_path: Path) -> None:
    """Same prediction and probabilities, different ground truth: identical features."""
    prob = np.full((4, *SHAPE), 0.25, dtype=np.float32)
    pred = np.zeros(SHAPE, dtype=np.uint8); pred[0, :2, :2] = 1
    first = _feature_case(tmp_path, prob, pred)
    gt_a = np.zeros(SHAPE, dtype=np.uint8); gt_a[0, :2, :2] = 1
    gt_b = np.zeros(SHAPE, dtype=np.uint8); gt_b[2, 3:, 3:] = 3
    for gt in (gt_a, gt_b):
        _save4d(tmp_path/"gt.nii.gz", gt)
        assert _feature_case(tmp_path, prob, pred) == first


def test_feature_extraction_works_without_any_ground_truth_file(tmp_path: Path) -> None:
    """No ground-truth file exists anywhere in the working directory."""
    prob = np.full((4, *SHAPE), 0.25, dtype=np.float32)
    f = _feature_case(tmp_path, prob, np.zeros(SHAPE, np.uint8))
    assert not list(tmp_path.glob("*gt*"))
    assert len(f) == 15


# -------------------------------------------------------------- aggregation

def _cohort(value: float = 0.9) -> pd.DataFrame:
    rows = []
    for population, n in EXPECTED_POPULATION_PATIENTS.items():
        for i in range(n):
            for phase in PHASES:
                rows.append({"patient_id": f"{population}_{i:03d}", "phase": phase,
                             "population": population, "vendor": "A", "dice_macro": value})
    return pd.DataFrame(rows)


def test_ed_es_aggregation_is_the_mean(tmp_path: Path) -> None:
    df = _cohort()
    df.loc[df.patient_id == "QC_TRAIN_000", "dice_macro"] = [0.6, 0.8]
    out = aggregate_to_patient(df, ["dice_macro"])
    assert out.loc[out.patient_id == "QC_TRAIN_000", "dice_macro"].iloc[0] == pytest.approx(0.7)
    assert len(out) == sum(EXPECTED_POPULATION_PATIENTS.values()) == 230


def test_missing_phase_and_duplicates_are_hard_failures() -> None:
    df = _cohort()
    with pytest.raises(Task021Failure) as e:
        aggregate_to_patient(df[df.phase == "ED"], ["dice_macro"])
    assert e.value.code == "patient_missing_phase"

    dup = pd.concat([df, df[df.patient_id == "QC_TRAIN_000"]], ignore_index=True)
    with pytest.raises(Task021Failure) as e:
        aggregate_to_patient(dup, ["dice_macro"])
    assert e.value.code == "duplicate_patient_phase"


def test_unexpected_population_and_count_are_hard_failures() -> None:
    df = _cohort()
    other = df[df.patient_id == "QC_TRAIN_000"].copy()
    other["population"] = "SEG_DEV"; other["patient_id"] = "SEG_DEV_000"
    with pytest.raises(Task021Failure) as e:
        aggregate_to_patient(pd.concat([df, other], ignore_index=True), ["dice_macro"])
    assert e.value.code == "unexpected_population"

    short = df[df.patient_id != "QC_TRAIN_000"]
    with pytest.raises(Task021Failure) as e:
        aggregate_to_patient(short, ["dice_macro"])
    assert e.value.code == "unexpected_population_patient_count"


def test_non_finite_case_value_is_a_hard_failure() -> None:
    df = _cohort()
    df.loc[0, "dice_macro"] = np.nan
    with pytest.raises(Task021Failure) as e:
        aggregate_to_patient(df, ["dice_macro"])
    assert e.value.code == "non_finite_case_value"


def test_frozen_label_map_reads_the_dataset(tmp_path: Path) -> None:
    p = tmp_path/"dataset.json"
    p.write_text(json.dumps({"labels": {"background":0,"LV":1,"MYO":2,"RV":3}}))
    assert frozen_label_map(p) == {"LV":1,"MYO":2,"RV":3}
    p.write_text(json.dumps({"labels": {"background":0,"LV":2,"MYO":1,"RV":3}}))
    with pytest.raises(Task021Failure) as e:
        frozen_label_map(p)
    assert e.value.code == "unexpected_label_mapping"


def test_populations_are_the_four_qc_cohorts() -> None:
    assert QC_POPULATIONS == ("QC_TRAIN", "CALIBRATION", "M3_ID_EVAL", "M3_OOD_EVAL")
    assert "SEG_DEV" not in QC_POPULATIONS and "EXCLUDED" not in QC_POPULATIONS
