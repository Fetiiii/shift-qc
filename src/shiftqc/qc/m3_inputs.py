"""TASK-021 — frozen M&Ms QC inputs: patient-level ``q_true`` and label-free features.

Two codepaths that never meet. :func:`case_q_true` reads ground truth and may never
be called by the feature path; :func:`case_qc_features` takes no ground-truth
argument and therefore cannot read one. The separation is structural, not a
convention, and is asserted by ``tests/test_task021_m3_qc_inputs.py``.

Nothing here defines a metric or a feature. Both sides call the already frozen
implementations:

* ``shiftqc.metrics.segmentation.dice_per_class`` — TASK-006 Dice, including the
  empty-mask convention (both empty gives 1.0, exactly one empty gives 0.0).
* ``shiftqc.features.ground_truth_free.extract_ground_truth_free_features`` —
  the TASK-007 fifteen-feature family, with the TASK-007 probability config.

Aggregation order is fixed by ``M3_SPEC`` and is not a choice made here:

* section 10, quality:  ``macroDice_phase = mean(Dice_LV, Dice_MYO, Dice_RV)``
  then ``q_true_patient = mean(macroDice_ED, macroDice_ES)``.
* section 16, features: ``f_patient = mean(f_ED, f_ES)``.

Label mapping is read from the dataset rather than assumed, so a silent relabelling
cannot swap the per-class columns.

Array ordering: the frozen TASK-006 driver reads prediction and reference through
SimpleITK, while the M&Ms ground truth is a 4D series from which one frame must be
taken, which nibabel does directly. Dice is a voxelwise set operation, so any
ordering shared by both arrays gives an identical result; this module therefore
reads both sides of the q_true comparison with nibabel and the report records an
explicit equivalence check against the SimpleITK path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
import pandas as pd

from shiftqc.features.ground_truth_free import extract_ground_truth_free_features
from shiftqc.metrics.segmentation import dice_per_class

QC_POPULATIONS: tuple[str, ...] = ("QC_TRAIN", "CALIBRATION", "M3_ID_EVAL", "M3_OOD_EVAL")
EXPECTED_POPULATION_PATIENTS: Mapping[str, int] = {
    "QC_TRAIN": 30, "CALIBRATION": 30, "M3_ID_EVAL": 70, "M3_OOD_EVAL": 100,
}
PHASES: tuple[str, ...] = ("ED", "ES")
FOREGROUND_LABELS: tuple[int, ...] = (1, 2, 3)

# The canonical column order of the frozen fifteen-feature family, exactly as the
# TASK-008 config lists it. The downstream Ridge consumes these names.
FROZEN_FEATURE_ORDER: tuple[str, ...] = (
    "entropy_mean", "entropy_p90", "entropy_p95", "confidence_mean", "confidence_p10",
    "foreground_fraction", "class1_fraction", "class2_fraction", "class3_fraction",
    "components_class1", "components_class2", "components_class3",
    "largest_component_fraction_class1", "largest_component_fraction_class2",
    "largest_component_fraction_class3",
)
Q_TRUE_CASE_COLUMNS: tuple[str, ...] = (
    "dice_lv", "dice_myo", "dice_rv", "dice_macro",
)


class Task021Failure(RuntimeError):
    """Raised when a TASK-021 input violates the frozen design. Never repaired."""

    def __init__(self, code: str, details: Any = None) -> None:
        super().__init__(f"{code}: {details}" if details is not None else code)
        self.code = code
        self.details = details


def frozen_label_map(dataset_json: Path) -> dict[str, int]:
    """Read LV/MYO/RV onto their integer labels from the dataset, never hardcoded."""
    labels = json.loads(Path(dataset_json).read_text(encoding="utf-8")).get("labels") or {}
    mapping: dict[str, int] = {}
    for name in ("LV", "MYO", "RV"):
        if name not in labels:
            raise Task021Failure("dataset_class_missing", {"class": name, "labels": labels})
        mapping[name] = int(labels[name])
    if tuple(mapping[n] for n in ("LV", "MYO", "RV")) != FOREGROUND_LABELS:
        raise Task021Failure("unexpected_label_mapping", mapping)
    return mapping


def _phase_frame(path: Path, frame: int) -> tuple[np.ndarray, Any]:
    """Take one 3D frame out of a 4D series, exactly as the staging did."""
    image = nib.load(path)
    if len(image.shape) != 4:
        raise Task021Failure("source_not_4d", {"path": str(path), "shape": image.shape})
    if not 0 <= int(frame) < image.shape[3]:
        raise Task021Failure("phase_frame_out_of_range",
                             {"path": str(path), "frame": int(frame), "frames": image.shape[3]})
    return np.asanyarray(image.dataobj[..., int(frame)]), image


# --------------------------------------------------------------------------- A
# q_true codepath. Reads ground truth. Must never be reachable from the feature path.

def case_q_true(
    ground_truth_source: Path,
    phase_frame: int,
    hard_prediction_path: Path,
    *,
    label_map: Mapping[str, int],
    label_integer_tolerance: float = 1e-5,
) -> dict[str, float]:
    """Per-case class Dice and its macro, using the frozen TASK-006 implementation."""
    gt_volume, _ = _phase_frame(Path(ground_truth_source), phase_frame)
    rounded = np.rint(gt_volume)
    if (not np.isfinite(gt_volume).all()
            or float(np.max(np.abs(gt_volume - rounded))) > label_integer_tolerance):
        raise Task021Failure("invalid_noninteger_ground_truth", str(ground_truth_source))
    reference = rounded.astype(np.int16)
    observed = {int(v) for v in np.unique(reference)}
    unexpected = sorted(observed - {0, *FOREGROUND_LABELS})
    if unexpected:
        raise Task021Failure("unexpected_ground_truth_labels",
                             {"path": str(ground_truth_source), "labels": unexpected})

    prediction = np.asanyarray(nib.load(hard_prediction_path).dataobj)
    if prediction.shape != reference.shape:
        raise Task021Failure("prediction_reference_shape_mismatch",
                             {"prediction": prediction.shape, "reference": reference.shape})
    prediction = np.rint(prediction).astype(np.int16)
    predicted_labels = {int(v) for v in np.unique(prediction)}
    if not predicted_labels.issubset({0, *FOREGROUND_LABELS}):
        raise Task021Failure("unexpected_prediction_labels", sorted(predicted_labels))

    per_class = dice_per_class(reference, prediction, FOREGROUND_LABELS)
    values = {
        "dice_lv": per_class[label_map["LV"]],
        "dice_myo": per_class[label_map["MYO"]],
        "dice_rv": per_class[label_map["RV"]],
    }
    # M3_SPEC section 10: the class macro is taken first, within the phase.
    values["dice_macro"] = float(np.mean(tuple(per_class.values()), dtype=np.float64))
    return values


# --------------------------------------------------------------------------- B
# Label-free codepath. Takes no ground truth, no q_true and no Dice, by signature.

def case_qc_features(
    input_image_path: Path,
    probability_path: Path,
    hard_prediction_path: Path,
    *,
    probability_config: Mapping[str, Any],
) -> dict[str, float]:
    """Extract the frozen fifteen features from deployment-available arrays only.

    The parameters are the whole contract: an image, a probability map and a hard
    prediction. There is no argument through which ground truth could enter.
    """
    import SimpleITK as sitk  # frozen TASK-007 reader; array order (z, y, x)

    image_object = sitk.ReadImage(str(input_image_path))
    prediction_object = sitk.ReadImage(str(hard_prediction_path))
    image = sitk.GetArrayFromImage(image_object)
    hard_prediction = sitk.GetArrayFromImage(prediction_object)
    if image.shape != hard_prediction.shape:
        raise Task021Failure("image_prediction_shape_mismatch",
                             {"image": image.shape, "prediction": hard_prediction.shape})
    with np.load(probability_path, allow_pickle=False) as archive:
        key = str(probability_config["array_key"])
        if key not in archive.files:
            raise Task021Failure("probability_array_missing",
                                 {"path": str(probability_path), "key": key})
        probabilities = archive[key]
    features = extract_ground_truth_free_features(
        image, probabilities, hard_prediction,
        probability_config=dict(probability_config),
        foreground_labels=FOREGROUND_LABELS,
    )
    missing = set(FROZEN_FEATURE_ORDER) - set(features)
    if missing:
        raise Task021Failure("feature_missing", sorted(missing))
    unexpected = set(features) - set(FROZEN_FEATURE_ORDER)
    if unexpected:
        raise Task021Failure("unexpected_feature", sorted(unexpected))
    return {name: features[name] for name in FROZEN_FEATURE_ORDER}


# ----------------------------------------------------------------- aggregation

def _validate_cases(cases: pd.DataFrame, value_columns: Sequence[str]) -> None:
    duplicated = cases.duplicated(subset=["patient_id", "phase"], keep=False)
    if bool(duplicated.any()):
        raise Task021Failure("duplicate_patient_phase",
                             sorted({str(v) for v in cases.loc[duplicated, "patient_id"]}))
    unexpected_phase = sorted(set(cases["phase"].astype(str)) - set(PHASES))
    if unexpected_phase:
        raise Task021Failure("unexpected_phase", unexpected_phase)
    phases = cases.groupby("patient_id")["phase"].agg(
        lambda values: tuple(sorted(str(v) for v in values)))
    incomplete = sorted(str(p) for p, got in phases.items() if got != tuple(sorted(PHASES)))
    if incomplete:
        raise Task021Failure("patient_missing_phase", incomplete)
    values = cases[list(value_columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise Task021Failure("non_finite_case_value", int((~np.isfinite(values)).sum()))
    unexpected_population = sorted(set(cases["population"]) - set(QC_POPULATIONS))
    if unexpected_population:
        raise Task021Failure("unexpected_population", unexpected_population)


def aggregate_to_patient(cases: pd.DataFrame, value_columns: Sequence[str]) -> pd.DataFrame:
    """Mean over ED and ES per patient. The only aggregation M3_SPEC allows."""
    _validate_cases(cases, value_columns)
    grouped = (
        cases.groupby(["patient_id", "population", "vendor"], sort=True, observed=True)
        .agg(**{c: (c, "mean") for c in value_columns})
        .reset_index()
    )
    counts = grouped["population"].value_counts().to_dict()
    for population, expected in EXPECTED_POPULATION_PATIENTS.items():
        if counts.get(population, 0) != expected:
            raise Task021Failure("unexpected_population_patient_count",
                                 {"population": population, "expected": expected,
                                  "observed": counts.get(population, 0)})
    return grouped
