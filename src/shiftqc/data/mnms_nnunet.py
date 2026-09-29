"""Leakage-safe M&Ms ED/ES export and nnU-Net planning audit for TASK-020A."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import nibabel as nib
import numpy as np
import pandas as pd

from shiftqc.data.mnms_audit import canonical_hash
from shiftqc.data.nnunet_export import sha256_file


PHASES = ("ED", "ES")
ALLOWED_SOURCE_SPLITS = ("SEG_TRAIN", "SEG_VAL")
CASE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_]+__(ED|ES)$")
SPLIT_COLUMNS = (
    "patient_id",
    "vendor",
    "official_partition",
    "scientific_role",
    "source_split",
    "image_path_relative",
    "gt_path_relative",
    "ed_frame",
    "es_frame",
)
CONVERSION_COLUMNS = (
    "patient_id",
    "vendor",
    "source_split",
    "phase",
    "phase_index",
    "nnunet_case_identifier",
    "source_image_relative",
    "source_gt_relative",
    "exported_image_relative",
    "exported_label_relative",
    "spatial_shape",
    "spacing_xyz",
    "observed_labels",
    "image_sha256",
    "label_sha256",
)


class MnmsNnunetFailure(RuntimeError):
    """Raised for TASK-020A export, leakage, or planning integrity failures."""

    def __init__(self, code: str, details: Any = None):
        self.code = code
        self.details = details
        super().__init__(f"{code}: {details}")


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_inputs(repository: Path, config: dict[str, Any]) -> None:
    checks = (
        (config["m3_spec"]["path"], config["m3_spec"]["sha256"], "M3_SPEC"),
        (
            config["frozen_split"]["parquet"],
            config["frozen_split"]["parquet_sha256"],
            "TASK-019 Parquet",
        ),
        (
            config["frozen_split"]["json"],
            config["frozen_split"]["json_sha256"],
            "TASK-019 JSON",
        ),
    )
    for relative, expected, label in checks:
        path = repository / relative
        if not path.is_file():
            raise MnmsNnunetFailure("frozen_input_missing", {"label": label, "path": relative})
        observed = sha256_file(path)
        if observed != expected:
            raise MnmsNnunetFailure(
                "frozen_input_hash_mismatch",
                {"label": label, "expected": expected, "observed": observed},
            )


def load_segmentation_patients(repository: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Load only frozen manifest metadata and select SEG_TRAIN/SEG_VAL patients."""
    verify_frozen_inputs(repository, config)
    split_path = repository / config["frozen_split"]["parquet"]
    inventory = pd.read_parquet(split_path, columns=list(SPLIT_COLUMNS))
    if len(inventory) != 345 or not inventory["patient_id"].is_unique:
        raise MnmsNnunetFailure(
            "frozen_inventory_integrity_failure",
            {"rows": len(inventory), "unique": int(inventory["patient_id"].nunique())},
        )
    selected = inventory.loc[
        inventory["source_split"].isin(ALLOWED_SOURCE_SPLITS), list(SPLIT_COLUMNS)
    ].copy()
    expected_counts = {
        name: int(value)
        for name, value in config["segmentation_splits"]["expected_patients"].items()
    }
    observed_counts = selected["source_split"].value_counts().to_dict()
    if observed_counts != expected_counts:
        raise MnmsNnunetFailure(
            "segmentation_patient_count_mismatch",
            {"expected": expected_counts, "observed": observed_counts},
        )
    if len(selected) != sum(expected_counts.values()) or not selected["patient_id"].is_unique:
        raise MnmsNnunetFailure("segmentation_patient_integrity_failure", len(selected))
    if not selected["scientific_role"].eq("SOURCE").all():
        raise MnmsNnunetFailure("non_source_patient_selected", None)
    if not selected["official_partition"].eq("training_labeled").all():
        raise MnmsNnunetFailure("non_training_labeled_patient_selected", None)
    observed_vendor = (
        selected.groupby(["source_split", "vendor"], observed=True).size().to_dict()
    )
    expected_vendor = {
        (split, vendor): int(count)
        for split, counts in config["segmentation_splits"]["expected_vendor_counts"].items()
        for vendor, count in counts.items()
    }
    if observed_vendor != expected_vendor:
        raise MnmsNnunetFailure(
            "segmentation_vendor_count_mismatch",
            {"expected": expected_vendor, "observed": observed_vendor},
        )
    return selected.sort_values("patient_id", kind="stable", ignore_index=True)


def build_case_table(patients: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Expand each patient into exactly one authoritative ED and one ES case."""
    records: list[dict[str, Any]] = []
    phase_columns = config["phase_policy"]["authoritative_columns"]
    for row in patients.itertuples(index=False):
        for phase in config["phase_policy"]["phases"]:
            phase_index = int(getattr(row, phase_columns[phase]))
            case_id = f"{row.patient_id}__{phase}"
            if not CASE_IDENTIFIER_PATTERN.fullmatch(case_id):
                raise MnmsNnunetFailure("unsafe_case_identifier", case_id)
            records.append(
                {
                    "patient_id": str(row.patient_id),
                    "vendor": str(row.vendor),
                    "source_split": str(row.source_split),
                    "phase": phase,
                    "phase_index": phase_index,
                    "nnunet_case_identifier": case_id,
                    "source_image_relative": str(row.image_path_relative),
                    "source_gt_relative": str(row.gt_path_relative),
                }
            )
    cases = pd.DataFrame.from_records(records).sort_values(
        ["patient_id", "phase"], kind="stable", ignore_index=True
    )
    validate_case_table(cases, patients, config)
    return cases


def validate_case_table(
    cases: pd.DataFrame, patients: pd.DataFrame, config: dict[str, Any]
) -> None:
    expected_total = len(patients) * len(PHASES)
    if len(cases) != expected_total or cases["nnunet_case_identifier"].duplicated().any():
        raise MnmsNnunetFailure(
            "case_table_integrity_failure",
            {"expected": expected_total, "observed": len(cases)},
        )
    per_patient = cases.groupby("patient_id", observed=True)["phase"].agg(list)
    invalid = {
        patient_id: phases
        for patient_id, phases in per_patient.items()
        if set(phases) != set(PHASES) or len(phases) != len(PHASES)
    }
    if invalid:
        raise MnmsNnunetFailure("patient_phase_pair_failure", invalid)
    split_nunique = cases.groupby("patient_id", observed=True)["source_split"].nunique()
    if not split_nunique.eq(1).all():
        raise MnmsNnunetFailure("patient_phase_split_leakage", None)
    expected_cases = {
        split: int(count) * len(PHASES)
        for split, count in config["segmentation_splits"]["expected_patients"].items()
    }
    observed_cases = cases["source_split"].value_counts().to_dict()
    if observed_cases != expected_cases:
        raise MnmsNnunetFailure(
            "phase_case_count_mismatch",
            {"expected": expected_cases, "observed": observed_cases},
        )


def build_custom_split(cases: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, list[str]]]:
    training = config["segmentation_splits"]["training"]
    validation = config["segmentation_splits"]["validation"]
    train = sorted(
        cases.loc[cases["source_split"].eq(training), "nnunet_case_identifier"].tolist()
    )
    val = sorted(
        cases.loc[cases["source_split"].eq(validation), "nnunet_case_identifier"].tolist()
    )
    if set(train) & set(val) or set(train) | set(val) != set(cases["nnunet_case_identifier"]):
        raise MnmsNnunetFailure("custom_split_case_leakage", None)
    train_patients = {case.rsplit("__", 1)[0] for case in train}
    val_patients = {case.rsplit("__", 1)[0] for case in val}
    if train_patients & val_patients:
        raise MnmsNnunetFailure("custom_split_patient_leakage", sorted(train_patients & val_patients))
    return [{"train": train, "val": val}]


def dataset_json(config: dict[str, Any], number_of_cases: int) -> dict[str, Any]:
    return {
        "channel_names": config["channel_names"],
        "labels": config["labels"],
        "numTraining": int(number_of_cases),
        "file_ending": config["file_ending"],
        "name": config["dataset_name"],
        "description": "SHIFT-QC M3 M&Ms frozen SEG_TRAIN/SEG_VAL ED+ES export",
        "source_split_manifest": config["frozen_split"]["parquet"],
        "case_naming": config["phase_policy"]["case_naming"],
    }


def _safe_source(dataset_root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise MnmsNnunetFailure("unsafe_source_relative_path", relative)
    root = dataset_root.resolve()
    source = (root / relative).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise MnmsNnunetFailure("source_file_missing_or_escaping", relative)
    return source


def _save_3d_volume(
    source_image: nib.spatialimages.SpatialImage,
    volume: np.ndarray,
    destination: Path,
    *,
    dtype: np.dtype[Any],
) -> None:
    header = source_image.header.copy()
    header.set_data_shape(volume.shape)
    header.set_data_dtype(dtype)
    output = nib.Nifti1Image(np.asarray(volume, dtype=dtype), source_image.affine, header)
    qform, qcode = source_image.get_qform(coded=True)
    sform, scode = source_image.get_sform(coded=True)
    if qform is not None:
        output.set_qform(qform, int(qcode))
    if sform is not None:
        output.set_sform(sform, int(scode))
    nib.save(output, destination)


def extract_phase_case(
    image_source: Path,
    gt_source: Path,
    phase_index: int,
    image_destination: Path,
    label_destination: Path,
    expected_labels: set[int],
    label_integer_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Extract one authoritative 3D frame without spatial manipulation."""
    image = nib.load(image_source)
    gt = nib.load(gt_source)
    if len(image.shape) != 4 or len(gt.shape) != 4 or image.shape != gt.shape:
        raise MnmsNnunetFailure(
            "source_4d_shape_mismatch",
            {"image": image.shape, "gt": gt.shape, "file": image_source.name},
        )
    if phase_index < 0 or phase_index >= image.shape[3]:
        raise MnmsNnunetFailure(
            "phase_index_out_of_range",
            {"phase_index": phase_index, "frames": image.shape[3], "file": image_source.name},
        )
    if not np.allclose(image.affine, gt.affine, rtol=0.0, atol=1e-5):
        raise MnmsNnunetFailure("source_image_gt_affine_mismatch", image_source.name)
    image_volume = np.asanyarray(image.dataobj[..., phase_index])
    gt_volume = np.asanyarray(gt.dataobj[..., phase_index])
    if not np.isfinite(image_volume).all():
        raise MnmsNnunetFailure("nonfinite_image_values", image_source.name)
    rounded_gt = np.rint(gt_volume)
    if (
        not np.isfinite(gt_volume).all()
        or float(np.max(np.abs(gt_volume - rounded_gt))) > label_integer_tolerance
    ):
        raise MnmsNnunetFailure("invalid_noninteger_gt", gt_source.name)
    observed_labels = sorted(int(value) for value in np.unique(rounded_gt))
    unexpected = sorted(set(observed_labels) - expected_labels)
    if unexpected:
        raise MnmsNnunetFailure(
            "unexpected_gt_labels", {"file": gt_source.name, "labels": unexpected}
        )
    image_dtype = np.dtype(image.get_data_dtype())
    _save_3d_volume(image, image_volume, image_destination, dtype=image_dtype)
    _save_3d_volume(gt, rounded_gt, label_destination, dtype=np.dtype(np.uint8))

    saved_image = nib.load(image_destination)
    saved_label = nib.load(label_destination)
    expected_shape = tuple(int(value) for value in image.shape[:3])
    if saved_image.shape != expected_shape or saved_label.shape != expected_shape:
        raise MnmsNnunetFailure("saved_phase_shape_mismatch", image_source.name)
    if not np.allclose(saved_image.affine, image.affine, rtol=0.0, atol=1e-5):
        raise MnmsNnunetFailure("saved_image_geometry_changed", image_source.name)
    if not np.allclose(saved_label.affine, gt.affine, rtol=0.0, atol=1e-5):
        raise MnmsNnunetFailure("saved_label_geometry_changed", gt_source.name)
    saved_labels = sorted(int(value) for value in np.unique(np.asanyarray(saved_label.dataobj)))
    if saved_labels != observed_labels:
        raise MnmsNnunetFailure(
            "saved_label_values_changed",
            {"expected": observed_labels, "observed": saved_labels},
        )
    spacing = [float(value) for value in image.header.get_zooms()[:3]]
    return {
        "spatial_shape": list(expected_shape),
        "spacing_xyz": spacing,
        "observed_labels": observed_labels,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def export_dataset(
    repository: Path,
    dataset_root: Path,
    nnunet_raw: Path,
    conversion_manifest_path: Path,
    config: dict[str, Any],
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Create Dataset502 atomically from the 90 frozen SEG_DEV patients."""
    patients = load_segmentation_patients(repository, config)
    cases = build_case_table(patients, config)
    dataset_directory = nnunet_raw / config["dataset_directory"]
    staging = nnunet_raw / f".{config['dataset_directory']}.partial"
    if dataset_directory.exists() or staging.exists():
        raise MnmsNnunetFailure(
            "dataset502_already_exists",
            {"dataset": str(dataset_directory), "partial": staging.exists()},
        )
    images_tr = staging / "imagesTr"
    labels_tr = staging / "labelsTr"
    images_tr.mkdir(parents=True)
    labels_tr.mkdir(parents=True)
    expected_labels = {int(value) for value in config["labels"].values()}
    records: list[dict[str, Any]] = []
    try:
        grouped = {row.patient_id: row for row in patients.itertuples(index=False)}
        for index, case in enumerate(cases.itertuples(index=False), start=1):
            patient = grouped[case.patient_id]
            image_source = _safe_source(dataset_root, case.source_image_relative)
            gt_source = _safe_source(dataset_root, case.source_gt_relative)
            image_relative = f"imagesTr/{case.nnunet_case_identifier}_0000.nii.gz"
            label_relative = f"labelsTr/{case.nnunet_case_identifier}.nii.gz"
            image_destination = staging / image_relative
            label_destination = staging / label_relative
            integrity = extract_phase_case(
                image_source,
                gt_source,
                int(case.phase_index),
                image_destination,
                label_destination,
                expected_labels,
                float(config["label_integer_tolerance"]),
            )
            records.append(
                {
                    **case._asdict(),
                    "exported_image_relative": image_relative,
                    "exported_label_relative": label_relative,
                    **integrity,
                    "image_sha256": sha256_file(image_destination),
                    "label_sha256": sha256_file(label_destination),
                }
            )
            if progress is not None:
                progress(index, len(cases), case.nnunet_case_identifier)
        conversion = pd.DataFrame.from_records(records).loc[:, list(CONVERSION_COLUMNS)]
        conversion = conversion.sort_values(
            "nnunet_case_identifier", kind="stable", ignore_index=True
        )
        if sorted({label for labels in conversion["observed_labels"] for label in labels}) != [0, 1, 2, 3]:
            raise MnmsNnunetFailure("exported_dataset_label_union_mismatch", None)
        _write_json(staging / "dataset.json", dataset_json(config, len(conversion)))
        _write_json(staging / "splits_final.json", build_custom_split(cases, config))
        os.replace(staging, dataset_directory)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    conversion_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = conversion_manifest_path.with_suffix(conversion_manifest_path.suffix + ".tmp")
    conversion.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(conversion_manifest_path)
    verify_export(dataset_directory, conversion, config)
    return dataset_directory, conversion


def verify_export(
    dataset_directory: Path, conversion: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    """Verify raw files, split grouping, labels, and patient-level visibility."""
    if tuple(conversion.columns) != CONVERSION_COLUMNS or len(conversion) != 180:
        raise MnmsNnunetFailure("conversion_manifest_schema_or_count_failure", len(conversion))
    expected_cases = set(conversion["nnunet_case_identifier"])
    images = {
        path.name.removesuffix("_0000.nii.gz")
        for path in (dataset_directory / "imagesTr").glob("*_0000.nii.gz")
    }
    labels = {
        path.name.removesuffix(".nii.gz")
        for path in (dataset_directory / "labelsTr").glob("*.nii.gz")
    }
    if images != expected_cases or labels != expected_cases:
        raise MnmsNnunetFailure(
            "raw_case_set_mismatch",
            {"images": len(images), "labels": len(labels), "expected": len(expected_cases)},
        )
    split = json.loads((dataset_directory / "splits_final.json").read_text(encoding="utf-8"))
    expected_split = build_custom_split(conversion, config)
    if split != expected_split or len(split) != 1:
        raise MnmsNnunetFailure("raw_custom_split_mismatch", None)
    dataset = json.loads((dataset_directory / "dataset.json").read_text(encoding="utf-8"))
    if dataset != dataset_json(config, 180):
        raise MnmsNnunetFailure("raw_dataset_json_mismatch", None)
    observed_union: set[int] = set()
    for row in conversion.itertuples(index=False):
        image_path = dataset_directory / row.exported_image_relative
        label_path = dataset_directory / row.exported_label_relative
        if sha256_file(image_path) != row.image_sha256 or sha256_file(label_path) != row.label_sha256:
            raise MnmsNnunetFailure("raw_case_hash_mismatch", row.nnunet_case_identifier)
        image = nib.load(image_path)
        label = nib.load(label_path)
        if image.shape != label.shape or not np.allclose(image.affine, label.affine, atol=1e-5, rtol=0.0):
            raise MnmsNnunetFailure("raw_image_label_geometry_mismatch", row.nnunet_case_identifier)
        labels_in_case = {int(value) for value in np.unique(np.asanyarray(label.dataobj))}
        if not labels_in_case.issubset({0, 1, 2, 3}):
            raise MnmsNnunetFailure("raw_unexpected_label", row.nnunet_case_identifier)
        observed_union.update(labels_in_case)
    if observed_union != {0, 1, 2, 3}:
        raise MnmsNnunetFailure("raw_label_union_mismatch", sorted(observed_union))
    return {
        "images": len(images),
        "labels": len(labels),
        "cases": len(expected_cases),
        "patients": int(conversion["patient_id"].nunique()),
        "label_union": sorted(observed_union),
    }


def patient_hash(values: Any) -> str:
    return canonical_hash(sorted(str(value) for value in values))


def case_hash(values: Any) -> str:
    return canonical_hash(sorted(str(value) for value in values))


def conversion_scientific_hash(conversion: pd.DataFrame) -> str:
    scientific = conversion.drop(columns=["image_sha256", "label_sha256"])
    return hashlib.sha256(
        scientific.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def raw_spacing_summary(conversion: pd.DataFrame) -> dict[str, Any]:
    spacings = np.asarray(conversion["spacing_xyz"].tolist(), dtype=np.float64)
    in_plane_per_case = spacings[:, :2].mean(axis=1)
    through_plane = spacings[:, 2]
    ratios = through_plane / in_plane_per_case
    return {
        "coordinate_order": "x_y_z_from_NIfTI_header",
        "cases": int(len(spacings)),
        "median_spacing_xyz": np.median(spacings, axis=0).tolist(),
        "median_in_plane_spacing": float(np.median(in_plane_per_case)),
        "median_through_plane_spacing": float(np.median(through_plane)),
        "median_anisotropy_ratio_through_to_in_plane": float(np.median(ratios)),
        "in_plane_spacing_range": [float(in_plane_per_case.min()), float(in_plane_per_case.max())],
        "through_plane_spacing_range": [float(through_plane.min()), float(through_plane.max())],
        "anisotropy_ratio_range": [float(ratios.min()), float(ratios.max())],
    }


def visibility_audit(repository: Path, conversion: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    inventory = pd.read_parquet(
        repository / config["frozen_split"]["parquet"],
        columns=["patient_id", "official_partition", "scientific_role", "source_split"],
    )
    visible = set(conversion["patient_id"])
    categories = {
        "QC_TRAIN": set(inventory.loc[inventory["source_split"].eq("QC_TRAIN"), "patient_id"]),
        "CALIBRATION": set(inventory.loc[inventory["source_split"].eq("CALIBRATION"), "patient_id"]),
        "M3_ID_EVAL": set(inventory.loc[inventory["scientific_role"].eq("M3_ID_EVAL"), "patient_id"]),
        "M3_OOD_EVAL": set(inventory.loc[inventory["scientific_role"].eq("M3_OOD_EVAL"), "patient_id"]),
        "EXCLUDED_VENDOR_C_TRAINING": set(
            inventory.loc[inventory["official_partition"].eq("training_unlabeled"), "patient_id"]
        ),
    }
    breakdown = {name: len(visible & patient_ids) for name, patient_ids in categories.items()}
    if any(breakdown.values()):
        raise MnmsNnunetFailure("forbidden_patient_visible_to_nnunet", breakdown)
    return {"forbidden_patient_counts": breakdown, "total_forbidden_visible": 0}


def dataset_id_conflicts(repository: Path, config: dict[str, Any]) -> list[str]:
    prefix = f"Dataset{int(config['dataset_id']):03d}_"
    conflicts: list[str] = []
    for root in config["nnunet_roots"].values():
        directory = repository / root
        if directory.is_dir():
            for candidate in directory.glob(f"{prefix}*"):
                if candidate.name != config["dataset_directory"]:
                    conflicts.append(candidate.relative_to(repository).as_posix())
    return sorted(conflicts)


def resolve_planned_configuration(
    name: str, configurations: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    value = configurations[name]
    parent = value.get("inherits_from")
    if parent is None:
        return dict(value)
    resolved = resolve_planned_configuration(parent, configurations)
    resolved.update(value)
    return resolved


def summarize_plans(plans: dict[str, Any], planner_vram_target_gb: float) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    configurations = plans["configurations"]
    for name in sorted(configurations):
        resolved = resolve_planned_configuration(name, configurations)
        architecture = resolved.get("architecture", {})
        kwargs = architecture.get("arch_kwargs", {})
        summaries[name] = {
            "configuration_name": name,
            "inherits_from": configurations[name].get("inherits_from"),
            "target_spacing": resolved.get("spacing"),
            "median_resampled_shape": resolved.get("median_image_size_in_voxels"),
            "patch_size": resolved.get("patch_size"),
            "batch_size": resolved.get("batch_size"),
            "architecture": architecture.get("network_class_name"),
            "number_of_stages": kwargs.get("n_stages"),
            "feature_widths": kwargs.get("features_per_stage"),
            "convolution_kernels": kwargs.get("kernel_sizes"),
            "pooling_strides": kwargs.get("strides"),
            "normalization": {
                "network": kwargs.get("norm_op"),
                "intensity": resolved.get("normalization_schemes"),
                "use_mask": resolved.get("use_mask_for_norm"),
            },
            "planner_reference_gpu_memory_target_gb": float(planner_vram_target_gb),
            "explicit_configuration_memory_estimate": None,
        }
    return summaries
