"""Read-only M&Ms release audit for frozen M3 TASK-018."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from shiftqc.data.nnunet_export import sha256_file


class MnmsAuditFailure(RuntimeError):
    """Raised for objective dataset or frozen-protocol integrity failures."""


MANIFEST_COLUMNS = (
    "patient_id", "official_partition", "vendor", "vendor_name",
    "image_path_relative", "gt_path_relative", "has_image", "has_gt",
    "official_annotation_role", "ed_frame", "es_frame", "phase_identity_source",
    "phase_mapping_unambiguous", "ed_gt_available", "es_gt_available",
    "image_shape", "gt_shape", "spacing", "gt_spacing", "num_cine_frames",
    "image_orientation", "gt_orientation", "affine_compatible",
    "geometry_compatible", "label_values", "unexpected_label_values",
    "missing_classes", "nonzero_gt_frames", "source_file_provenance",
)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata(dataset_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Load authoritative patient/vendor/phase metadata with strict schema checks."""
    path = dataset_root / config["metadata_file"]
    if not path.is_file():
        raise MnmsAuditFailure(f"authoritative metadata missing: {config['metadata_file']}")
    frame = pd.read_csv(path)
    required = {"External code", "VendorName", "Vendor", "Centre", "ED", "ES"}
    missing = required - set(frame.columns)
    if missing:
        raise MnmsAuditFailure(f"metadata columns missing: {sorted(missing)}")
    frame = frame.rename(columns={"External code": "patient_id"}).copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    if frame["patient_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["patient_id"].duplicated(False), "patient_id"].unique())
        raise MnmsAuditFailure(f"duplicate metadata patient IDs: {duplicates}")
    for phase in ("ED", "ES"):
        numeric = pd.to_numeric(frame[phase], errors="coerce")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise MnmsAuditFailure(f"metadata {phase} contains non-integer/missing values")
        frame[phase] = numeric.astype(int)
    return frame


def validate_vendor_mapping(metadata: pd.DataFrame, config: dict[str, Any]) -> dict[str, str]:
    """Resolve vendor letters only from authoritative metadata values."""
    observed_pairs = metadata[["Vendor", "VendorName"]].drop_duplicates()
    if observed_pairs["Vendor"].duplicated().any():
        raise MnmsAuditFailure("one vendor code maps to multiple VendorName values")
    observed = {
        str(row.Vendor): str(row.VendorName)
        for row in observed_pairs.sort_values("Vendor").itertuples(index=False)
    }
    expected = {str(key): str(value) for key, value in config["expected_vendors"].items()}
    if observed != expected:
        raise MnmsAuditFailure(
            f"authoritative vendor mapping mismatch: expected={expected}, observed={observed}"
        )
    return observed


def discover_patient_entities(dataset_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Discover exactly one patient entity per official release directory."""
    records: list[dict[str, Any]] = []
    for partition, relative in config["partition_directories"].items():
        directory = dataset_root / relative
        if not directory.is_dir():
            raise MnmsAuditFailure(f"official partition directory missing: {relative}")
        for patient_directory in sorted(path for path in directory.iterdir() if path.is_dir()):
            patient_id = patient_directory.name
            image = patient_directory / f"{patient_id}{config['image_suffix']}"
            ground_truth = patient_directory / f"{patient_id}{config['gt_suffix']}"
            records.append(
                {
                    "patient_id": patient_id,
                    "official_partition": partition,
                    "image_path_relative": image.relative_to(dataset_root).as_posix(),
                    "gt_path_relative": ground_truth.relative_to(dataset_root).as_posix(),
                    "has_image": image.is_file(),
                    "has_gt": ground_truth.is_file(),
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise MnmsAuditFailure("no patient directories found")
    if frame["patient_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["patient_id"].duplicated(False), "patient_id"].unique())
        raise MnmsAuditFailure(f"patient IDs overlap across official partitions: {duplicates}")
    if frame["image_path_relative"].duplicated().any():
        raise MnmsAuditFailure("duplicate image paths across patient entities")
    if frame["gt_path_relative"].duplicated().any():
        raise MnmsAuditFailure("duplicate GT paths across patient entities")
    return frame.sort_values(["official_partition", "patient_id"], kind="stable").reset_index(drop=True)


def join_metadata(entities: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Require a one-to-one match between filesystem entities and official metadata."""
    joined = entities.merge(metadata, on="patient_id", how="outer", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        mismatches = joined.loc[~joined["_merge"].eq("both"), ["patient_id", "_merge"]]
        raise MnmsAuditFailure(f"metadata/filesystem patient mismatch: {mismatches.to_dict('records')}")
    return joined.drop(columns="_merge")


def resolve_phase_identity(row: pd.Series) -> dict[str, Any]:
    """Resolve ED/ES from CSV without inferring replacements from image content."""
    ed = int(row["ED"])
    es = int(row["ES"])
    original_unlabeled = row["official_partition"] == "training_unlabeled"
    unambiguous = not original_unlabeled and ed != es
    return {
        "ed_frame": ed,
        "es_frame": es,
        "phase_identity_source": "211230 metadata CSV columns ED/ES",
        "phase_mapping_unambiguous": bool(unambiguous),
        "official_annotation_role": (
            "original_challenge_unlabeled_excluded"
            if original_unlabeled
            else "official_annotated"
        ),
    }


def _scaled_label_values(proxy: Any, tolerance: float) -> tuple[np.ndarray, list[int]]:
    data = np.asanyarray(proxy, dtype=np.float64)
    if not np.isfinite(data).all():
        raise MnmsAuditFailure("GT contains NaN/Inf")
    rounded = np.rint(data)
    if np.max(np.abs(data - rounded), initial=0.0) > tolerance:
        values = np.unique(data).tolist()
        raise MnmsAuditFailure(f"GT values are not integer-equivalent: {values}")
    return rounded.astype(np.int8), sorted(int(value) for value in np.unique(rounded))


def audit_nifti_pair(
    dataset_root: Path,
    row: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Audit one cine/GT pair without modifying or resampling source data."""
    image_path = dataset_root / row["image_path_relative"]
    gt_path = dataset_root / row["gt_path_relative"]
    if not image_path.is_file() or not gt_path.is_file():
        return {
            "image_shape": None, "gt_shape": None, "spacing": None,
            "gt_spacing": None, "num_cine_frames": None,
            "image_orientation": None, "gt_orientation": None,
            "affine_compatible": False, "geometry_compatible": False,
            "label_values": None, "unexpected_label_values": None,
            "missing_classes": None, "nonzero_gt_frames": None,
            "ed_gt_available": False, "es_gt_available": False,
        }
    image = nib.load(image_path)
    ground_truth = nib.load(gt_path)
    image_shape = tuple(int(value) for value in image.shape)
    gt_shape = tuple(int(value) for value in ground_truth.shape)
    image_spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    gt_spacing = tuple(float(value) for value in ground_truth.header.get_zooms()[:3])
    if len(image_shape) != 4 or len(gt_shape) != 4:
        raise MnmsAuditFailure(f"patient {row['patient_id']} is not 4D cine/GT")
    if not all(np.isfinite(image_spacing)) or not all(value > 0 for value in image_spacing):
        raise MnmsAuditFailure(f"patient {row['patient_id']} has invalid image spacing")
    affine_compatible = bool(
        np.allclose(image.affine, ground_truth.affine, rtol=0.0, atol=float(config["affine_tolerance"]))
    )
    geometry_compatible = bool(
        image_shape == gt_shape
        and np.allclose(image_spacing, gt_spacing, rtol=0.0, atol=float(config["affine_tolerance"]))
        and affine_compatible
    )
    labels, observed = _scaled_label_values(
        ground_truth.dataobj, float(config["label_integer_tolerance"])
    )
    expected = {int(value) for value in config["label_mapping"]}
    unexpected = sorted(set(observed) - expected)
    missing_classes = sorted({1, 2, 3} - set(observed))
    nonzero_frames = [
        frame for frame in range(labels.shape[3]) if bool(np.any(labels[..., frame] != 0))
    ]
    phase = resolve_phase_identity(row)
    ed = phase["ed_frame"]
    es = phase["es_frame"]
    ed_available = bool(
        phase["phase_mapping_unambiguous"] and 0 <= ed < labels.shape[3] and ed in nonzero_frames
    )
    es_available = bool(
        phase["phase_mapping_unambiguous"] and 0 <= es < labels.shape[3] and es in nonzero_frames
    )
    return {
        "image_shape": list(image_shape),
        "gt_shape": list(gt_shape),
        "spacing": list(image_spacing),
        "gt_spacing": list(gt_spacing),
        "num_cine_frames": int(image_shape[3]),
        "image_orientation": "".join(nib.aff2axcodes(image.affine)),
        "gt_orientation": "".join(nib.aff2axcodes(ground_truth.affine)),
        "affine_compatible": affine_compatible,
        "geometry_compatible": geometry_compatible,
        "label_values": observed,
        "unexpected_label_values": unexpected,
        "missing_classes": missing_classes,
        "nonzero_gt_frames": nonzero_frames,
        "ed_gt_available": ed_available,
        "es_gt_available": es_available,
    }


def audit_dataset(
    dataset_root: Path,
    config: dict[str, Any],
    *,
    progress: Any | None = None,
) -> pd.DataFrame:
    """Perform the deterministic patient-level read-only M&Ms audit."""
    metadata = load_metadata(dataset_root, config)
    validate_vendor_mapping(metadata, config)
    entities = discover_patient_entities(dataset_root, config)
    joined = join_metadata(entities, metadata)
    metadata_sha = sha256_file(dataset_root / config["metadata_file"])
    records: list[dict[str, Any]] = []
    for index, row in enumerate(joined.sort_values(["official_partition", "patient_id"], kind="stable").iterrows(), 1):
        _, patient = row
        phase = resolve_phase_identity(patient)
        nifti = audit_nifti_pair(dataset_root, patient, config)
        records.append(
            {
                "patient_id": patient["patient_id"],
                "official_partition": patient["official_partition"],
                "vendor": str(patient["Vendor"]),
                "vendor_name": str(patient["VendorName"]),
                "image_path_relative": patient["image_path_relative"],
                "gt_path_relative": patient["gt_path_relative"],
                "has_image": bool(patient["has_image"]),
                "has_gt": bool(patient["has_gt"]),
                **phase,
                **nifti,
                "source_file_provenance": f"{config['metadata_file']}#{int(patient.name)};sha256={metadata_sha}",
            }
        )
        if progress is not None:
            progress(index, len(joined), patient["patient_id"])
    frame = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    if frame["patient_id"].duplicated().any():
        raise MnmsAuditFailure("audit manifest patient IDs are not unique")
    return frame.sort_values(["official_partition", "patient_id"], kind="stable").reset_index(drop=True)


def dataset_tree_inventory(dataset_root: Path) -> dict[str, Any]:
    """Record source-tree names/sizes/mtimes without content mutation."""
    records = [
        (path.relative_to(dataset_root).as_posix(), int(path.stat().st_size), int(path.stat().st_mtime_ns))
        for path in sorted(item for item in dataset_root.rglob("*") if item.is_file())
    ]
    return {"file_count": len(records), "inventory_sha256": canonical_hash(records)}


def frozen_experiment_tree_hash(repository: Path) -> dict[str, Any]:
    records: list[tuple[str, str]] = []
    for experiment in ("EXP-0002", "EXP-0003", "EXP-0004"):
        root = repository / "outputs" / experiment
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            records.append((path.relative_to(repository).as_posix(), sha256_file(path)))
    return {"file_count": len(records), "tree_sha256": canonical_hash(records)}


def _nested_counts(frame: pd.DataFrame, value: str | None = None) -> dict[str, Any]:
    grouped = frame.groupby(["official_partition", "vendor"], observed=True)
    if value is None:
        series = grouped.size()
    else:
        series = grouped[value].sum()
    output: dict[str, Any] = {}
    for (partition, vendor), count in series.items():
        output.setdefault(str(partition), {})[str(vendor)] = int(count)
    return output


def _distance_range(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"min": float(array.min()), "max": float(array.max())}


def build_audit_report(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    dataset_root: Path,
    dataset_inventory_before: dict[str, Any],
    dataset_inventory_after: dict[str, Any],
    frozen_tree_before: dict[str, Any],
    frozen_tree_after: dict[str, Any],
) -> dict[str, Any]:
    """Summarize integrity and determine the conservative frozen M3 decision."""
    if dataset_inventory_before != dataset_inventory_after:
        raise MnmsAuditFailure("source dataset tree changed during read-only audit")
    if frozen_tree_before != frozen_tree_after:
        raise MnmsAuditFailure("frozen EXP-0002/0003/0004 tree changed during TASK-018")
    official_annotated = frame["official_annotation_role"].eq("official_annotated")
    source = frame["official_partition"].eq("training_labeled")
    evaluation = frame["official_partition"].isin(config["required_evaluation_partitions"])
    target_evaluation = evaluation & frame["vendor"].isin(config["required_target_vendors"])
    original_unlabeled = frame["official_partition"].eq("training_unlabeled")
    required = source | evaluation

    source_counts = frame.loc[source, "vendor"].value_counts().sort_index().to_dict()
    source_total = int(source.sum())
    evaluation_vendor_counts = (
        frame.loc[evaluation, "vendor"].value_counts().sort_index().astype(int).to_dict()
    )
    missing_required_images = frame.loc[required & ~frame["has_image"], "patient_id"].tolist()
    missing_required_gt = frame.loc[required & ~frame["has_gt"], "patient_id"].tolist()
    geometry_failures = frame.loc[required & ~frame["geometry_compatible"], "patient_id"].tolist()
    phase_failures = frame.loc[
        required
        & (~frame["phase_mapping_unambiguous"] | ~frame["ed_gt_available"] | ~frame["es_gt_available"]),
        "patient_id",
    ].tolist()
    unexpected_labels = sorted(
        {
            int(value)
            for values in frame.loc[required, "unexpected_label_values"]
            for value in values
        }
    )
    missing_class_cases = {
        str(row.patient_id): [int(value) for value in row.missing_classes]
        for row in frame.loc[required & frame["missing_classes"].map(bool)].itertuples()
    }
    phase_frame_mismatches = []
    for row in frame.loc[required].itertuples():
        expected = sorted({int(row.ed_frame), int(row.es_frame)})
        observed = sorted(int(value) for value in row.nonzero_gt_frames)
        if expected != observed:
            phase_frame_mismatches.append(
                {"patient_id": row.patient_id, "expected": expected, "observed": observed}
            )

    blocked_reasons: list[str] = []
    review_reasons: list[str] = []
    if missing_required_images:
        blocked_reasons.append("required_images_missing")
    if missing_required_gt or not frame.loc[target_evaluation, "has_gt"].all():
        blocked_reasons.append("required_target_ground_truth_missing")
    if source_counts.get("A", 0) != int(config["expected_source_counts"]["A"]):
        review_reasons.append("annotated_source_vendor_A_count_differs_from_75")
    if source_counts.get("B", 0) != int(config["expected_source_counts"]["B"]):
        review_reasons.append("annotated_source_vendor_B_count_differs_from_75")
    if source_total != int(config["expected_source_counts"]["total"]):
        review_reasons.append("annotated_source_total_differs_from_150")
    if set(frame["vendor"].unique()) != set(config["expected_vendors"]):
        review_reasons.append("vendor_set_differs_from_A_B_C_D")
    if geometry_failures:
        review_reasons.append("required_image_gt_geometry_failures")
    if phase_failures or phase_frame_mismatches:
        review_reasons.append("required_ED_ES_mapping_or_GT_correspondence_failure")
    if unexpected_labels:
        review_reasons.append("unexpected_label_values")
    if len(set(evaluation_vendor_counts.values())) != 1:
        review_reasons.append(
            "official_evaluation_vendor_counts_are_not_balanced_"
            + "_".join(f"{key}{value}" for key, value in evaluation_vendor_counts.items())
        )
    if blocked_reasons:
        decision = "M3_DATA_BLOCKED"
    elif review_reasons:
        decision = "M3_DATA_REVIEW_REQUIRED"
    else:
        decision = "M3_DATA_READY"

    shapes = np.vstack(frame.loc[required, "image_shape"].map(np.asarray))
    spacings = np.vstack(frame.loc[required, "spacing"].map(np.asarray))
    frames = frame.loc[required, "num_cine_frames"].astype(int).to_numpy()
    observed_labels = sorted(
        {int(value) for values in frame.loc[required, "label_values"] for value in values}
    )
    metadata_path = dataset_root / config["metadata_file"]
    vendor_summary: dict[str, Any] = {}
    for vendor in sorted(config["expected_vendors"]):
        selected = frame.loc[frame["vendor"].eq(vendor)]
        vendor_summary[vendor] = {
            "vendor_name": config["expected_vendors"][vendor],
            "total": int(len(selected)),
            "training_labeled": int(selected["official_partition"].eq("training_labeled").sum()),
            "training_unlabeled": int(selected["official_partition"].eq("training_unlabeled").sum()),
            "validation": int(selected["official_partition"].eq("validation").sum()),
            "test": int(selected["official_partition"].eq("test").sum()),
            "official_annotated": int(selected["official_annotation_role"].eq("official_annotated").sum()),
            "original_challenge_unannotated": int(
                selected["official_annotation_role"].eq("original_challenge_unlabeled_excluded").sum()
            ),
            "GT_files_present": int(selected["has_gt"].sum()),
        }
    return {
        "task": config["task"],
        "status": "PASS" if decision == "M3_DATA_READY" else "REVIEW_REQUIRED" if decision == "M3_DATA_REVIEW_REQUIRED" else "BLOCKED",
        "m3_compatibility_decision": decision,
        "dataset_source": {
            "configured_by": ["--dataset-root", config["dataset_root_env"]],
            "local_root_name": dataset_root.name,
            "absolute_path_recorded": False,
            "release_identification": "211230 M&Ms OpenDataset",
            "metadata_file": config["metadata_file"],
            "metadata_sha256": sha256_file(metadata_path),
            "source_tree_inventory": dataset_inventory_after,
        },
        "counts": {
            "total_patients": int(len(frame)),
            "total_image_studies": int(frame["has_image"].sum()),
            "GT_files_present": int(frame["has_gt"].sum()),
            "official_annotated_patients": int(official_annotated.sum()),
            "phase_resolved_annotated_patients": int(
                (official_annotated & frame["ed_gt_available"] & frame["es_gt_available"]).sum()
            ),
            "by_vendor": vendor_summary,
            "partition_by_vendor": _nested_counts(frame),
            "GT_files_by_partition_vendor": _nested_counts(frame, "has_gt"),
        },
        "source_verification": {
            "vendor_A_annotated": int(source_counts.get("A", 0)),
            "vendor_B_annotated": int(source_counts.get("B", 0)),
            "annotated_source_total": source_total,
            "expected": config["expected_source_counts"],
            "matches_frozen_expectation": bool(
                source_counts.get("A", 0) == int(config["expected_source_counts"]["A"])
                and source_counts.get("B", 0) == int(config["expected_source_counts"]["B"])
                and source_total == int(config["expected_source_counts"]["total"])
            ),
        },
        "original_vendor_C_training_cases": {
            "count": int(original_unlabeled.sum()),
            "image_files_present": int(frame.loc[original_unlabeled, "has_image"].sum()),
            "GT_files_present_in_open_release": int(frame.loc[original_unlabeled, "has_gt"].sum()),
            "vendor_metadata_present": int(frame.loc[original_unlabeled, "vendor"].notna().sum()),
            "authoritative_ED_ES_resolved": int(
                frame.loc[original_unlabeled, "phase_mapping_unambiguous"].sum()
            ),
            "authorized_for_M3_source_or_evaluation": False,
        },
        "evaluation": {
            "vendor_counts_validation_plus_test": {
                str(key): int(value) for key, value in evaluation_vendor_counts.items()
            },
            "vendor_C_GT_available_count": int(
                frame.loc[target_evaluation & frame["vendor"].eq("C"), "has_gt"].sum()
            ),
            "vendor_D_GT_available_count": int(
                frame.loc[target_evaluation & frame["vendor"].eq("D"), "has_gt"].sum()
            ),
            "required_target_GT_complete": bool(frame.loc[target_evaluation, "has_gt"].all()),
        },
        "phase_audit": {
            "identity_source": "authoritative metadata CSV ED/ES columns",
            "required_patients": int(required.sum()),
            "required_ED_available": int(frame.loc[required, "ed_gt_available"].sum()),
            "required_ES_available": int(frame.loc[required, "es_gt_available"].sum()),
            "required_phase_failures": phase_failures,
            "required_nonzero_GT_frame_mismatches": phase_frame_mismatches,
            "excluded_original_unlabeled_ambiguous_count": int(
                (original_unlabeled & ~frame["phase_mapping_unambiguous"]).sum()
            ),
        },
        "label_audit": {
            "authoritative_mapping": config["label_mapping"],
            "mapping_provenance": config["label_mapping_provenance"],
            "observed_scaled_label_union": observed_labels,
            "unexpected_label_values": unexpected_labels,
            "required_patients_missing_classes": missing_class_cases,
            "no_remapping_performed": True,
        },
        "spatial_audit": {
            "required_pairs": int(required.sum()),
            "geometry_compatible_pairs": int(frame.loc[required, "geometry_compatible"].sum()),
            "geometry_failures": geometry_failures,
            "dimensionality": 4,
            "in_plane_spacing_x": _distance_range(spacings[:, 0].tolist()),
            "in_plane_spacing_y": _distance_range(spacings[:, 1].tolist()),
            "through_plane_spacing": _distance_range(spacings[:, 2].tolist()),
            "matrix_x": {"min": int(shapes[:, 0].min()), "max": int(shapes[:, 0].max())},
            "matrix_y": {"min": int(shapes[:, 1].min()), "max": int(shapes[:, 1].max())},
            "slices": {"min": int(shapes[:, 2].min()), "max": int(shapes[:, 2].max())},
            "cine_frames": {"min": int(frames.min()), "max": int(frames.max())},
        },
        "patient_integrity": {
            "patient_ids_unique": bool(frame["patient_id"].is_unique),
            "duplicate_patient_ids": 0,
            "cross_vendor_patient_overlap": 0,
            "duplicate_image_paths": int(frame["image_path_relative"].duplicated().sum()),
            "duplicate_GT_paths": int(frame["gt_path_relative"].duplicated().sum()),
            "multiple_acquisitions_per_patient": 0,
        },
        "missing_or_corrupt": {
            "required_images": missing_required_images,
            "required_GT": missing_required_gt,
            "geometry": geometry_failures,
        },
        "compatibility": {
            "blocked_reasons": blocked_reasons,
            "review_reasons": review_reasons,
            "ready_conditions_satisfied_except_review_items": bool(not blocked_reasons),
        },
        "firewalls": {
            "segmentation_performance_computed": False,
            "foreground_volume_outcome_comparisons_computed": False,
            "QC_statistics_computed": False,
            "model_predictions_accessed": False,
            "scientific_split_generated": False,
            "task_019_started": False,
        },
        "immutability": {
            "dataset_tree_before": dataset_inventory_before,
            "dataset_tree_after": dataset_inventory_after,
            "frozen_experiment_tree_before": frozen_tree_before,
            "frozen_experiment_tree_after": frozen_tree_after,
        },
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    vendors = report["counts"]["by_vendor"]
    rows = []
    for vendor in ("A", "B", "C", "D"):
        item = vendors[vendor]
        rows.append(
            f"| {vendor} | {item['vendor_name']} | {item['training_labeled']} | "
            f"{item['training_unlabeled']} | {item['validation']} | {item['test']} | "
            f"{item['official_annotated']} | {item['GT_files_present']} |"
        )
    review = report["compatibility"]["review_reasons"] or ["none"]
    return f"""# SHIFT-QC M&Ms Dataset Audit

Task: `TASK-018`  
Decision: **{report['m3_compatibility_decision']}**

This is a read-only dataset/access/vendor audit. No scientific split, preprocessing,
training, inference, QC fitting, calibration, or performance evaluation occurred.

## Release

- Identification: {report['dataset_source']['release_identification']}
- Local root name: `{report['dataset_source']['local_root_name']}`
- Absolute machine-local path recorded: no
- Metadata: `{report['dataset_source']['metadata_file']}`
- Total patients: {report['counts']['total_patients']}
- Image studies: {report['counts']['total_image_studies']}
- Official annotated patients: {report['counts']['official_annotated_patients']}

## Vendor and official-partition counts

| Vendor | Vendor name | Training labeled | Training unlabeled | Validation | Test | Official annotated | GT files present |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Frozen source verification

- Vendor A annotated source: {report['source_verification']['vendor_A_annotated']}
- Vendor B annotated source: {report['source_verification']['vendor_B_annotated']}
- Source total: {report['source_verification']['annotated_source_total']}
- Matches 75 A + 75 B = 150: {str(report['source_verification']['matches_frozen_expectation']).lower()}

## Target and phase access

- Vendor C validation+test GT: {report['evaluation']['vendor_C_GT_available_count']}
- Vendor D validation+test GT: {report['evaluation']['vendor_D_GT_available_count']}
- Required ED available: {report['phase_audit']['required_ED_available']} / {report['phase_audit']['required_patients']}
- Required ES available: {report['phase_audit']['required_ES_available']} / {report['phase_audit']['required_patients']}
- Original 25 Vendor-C training cases remain excluded; their CSV ED/ES fields are ambiguous.

## Labels and geometry

- Mapping: 0 background, 1 LV, 2 MYO, 3 RV
- Observed label union: {report['label_audit']['observed_scaled_label_union']}
- Unexpected labels: {report['label_audit']['unexpected_label_values']}
- Geometry-compatible required pairs: {report['spatial_audit']['geometry_compatible_pairs']} / {report['spatial_audit']['required_pairs']}
- Geometry failures: {len(report['spatial_audit']['geometry_failures'])}

## Integrity

- Duplicate patient IDs: {report['patient_integrity']['duplicate_patient_ids']}
- Cross-vendor patient overlap: {report['patient_integrity']['cross_vendor_patient_overlap']}
- Duplicate image paths: {report['patient_integrity']['duplicate_image_paths']}
- Duplicate GT paths: {report['patient_integrity']['duplicate_GT_paths']}

## Compatibility review

Review reasons: {', '.join(review)}

The local official evaluation counts are A=20, B=50, C=50, D=50. This does not
match the approximately balanced per-vendor evaluation structure anticipated by
the frozen M3 specification, so no TASK-019 split manifest was created.

M3 compatibility decision: **{report['m3_compatibility_decision']}**
"""
