"""Leakage-safe nnU-Net v2 export for the frozen M0 segmentation splits."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd


class ExportFailure(RuntimeError):
    """A hard TASK-004 dataset-export failure."""

    def __init__(self, code: str, details: Any):
        self.code = code
        self.details = details
        super().__init__(f"{code}: {details}")


def load_export_config(path: Path) -> dict[str, Any]:
    """Load versioned nnU-Net export parameters."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    """Compute a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_directory_name(config: dict[str, Any]) -> str:
    return f"Dataset{int(config['dataset_id']):03d}_{config['dataset_name']}"


def validate_frozen_split_files(repository: Path, config: dict[str, Any]) -> None:
    """Refuse export if either frozen split artifact changed byte-for-byte."""
    paths_and_hashes = (
        (config["frozen_split_json"], config["frozen_split_json_sha256"]),
        (config["frozen_split_parquet"], config["frozen_split_parquet_sha256"]),
    )
    for relative_path, expected_hash in paths_and_hashes:
        path = repository / relative_path
        if not path.is_file():
            raise ExportFailure("frozen_split_missing", relative_path)
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise ExportFailure(
                "frozen_split_hash_mismatch",
                {
                    "path": relative_path,
                    "expected": expected_hash,
                    "observed": observed_hash,
                },
            )
    split_json = json.loads(
        (repository / config["frozen_split_json"]).read_text(encoding="utf-8")
    )
    if split_json.get("status") != "FROZEN":
        raise ExportFailure("source_manifest_not_frozen", split_json.get("status"))


def load_export_cases(repository: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Join the frozen assignments to SOURCE audit paths and enforce firewalls."""
    validate_frozen_split_files(repository, config)
    assignments = pd.read_parquet(repository / config["frozen_split_parquet"])
    allowed_splits = set(config["segmentation_development_splits"])
    if set(assignments["source_split"].unique()) != allowed_splits | {
        "QC_TRAIN",
        "CALIBRATION",
        "ID_TEST",
    }:
        raise ExportFailure(
            "unexpected_frozen_split", sorted(assignments["source_split"].unique())
        )
    selected = assignments.loc[
        assignments["source_split"].isin(allowed_splits),
        ["patient_id", "case_id", "source_split"],
    ].copy()
    expected_selected = sum(
        int(json.loads((repository / config["frozen_split_json"]).read_text())["counts"][name])
        for name in allowed_splits
    )
    if len(selected) != expected_selected:
        raise ExportFailure(
            "segmentation_development_count_mismatch",
            {"expected": expected_selected, "observed": len(selected)},
        )

    audit_columns = [
        "patient_id",
        "case_id",
        "split",
        "m3da_fold",
        "image_member",
        "mask_member",
        "mask_content_audited",
        "observed_raw_mask_labels",
        "observed_mask_labels",
        "foreground_voxel_count",
    ]
    audit = pd.read_parquet(
        repository / config["m3da_audit_manifest"], columns=audit_columns
    )
    source_audit = audit.loc[audit["split"].eq("source")].copy()
    cases = selected.merge(
        source_audit,
        on=["patient_id", "case_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not cases["_merge"].eq("both").all():
        missing = cases.loc[~cases["_merge"].eq("both"), "patient_id"].tolist()
        raise ExportFailure("source_audit_case_missing", missing)
    cases = cases.drop(columns="_merge")
    if not cases["m3da_fold"].eq("source").all():
        raise ExportFailure("non_source_case_selected", None)
    if not cases["mask_content_audited"].eq(True).all():
        raise ExportFailure("selected_source_mask_not_audited", None)
    if cases["patient_id"].duplicated().any():
        raise ExportFailure("duplicate_export_patient", None)
    return cases.sort_values("patient_id", kind="stable", ignore_index=True)


def _safe_source(dataset_root: Path, member: str) -> Path:
    root = dataset_root.resolve()
    source = (root / member).resolve()
    if not source.is_relative_to(root):
        raise ExportFailure("source_path_escapes_dataset_root", member)
    if not source.is_file():
        raise ExportFailure("source_file_missing", member)
    return source


def _map_and_save_label(
    source: Path,
    destination: Path,
    mapping: dict[int, int],
    expected_labels: set[int],
) -> tuple[list[int], list[int], tuple[int, ...]]:
    image = nib.load(source)
    raw = np.asanyarray(image.dataobj)
    if not np.array_equal(raw, np.rint(raw)):
        raise ExportFailure("non_integer_source_label", source.name)
    raw_labels = sorted(int(value) for value in np.unique(raw))
    unexpected_raw = sorted(set(raw_labels) - set(mapping))
    if unexpected_raw:
        raise ExportFailure(
            "unexpected_source_label", {"file": source.name, "labels": unexpected_raw}
        )
    mapped = np.zeros(raw.shape, dtype=np.uint8)
    for raw_label, mapped_label in mapping.items():
        mapped[raw == raw_label] = mapped_label
    mapped_labels = sorted(int(value) for value in np.unique(mapped))
    if set(mapped_labels) - expected_labels:
        raise ExportFailure(
            "unexpected_mapped_label",
            {"file": source.name, "labels": mapped_labels},
        )
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mapped, image.affine, header), destination)
    saved = nib.load(destination)
    saved_labels = sorted(int(value) for value in np.unique(np.asanyarray(saved.dataobj)))
    if saved_labels != mapped_labels:
        raise ExportFailure(
            "saved_label_verification_failed",
            {"expected": mapped_labels, "observed": saved_labels},
        )
    return raw_labels, mapped_labels, tuple(int(value) for value in raw.shape)


def _dataset_json(config: dict[str, Any], number_of_cases: int) -> dict[str, Any]:
    return {
        "channel_names": config["channel_names"],
        "labels": config["labels"],
        "numTraining": number_of_cases,
        "file_ending": config["file_ending"],
        "name": config["dataset_name"],
        "description": "SHIFT-QC M0 M3DA Task05 frozen SEG_DEV export",
        "source_split_manifest": config["frozen_split_json"],
    }


def _split_json(cases: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, list[str]]]:
    return [
        {
            "train": sorted(
                cases.loc[
                    cases["source_split"].eq(config["training_split"]), "patient_id"
                ].tolist()
            ),
            "val": sorted(
                cases.loc[
                    cases["source_split"].eq(config["validation_split"]), "patient_id"
                ].tolist()
            ),
        }
    ]


def export_dataset(
    repository: Path,
    dataset_root: Path,
    nnunet_raw: Path,
    conversion_manifest: Path,
    artifact_directory: Path,
    config: dict[str, Any],
) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    """Create the nnU-Net raw dataset once without exposing non-SEG_DEV cases."""
    cases = load_export_cases(repository, config)
    dataset_name = dataset_directory_name(config)
    destination = nnunet_raw / dataset_name
    staging = nnunet_raw / f".{dataset_name}.partial"
    if destination.exists() or staging.exists():
        raise ExportFailure(
            "nnunet_raw_dataset_already_exists",
            {"destination": destination.name, "partial": staging.exists()},
        )

    images_tr = staging / "imagesTr"
    labels_tr = staging / "labelsTr"
    images_tr.mkdir(parents=True)
    labels_tr.mkdir(parents=True)
    mapping = {int(key): int(value) for key, value in config["m3da_label_mapping"].items()}
    expected_labels = {int(value) for value in config["labels"].values()}
    records: list[dict[str, Any]] = []
    for row in cases.itertuples(index=False):
        image_source = _safe_source(dataset_root, row.image_member)
        mask_source = _safe_source(dataset_root, row.mask_member)
        image_relative = f"imagesTr/{row.patient_id}_0000.nii.gz"
        mask_relative = f"labelsTr/{row.patient_id}.nii.gz"
        image_destination = staging / image_relative
        mask_destination = staging / mask_relative
        shutil.copy2(image_source, image_destination)
        raw_labels, mapped_labels, mask_shape = _map_and_save_label(
            mask_source, mask_destination, mapping, expected_labels
        )
        image = nib.load(image_destination)
        image_shape = tuple(int(value) for value in image.shape)
        if image_shape != mask_shape:
            raise ExportFailure(
                "exported_image_mask_shape_mismatch",
                {"patient_id": row.patient_id, "image": image_shape, "mask": mask_shape},
            )
        records.append(
            {
                "patient_id": row.patient_id,
                "m3da_case_id": row.case_id,
                "source_split": row.source_split,
                "nnunet_case_identifier": row.patient_id,
                "source_image_member": row.image_member,
                "source_mask_member": row.mask_member,
                "exported_image": image_relative,
                "exported_label": mask_relative,
                "image_shape": list(image_shape),
                "raw_labels": raw_labels,
                "mapped_labels": mapped_labels,
                "foreground_voxel_count": int(row.foreground_voxel_count),
                "image_sha256": sha256_file(image_destination),
                "label_sha256": sha256_file(mask_destination),
            }
        )

    dataset_json = _dataset_json(config, len(cases))
    split_json = _split_json(cases, config)
    (staging / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (staging / "splits_final.json").write_text(
        json.dumps(split_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(staging, destination)

    frame = pd.DataFrame.from_records(records).sort_values(
        "patient_id", kind="stable", ignore_index=True
    )
    conversion_manifest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(conversion_manifest, index=False, engine="pyarrow")
    artifact_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(destination / "dataset.json", artifact_directory / "dataset.json")
    shutil.copy2(
        destination / "splits_final.json", artifact_directory / "splits_final.json"
    )
    report = {
        "status": "PASS",
        "dataset": dataset_name,
        "cases_exported": int(len(frame)),
        "seg_train": int(frame["source_split"].eq(config["training_split"]).sum()),
        "seg_val": int(frame["source_split"].eq(config["validation_split"]).sum()),
        "non_segmentation_cases_exported": 0,
        "raw_labels_observed": sorted(
            {int(label) for labels in frame["raw_labels"] for label in labels}
        ),
        "mapped_labels_observed": sorted(
            {int(label) for labels in frame["mapped_labels"] for label in labels}
        ),
        "frozen_split_hashes_verified": True,
    }
    (artifact_directory / "conversion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination, frame, report
