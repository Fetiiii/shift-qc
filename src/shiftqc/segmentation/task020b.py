"""Frozen configuration and artifact integrity helpers for TASK-020B."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from shiftqc.data.mnms_audit import canonical_hash, frozen_experiment_tree_hash
from shiftqc.data.mnms_nnunet import visibility_audit
from shiftqc.data.nnunet_export import sha256_file


class Task020BFailure(RuntimeError):
    def __init__(self, code: str, details: Any = None):
        self.code = code
        self.details = details
        super().__init__(f"{code}: {details}")


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def task020a_artifact_paths(repository: Path, config: dict[str, Any]) -> dict[str, Path]:
    root = repository / "outputs/EXP-0005/TASK-020A"
    return {name: root / name for name in config["upstream"]["task020a_artifacts"]}


def verify_upstream(repository: Path, config: dict[str, Any]) -> dict[str, Any]:
    checks = {
        config["upstream"]["m3_split_parquet"]["path"]: config["upstream"][
            "m3_split_parquet"
        ]["sha256"],
        config["upstream"]["conversion_manifest"]["path"]: config["upstream"][
            "conversion_manifest"
        ]["sha256"],
    }
    checks.update(
        {
            str(path.relative_to(repository)): config["upstream"]["task020a_artifacts"][name]
            for name, path in task020a_artifact_paths(repository, config).items()
        }
    )
    observed: dict[str, str] = {}
    for relative, expected in checks.items():
        path = repository / relative
        if not path.is_file():
            raise Task020BFailure("frozen_upstream_missing", relative)
        digest = sha256_file(path)
        if digest != expected:
            raise Task020BFailure(
                "frozen_upstream_hash_mismatch",
                {"path": relative, "expected": expected, "observed": digest},
            )
        observed[relative] = digest
    return observed


def verify_selected_plan(repository: Path, config: dict[str, Any]) -> dict[str, Any]:
    plans_path = repository / "outputs/EXP-0005/TASK-020A/nnUNetPlans.json"
    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    selected = config["configuration"]
    if selected != "2d":
        raise Task020BFailure("selected_configuration_not_frozen_2d", selected)
    planned = plans["configurations"].get(selected)
    if planned is None:
        raise Task020BFailure("selected_configuration_missing_from_plans", selected)
    frozen = config["frozen_plan"]
    architecture = planned["architecture"]
    kwargs = architecture["arch_kwargs"]
    observed = {
        "target_spacing": planned["spacing"],
        "median_resampled_shape": planned["median_image_size_in_voxels"],
        "patch_size": planned["patch_size"],
        "batch_size": int(planned["batch_size"]),
        "architecture_suffix": architecture["network_class_name"].split(".")[-1],
        "stages": int(kwargs["n_stages"]),
        "features": kwargs["features_per_stage"],
        "network_normalization": kwargs["norm_op"],
        "intensity_normalization": planned["normalization_schemes"],
    }
    expected = {
        "target_spacing": frozen["target_spacing"],
        "median_resampled_shape": frozen["median_resampled_shape"],
        "patch_size": frozen["patch_size"],
        "batch_size": int(frozen["batch_size"]),
        "architecture_suffix": frozen["architecture_suffix"],
        "stages": int(frozen["stages"]),
        "features": frozen["features"],
        "network_normalization": frozen["network_normalization"],
        "intensity_normalization": frozen["intensity_normalization"],
    }
    if observed != expected:
        raise Task020BFailure(
            "frozen_2d_plan_mismatch", {"expected": expected, "observed": observed}
        )
    return observed


def filesystem_preflight(repository: Path, config: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "nnunet_raw": repository / config["nnunet_roots"]["raw"],
        "nnunet_preprocessed": repository / config["nnunet_roots"]["preprocessed"],
        "nnunet_results": repository / config["nnunet_roots"]["results"],
        "exp0005_outputs": repository / "outputs/EXP-0005",
    }
    minimum = int(config["preprocessing"]["minimum_free_bytes"])
    report: dict[str, Any] = {}
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(path)
        report[name] = {
            "path_provenance": path.relative_to(repository).as_posix(),
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
            "minimum_required_free_bytes": minimum,
            "sufficient": int(usage.free) >= minimum,
        }
    if not all(item["sufficient"] for item in report.values()):
        raise Task020BFailure("insufficient_storage", report)
    return report


def tree_hash(root: Path) -> dict[str, Any]:
    records = [
        (path.relative_to(root).as_posix(), sha256_file(path))
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return {"file_count": len(records), "sha256": canonical_hash(records)}


def preprocessed_case_sets(configuration_directory: Path) -> dict[str, set[str]]:
    data = {
        path.name.removesuffix(".b2nd")
        for path in configuration_directory.glob("*.b2nd")
        if not path.name.endswith("_seg.b2nd")
    }
    segmentations = {
        path.name.removesuffix("_seg.b2nd")
        for path in configuration_directory.glob("*_seg.b2nd")
    }
    properties = {
        path.name.removesuffix(".pkl")
        for path in configuration_directory.glob("*.pkl")
    }
    return {"data": data, "segmentations": segmentations, "properties": properties}


def verify_preprocessed_dataset(
    repository: Path, config: dict[str, Any]
) -> dict[str, Any]:
    dataset = config["dataset"]
    base = repository / config["nnunet_roots"]["preprocessed"] / dataset
    selected = base / f"{config['plans_identifier']}_{config['configuration']}"
    forbidden_3d = base / f"{config['plans_identifier']}_3d_fullres"
    if not selected.is_dir():
        raise Task020BFailure("selected_preprocessing_directory_missing", str(selected))
    if forbidden_3d.exists():
        raise Task020BFailure("forbidden_3d_fullres_preprocessing_present", str(forbidden_3d))
    conversion = pd.read_parquet(
        repository / config["upstream"]["conversion_manifest"]["path"]
    )
    expected_cases = set(conversion["nnunet_case_identifier"])
    observed = preprocessed_case_sets(selected)
    mismatches = {
        name: {
            "missing": sorted(expected_cases - values),
            "unexpected": sorted(values - expected_cases),
        }
        for name, values in observed.items()
        if values != expected_cases
    }
    if mismatches:
        raise Task020BFailure("preprocessed_case_set_mismatch", mismatches)
    split_path = base / "splits_final.json"
    if sha256_file(split_path) != config["upstream"]["task020a_artifacts"][
        "splits_final.json"
    ]:
        raise Task020BFailure("frozen_custom_split_changed", None)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if len(split) != 1:
        raise Task020BFailure("unexpected_fold_count", len(split))
    train_cases = set(split[0]["train"])
    validation_cases = set(split[0]["val"])
    train_patients = {case.rsplit("__", 1)[0] for case in train_cases}
    validation_patients = {case.rsplit("__", 1)[0] for case in validation_cases}
    expected = config["expected_counts"]
    counts = {
        "preprocessed_cases": len(expected_cases),
        "train_cases": len(train_cases),
        "validation_cases": len(validation_cases),
        "train_patients": len(train_patients),
        "validation_patients": len(validation_patients),
        "train_validation_case_overlap": len(train_cases & validation_cases),
        "train_validation_patient_overlap": len(train_patients & validation_patients),
    }
    expected_counts = {
        "preprocessed_cases": int(expected["cases"]),
        "train_cases": int(expected["train_cases"]),
        "validation_cases": int(expected["validation_cases"]),
        "train_patients": int(expected["train_patients"]),
        "validation_patients": int(expected["validation_patients"]),
        "train_validation_case_overlap": 0,
        "train_validation_patient_overlap": 0,
    }
    if counts != expected_counts:
        raise Task020BFailure(
            "preprocessed_split_count_mismatch",
            {"expected": expected_counts, "observed": counts},
        )
    patient_phase_counts = (
        conversion.groupby("patient_id", observed=True)["phase"].agg(lambda values: set(values))
    )
    if not patient_phase_counts.eq({"ED", "ES"}).all():
        raise Task020BFailure("preprocessed_patient_phase_grouping_failure", None)
    leakage = visibility_audit(repository, conversion, _task020a_compat_config(config))
    return {
        **counts,
        "configuration_directory": selected.relative_to(repository).as_posix(),
        "configuration_output_size_bytes": sum(
            path.stat().st_size for path in selected.rglob("*") if path.is_file()
        ),
        "configuration_tree": tree_hash(selected),
        "leakage": leakage,
        "phase_grouping_preserved": True,
        "plans_identifier": config["plans_identifier"],
        "configuration": config["configuration"],
    }


def _task020a_compat_config(config: dict[str, Any]) -> dict[str, Any]:
    return {"frozen_split": {"parquet": config["upstream"]["m3_split_parquet"]["path"]}}


def frozen_state(repository: Path, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "exp0002_0003_0004": frozen_experiment_tree_hash(repository),
        "task020a": {
            name: sha256_file(path)
            for name, path in task020a_artifact_paths(repository, config).items()
        },
        "task019_split": sha256_file(
            repository / config["upstream"]["m3_split_parquet"]["path"]
        ),
        "task020a_conversion": sha256_file(
            repository / config["upstream"]["conversion_manifest"]["path"]
        ),
    }
