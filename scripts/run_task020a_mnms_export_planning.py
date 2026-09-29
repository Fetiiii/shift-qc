#!/usr/bin/env python3
"""Export Dataset502, extract its fingerprint, and run standard nnU-Net planning."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import (
    ExperimentPlanner,
)

from shiftqc.data.mnms_audit import dataset_tree_inventory, frozen_experiment_tree_hash
from shiftqc.data.mnms_nnunet import (
    ALLOWED_SOURCE_SPLITS,
    MnmsNnunetFailure,
    build_case_table,
    case_hash,
    conversion_scientific_hash,
    dataset_id_conflicts,
    export_dataset,
    load_config,
    load_segmentation_patients,
    patient_hash,
    raw_spacing_summary,
    summarize_plans,
    verify_export,
    verify_frozen_inputs,
    visibility_audit,
)
from shiftqc.data.nnunet_export import sha256_file


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs/datasets/nnunet_m3_mnms_task020a.json"


def now() -> str:
    return datetime.now().astimezone().isoformat()


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPOSITORY, check=True,
        capture_output=True, text=True,
    ).stdout
    return {"commit": commit, "working_tree_dirty": bool(status.strip())}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_nnunet(command: list[str], environment: dict[str, str]) -> None:
    print("Running:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=REPOSITORY, env=environment, check=False)
    if completed.returncode != 0:
        raise MnmsNnunetFailure(
            "nnunet_command_failed", {"command": command, "returncode": completed.returncode}
        )


def _artifact_paths(config: dict[str, Any]) -> dict[str, Path]:
    directory = REPOSITORY / config["outputs"]["artifact_directory"]
    return {
        "directory": directory,
        "export_report": REPOSITORY / config["outputs"]["export_report"],
        "planning_report": REPOSITORY / config["outputs"]["planning_report"],
        "dataset_json": directory / "dataset.json",
        "splits_final": directory / "splits_final.json",
        "fingerprint": directory / "dataset_fingerprint.json",
        "plans": directory / "nnUNetPlans.json",
    }


def _verify_existing(config: dict[str, Any]) -> dict[str, Any]:
    paths = _artifact_paths(config)
    required = {name: path for name, path in paths.items() if name != "directory"}
    missing = {name: str(path) for name, path in required.items() if not path.is_file()}
    if missing:
        raise MnmsNnunetFailure("frozen_task020a_artifact_missing", missing)
    conversion_path = REPOSITORY / config["outputs"]["conversion_manifest"]
    if not conversion_path.is_file():
        raise MnmsNnunetFailure("conversion_manifest_missing", str(conversion_path))
    conversion = pd.read_parquet(conversion_path)
    raw_dataset = REPOSITORY / config["nnunet_roots"]["raw"] / config["dataset_directory"]
    verify_export(raw_dataset, conversion, config)
    report = json.loads(paths["planning_report"].read_text(encoding="utf-8"))
    observed = {
        "dataset_json": sha256_file(raw_dataset / "dataset.json"),
        "splits_final": sha256_file(raw_dataset / "splits_final.json"),
        "fingerprint": sha256_file(
            REPOSITORY / config["nnunet_roots"]["preprocessed"]
            / config["dataset_directory"] / "dataset_fingerprint.json"
        ),
        "plans": sha256_file(
            REPOSITORY / config["nnunet_roots"]["preprocessed"]
            / config["dataset_directory"] / "nnUNetPlans.json"
        ),
    }
    if report["artifact_hashes"] != observed:
        raise MnmsNnunetFailure(
            "frozen_task020a_artifact_hash_mismatch",
            {"expected": report["artifact_hashes"], "observed": observed},
        )
    return {"status": "PASS", "mode": "verify-existing", "artifact_hashes": observed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started_at = now()
    try:
        config = load_config(args.config)
        verify_frozen_inputs(REPOSITORY, config)
        conflicts = dataset_id_conflicts(REPOSITORY, config)
        if conflicts:
            raise MnmsNnunetFailure("dataset_id_502_occupied_by_unrelated_dataset", conflicts)
        if args.verify_existing:
            print(json.dumps(_verify_existing(config), indent=2, sort_keys=True))
            return 0
        dataset_root = args.dataset_root
        if dataset_root is None:
            configured_root = os.environ.get(config["dataset_root_env"])
            dataset_root = Path(configured_root) if configured_root else None
        if dataset_root is None or not dataset_root.is_dir():
            raise MnmsNnunetFailure(
                "mnms_dataset_root_unavailable", config["dataset_root_env"]
            )
        paths = _artifact_paths(config)
        conversion_path = REPOSITORY / config["outputs"]["conversion_manifest"]
        task_outputs = [
            conversion_path,
            paths["export_report"],
            paths["planning_report"],
            paths["dataset_json"],
            paths["splits_final"],
            paths["fingerprint"],
            paths["plans"],
        ]
        existing_outputs = [str(path) for path in task_outputs if path.exists()]
        raw_dataset = REPOSITORY / config["nnunet_roots"]["raw"] / config["dataset_directory"]
        preprocessed_dataset = (
            REPOSITORY / config["nnunet_roots"]["preprocessed"] / config["dataset_directory"]
        )
        results_dataset = REPOSITORY / config["nnunet_roots"]["results"] / config["dataset_directory"]
        if existing_outputs or raw_dataset.exists() or preprocessed_dataset.exists() or results_dataset.exists():
            raise MnmsNnunetFailure(
                "task020a_destination_already_exists",
                {
                    "artifacts": existing_outputs,
                    "raw": raw_dataset.exists(),
                    "preprocessed": preprocessed_dataset.exists(),
                    "results": results_dataset.exists(),
                },
            )

        dataset_before = dataset_tree_inventory(dataset_root)
        frozen_before = frozen_experiment_tree_hash(REPOSITORY)
        patients = load_segmentation_patients(REPOSITORY, config)
        expected_cases = build_case_table(patients, config)

        def progress(index: int, total: int, case_id: str) -> None:
            if index == 1 or index % 20 == 0 or index == total:
                print(f"TASK-020A export {index}/{total} case={case_id}", flush=True)

        raw_dataset, conversion = export_dataset(
            REPOSITORY,
            dataset_root,
            REPOSITORY / config["nnunet_roots"]["raw"],
            conversion_path,
            config,
            progress=progress,
        )
        dataset_after_export = dataset_tree_inventory(dataset_root)
        if dataset_after_export != dataset_before:
            raise MnmsNnunetFailure(
                "source_dataset_changed_during_export",
                {"before": dataset_before, "after": dataset_after_export},
            )
        visibility = visibility_audit(REPOSITORY, conversion, config)
        paths["directory"].mkdir(parents=True, exist_ok=False)
        shutil.copy2(raw_dataset / "dataset.json", paths["dataset_json"])
        shutil.copy2(raw_dataset / "splits_final.json", paths["splits_final"])
        export_report = {
            "task": config["task"],
            "experiment_id": config["experiment_id"],
            "status": "PASS",
            "dataset": config["dataset_directory"],
            "dataset_id": int(config["dataset_id"]),
            "dataset_root_provenance": {
                "configuration": config["dataset_root_env"],
                "absolute_path_recorded": False,
                "source_paths_in_manifest_are_relative": True,
            },
            "source_split_manifest": {
                "path": config["frozen_split"]["parquet"],
                "sha256": config["frozen_split"]["parquet_sha256"],
            },
            "case_naming": config["phase_policy"]["case_naming"],
            "patient_counts": {
                "total": int(conversion["patient_id"].nunique()),
                "SEG_TRAIN": int(
                    conversion.loc[conversion["source_split"].eq("SEG_TRAIN"), "patient_id"].nunique()
                ),
                "SEG_VAL": int(
                    conversion.loc[conversion["source_split"].eq("SEG_VAL"), "patient_id"].nunique()
                ),
            },
            "case_counts": {
                "total": len(conversion),
                "train": int(conversion["source_split"].eq("SEG_TRAIN").sum()),
                "validation": int(conversion["source_split"].eq("SEG_VAL").sum()),
            },
            "vendor_counts": {
                split: {
                    vendor: int(
                        conversion.loc[
                            conversion["source_split"].eq(split) & conversion["vendor"].eq(vendor),
                            "patient_id",
                        ].nunique()
                    )
                    for vendor in ("A", "B")
                }
                for split in ALLOWED_SOURCE_SPLITS
            },
            "patient_id_hash": patient_hash(conversion["patient_id"].unique()),
            "case_id_hash": case_hash(conversion["nnunet_case_identifier"]),
            "train_patient_hash": patient_hash(
                conversion.loc[conversion["source_split"].eq("SEG_TRAIN"), "patient_id"].unique()
            ),
            "val_patient_hash": patient_hash(
                conversion.loc[conversion["source_split"].eq("SEG_VAL"), "patient_id"].unique()
            ),
            "train_case_hash": case_hash(
                conversion.loc[
                    conversion["source_split"].eq("SEG_TRAIN"), "nnunet_case_identifier"
                ]
            ),
            "val_case_hash": case_hash(
                conversion.loc[
                    conversion["source_split"].eq("SEG_VAL"), "nnunet_case_identifier"
                ]
            ),
            "conversion_scientific_content_sha256": conversion_scientific_hash(conversion),
            "label_mapping": config["labels"],
            "label_storage_validation": {
                "integer_tolerance": float(config["label_integer_tolerance"]),
                "operation": "validate within tolerance then store canonical uint8 labels",
                "semantic_remapping_performed": False,
            },
            "observed_label_union": sorted(
                {label for labels in conversion["observed_labels"] for label in labels}
            ),
            "raw_spacing": raw_spacing_summary(conversion),
            "leakage_audit": visibility,
            "source_dataset_immutability": {
                "before": dataset_before,
                "after": dataset_after_export,
                "unchanged": True,
            },
            "deterministic_case_table_verified": expected_cases[
                "nnunet_case_identifier"
            ].tolist()
            == conversion.sort_values(
                ["patient_id", "phase"], kind="stable"
            )["nnunet_case_identifier"].tolist(),
            "dataset_json_sha256": sha256_file(raw_dataset / "dataset.json"),
            "splits_final_sha256": sha256_file(raw_dataset / "splits_final.json"),
            "conversion_manifest_sha256": sha256_file(conversion_path),
            "preprocessing_performed": False,
            "training_started": False,
            "inference_started": False,
            "outcome_evaluation_performed": False,
        }
        _write_json(paths["export_report"], export_report)

        environment = dict(os.environ)
        environment.update(
            {
                "nnUNet_raw": str(REPOSITORY / config["nnunet_roots"]["raw"]),
                "nnUNet_preprocessed": str(
                    REPOSITORY / config["nnunet_roots"]["preprocessed"]
                ),
                "nnUNet_results": str(REPOSITORY / config["nnunet_roots"]["results"]),
            }
        )
        _run_nnunet(
            [
                str(REPOSITORY / ".venv/bin/nnUNetv2_extract_fingerprint"),
                "-d",
                str(config["dataset_id"]),
                "-fpe",
                config["fingerprint"]["extractor"],
                "-np",
                str(config["fingerprint"]["processes"]),
                "--verify_dataset_integrity",
            ],
            environment,
        )
        _run_nnunet(
            [
                str(REPOSITORY / ".venv/bin/nnUNetv2_plan_experiment"),
                "-d",
                str(config["dataset_id"]),
                "-pl",
                config["planner"]["class"],
            ],
            environment,
        )
        fingerprint_path = preprocessed_dataset / "dataset_fingerprint.json"
        plans_path = preprocessed_dataset / "nnUNetPlans.json"
        if not fingerprint_path.is_file() or not plans_path.is_file():
            raise MnmsNnunetFailure(
                "nnunet_planning_artifact_missing",
                {"fingerprint": fingerprint_path.is_file(), "plans": plans_path.is_file()},
            )
        configuration_directories = sorted(
            path.name
            for path in preprocessed_dataset.iterdir()
            if path.is_dir() and path.name.startswith("nnUNetPlans_")
        )
        if configuration_directories:
            raise MnmsNnunetFailure(
                "unexpected_preprocessing_output_present", configuration_directories
            )
        shutil.copy2(raw_dataset / "splits_final.json", preprocessed_dataset / "splits_final.json")
        shutil.copy2(fingerprint_path, paths["fingerprint"])
        shutil.copy2(plans_path, paths["plans"])
        plans = json.loads(plans_path.read_text(encoding="utf-8"))
        fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        planner_vram_target = float(
            inspect.signature(ExperimentPlanner).parameters["gpu_memory_target_in_gb"].default
        )
        artifact_hashes = {
            "dataset_json": sha256_file(raw_dataset / "dataset.json"),
            "splits_final": sha256_file(raw_dataset / "splits_final.json"),
            "fingerprint": sha256_file(fingerprint_path),
            "plans": sha256_file(plans_path),
        }
        dataset_after = dataset_tree_inventory(dataset_root)
        frozen_after = frozen_experiment_tree_hash(REPOSITORY)
        if dataset_after != dataset_before:
            raise MnmsNnunetFailure("source_dataset_changed_during_planning", None)
        if frozen_after != frozen_before:
            raise MnmsNnunetFailure(
                "frozen_experiment_tree_changed", {"before": frozen_before, "after": frozen_after}
            )
        planning_report = {
            "task": config["task"],
            "experiment_id": config["experiment_id"],
            "status": "PASS",
            "started_at": started_at,
            "ended_at": now(),
            "dataset": config["dataset_directory"],
            "dataset_id": int(config["dataset_id"]),
            "source_split_manifest_sha256": config["frozen_split"]["parquet_sha256"],
            "input_population": {
                "patients": int(conversion["patient_id"].nunique()),
                "cases": len(conversion),
                "case_id_hash": case_hash(conversion["nnunet_case_identifier"]),
                "patient_id_hash": patient_hash(conversion["patient_id"].unique()),
                "fingerprint_source_only": True,
                "target_evaluation_cases": 0,
            },
            "raw_spacing": raw_spacing_summary(conversion),
            "fingerprint": {
                "extractor": config["fingerprint"]["extractor"],
                "dataset_integrity_verified": True,
                "spacings_count": len(fingerprint["spacings"]),
                "shapes_after_crop_count": len(fingerprint["shapes_after_crop"]),
                "median_relative_size_after_cropping": fingerprint.get(
                    "median_relative_size_after_cropping"
                ),
                "foreground_intensity_properties_per_channel": fingerprint.get(
                    "foreground_intensity_properties_per_channel"
                ),
            },
            "planner": {
                "class": plans.get("experiment_planner_used"),
                "overrides_used": False,
                "original_median_spacing_after_transpose": plans.get(
                    "original_median_spacing_after_transp"
                ),
                "original_median_shape_after_transpose": plans.get(
                    "original_median_shape_after_transp"
                ),
                "available_configurations": sorted(plans["configurations"]),
                "configurations": summarize_plans(plans, planner_vram_target),
                "configuration_scientifically_selected": None,
            },
            "artifact_hashes": artifact_hashes,
            "artifacts": {
                "raw_dataset": f"{config['nnunet_roots']['raw']}/{config['dataset_directory']}",
                "preprocessed_planning_directory": (
                    f"{config['nnunet_roots']['preprocessed']}/{config['dataset_directory']}"
                ),
                "conversion_manifest": config["outputs"]["conversion_manifest"],
            },
            "environment": {
                "python": platform.python_version(),
                "nnunetv2": importlib.metadata.version("nnunetv2"),
                "numpy": importlib.metadata.version("numpy"),
                "pandas": importlib.metadata.version("pandas"),
                "nibabel": importlib.metadata.version("nibabel"),
            },
            "git": _git_state(),
            "integrity": {
                "source_dataset_before": dataset_before,
                "source_dataset_after": dataset_after,
                "source_dataset_unchanged": True,
                "frozen_exp0002_0003_0004_before": frozen_before,
                "frozen_exp0002_0003_0004_after": frozen_after,
                "frozen_experiments_unchanged": True,
                "leakage_audit": visibility,
                "preprocessing_configuration_directories": configuration_directories,
                "full_preprocessing_performed": False,
                "training_started": False,
                "inference_started": False,
                "outcome_evaluation_performed": False,
                "configuration_selected": False,
            },
        }
        _write_json(paths["planning_report"], planning_report)
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "dataset": config["dataset_directory"],
                    "configurations": planning_report["planner"]["available_configurations"],
                    "artifact_hashes": artifact_hashes,
                    "export_report_sha256": sha256_file(paths["export_report"]),
                    "planning_report_sha256": sha256_file(paths["planning_report"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except MnmsNnunetFailure as error:
        print(
            json.dumps(
                {"status": "FAIL", "failure": {"code": error.code, "details": error.details}},
                indent=2,
                default=str,
            ),
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {"status": "FAIL", "failure": {"code": type(error).__name__, "details": str(error)}},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
