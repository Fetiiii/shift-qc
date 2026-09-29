#!/usr/bin/env python3
"""Run frozen checkpoint_final image-only inference and validate TASK-020C outputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from shiftqc.data.nnunet_export import sha256_file
from shiftqc.segmentation.task020c import (
    Task020CFailure,
    atomic_json,
    load_json,
    prepare_inference_inputs,
    storage_preflight,
    validate_predictions,
    verify_frozen_training_state,
    verify_upstream,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs/experiments/exp_0005_task020c_full_2d.json"


def now() -> str:
    return datetime.now().astimezone().isoformat()


def git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPOSITORY, check=True,
        capture_output=True, text=True,
    ).stdout
    return {"commit": commit, "working_tree_dirty": bool(status.strip())}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--fold", type=int, default=None,
                       help="override config fold; resolves the per-fold inference block")
    value.add_argument(
        "--continue-prediction", action="store_true",
        help="Resume only missing cases with the same frozen checkpoint and inputs.",
    )
    value.add_argument(
        "--finalize-existing", action="store_true",
        help="Validate an already complete prediction directory without inference.",
    )
    return value


def main() -> int:
    args = parser().parse_args()
    config = load_json(args.config)
    if config.get("task") != "TASK-020E":
        # TASK-020C is superseded by D-008 (five folds, nnUNetTrainer_250epochs).
        # The module default still points at the superseded config, so a forgotten
        # --config would silently start the wrong 1000-epoch single-fold run.
        raise SystemExit(
            f"refusing to run: {args.config} declares task "
            f"{config.get('task')!r}, not 'TASK-020E'. TASK-020C is superseded by "
            "D-008. Pass --config configs/experiments/exp_0005_task020d_5fold_2d.json"
        )
    fold = int(config["fold"]) if args.fold is None else int(args.fold)
    # D-011/D-013: fold 0 is the primary prediction and needs probabilities for the
    # QC feature family; folds 1-4 feed only the agreement signal, which uses argmax.
    fold_block = config.get("folds", {}).get(str(fold), {})
    if fold_block.get("roles"):
        config["inference"] = {**config["inference"], "roles": list(fold_block["roles"])}
    save_probabilities = bool(fold_block.get("save_probabilities", True))
    for key in ("inference_patients", "inference_cases"):
        if key in fold_block:
            config["expected_counts"][key] = int(fold_block[key])
    for key in ("training_report", "inference_report", "inference_manifest"):
        configured = Path(config["outputs"][key])
        config["outputs"][key] = str(
            configured.with_name(f"{configured.stem}_fold{fold}{configured.suffix}")
        )
    config["nnunet_roots"] = {
        **config["nnunet_roots"],
        "prediction_output": f"{config['nnunet_roots']['prediction_output']}/fold_{fold}",
    }
    training_report_path = REPOSITORY / config["outputs"]["training_report"]
    report_path = REPOSITORY / config["outputs"]["inference_report"]
    manifest_path = REPOSITORY / config["outputs"]["inference_manifest"]
    # Fold 0 keeps the original shared staging path so that its recorded
    # input_manifest_sha256 stays exactly what the frozen run produced. Folds 1-4
    # stage into their own directory because they need a different case set (340
    # rather than 460) and would otherwise collide with fold 0's manifest.
    if fold == 0:
        input_manifest_path = report_path.parent / "m3_inference_inputs.parquet"
        input_root = REPOSITORY / config["nnunet_roots"]["inference_input"]
    else:
        input_manifest_path = report_path.parent / f"m3_inference_inputs_fold{fold}.parquet"
        input_root = REPOSITORY / config["nnunet_roots"]["inference_input"] / f"fold_{fold}"
    prediction_root = REPOSITORY / config["nnunet_roots"]["prediction_output"]
    started = perf_counter()
    report: dict[str, Any] = {
        "task": config["task"], "experiment_id": config["experiment_id"],
        "status": "RUNNING", "started_at": now(), "ended_at": None,
        "dataset": config["dataset"], "configuration": config["configuration"],
        "fold": fold, "checkpoint_name": config["checkpoint_for_inference"],
        "standard_inference_tta_mirroring_enabled": bool(config["inference"]["tta_mirroring"]),
        "save_probabilities": save_probabilities, "gt_access_count": 0,
        "inference_roles": list(config["inference"].get("roles", [])),
        "segmentation_metrics_computed": [], "qc_fitting_performed": False,
        "conformal_calibration_performed": False, "task021_started": False,
    }
    try:
        if not training_report_path.is_file():
            raise Task020CFailure("training_report_missing", None)
        training = load_json(training_report_path)
        if training.get("final_training_status") != "COMPLETE":
            raise Task020CFailure("training_not_complete", training.get("final_training_status"))
        if training.get("checkpoint_best_selected_for_inference") is not False:
            raise Task020CFailure("checkpoint_best_selection_policy_violation", None)
        checkpoint_path = REPOSITORY / training["checkpoint_final_path"]
        checkpoint_sha256 = sha256_file(checkpoint_path)
        if checkpoint_sha256 != training["checkpoint_hashes"]["checkpoint_final.pth"]:
            raise Task020CFailure("checkpoint_final_hash_mismatch", None)
        if prediction_root.exists() and not (args.continue_prediction or args.finalize_existing):
            raise Task020CFailure("prediction_root_already_exists", prediction_root.relative_to(REPOSITORY).as_posix())
        if args.finalize_existing and not prediction_root.is_dir():
            raise Task020CFailure("prediction_root_missing_for_finalization", None)

        dataset_root_value = os.environ.get(config["dataset_root_env"])
        if not dataset_root_value:
            raise Task020CFailure("mnms_dataset_root_environment_missing", config["dataset_root_env"])
        dataset_root = Path(dataset_root_value)
        if not dataset_root.is_dir():
            raise Task020CFailure("mnms_dataset_root_missing", config["dataset_root_env"])
        upstream = verify_upstream(REPOSITORY, config)
        frozen = verify_frozen_training_state(REPOSITORY, config)
        disk_before = storage_preflight(REPOSITORY, config)
        input_manifest = prepare_inference_inputs(
            REPOSITORY, dataset_root, input_root, input_manifest_path, config
        )
        if len(input_manifest) != int(config["expected_counts"]["inference_cases"]):
            raise Task020CFailure("inference_input_case_count_mismatch", len(input_manifest))

        report.update(
            {
                "status": "VALIDATING_EXISTING" if args.finalize_existing else "INFERENCE_RUNNING",
                "frozen_checkpoint_sha256": checkpoint_sha256,
                "upstream_hashes_verified": upstream, "frozen_state": frozen,
                "disk_usage_before": disk_before,
                "input_manifest": input_manifest_path.relative_to(REPOSITORY).as_posix(),
                "input_manifest_sha256": sha256_file(input_manifest_path),
                "input_case_count": len(input_manifest),
                "input_patient_count": int(input_manifest["patient_id"].nunique()),
                "dataset_root_provenance": {
                    "environment_variable": config["dataset_root_env"],
                    "absolute_path_recorded": False,
                    "release": "researcher-accessible official 345-case M&Ms OpenDataset",
                },
                "git": git_state(), "python_version": platform.python_version(),
                "nnunet_version": importlib.metadata.version("nnunetv2"),
            }
        )
        atomic_json(report_path, report)

        if not args.finalize_existing:
            predictor = Path(sys.executable).parent / "nnUNetv2_predict"
            command = [
                str(predictor), "-i", str(input_root), "-o", str(prediction_root),
                "-d", str(config["dataset_id"]), "-p", config["plans_identifier"],
                "-tr", config["trainer"], "-c", config["configuration"],
                "-f", str(fold), "-chk", config["checkpoint_for_inference"],
                "-npp", str(config["inference"]["preprocessing_processes"]),
                "-nps", str(config["inference"]["export_processes"]),
            ]
            if save_probabilities:
                command.append("--save_probabilities")
            # Deliberately omit --disable_tta: standard mirroring remains enabled.
            if args.continue_prediction:
                command.append("--continue_prediction")
            environment = os.environ.copy()
            environment.update(
                {
                    "nnUNet_raw": str(REPOSITORY / config["nnunet_roots"]["raw"]),
                    "nnUNet_preprocessed": str(REPOSITORY / config["nnunet_roots"]["preprocessed"]),
                    "nnUNet_results": str(REPOSITORY / config["nnunet_roots"]["results"]),
                    "MPLCONFIGDIR": "/tmp/shiftqc_task020c_matplotlib",
                }
            )
            Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
            subprocess.run(command, cwd=REPOSITORY, env=environment, check=True)

        prediction_manifest, integrity = validate_predictions(
            REPOSITORY, input_manifest, prediction_root, checkpoint_sha256, save_probabilities
        )
        # Compare against this fold's roles only; the config block lists all four.
        selected_roles = set(config["inference"]["roles"])
        expected_counts = {
            role: counts
            for role, counts in config["expected_counts"]["inference"].items()
            if role in selected_roles
        }
        if integrity["counts"] != expected_counts:
            raise Task020CFailure(
                "prediction_cohort_count_mismatch",
                {"expected": expected_counts, "observed": integrity["counts"]},
            )
        if integrity["rows"] != int(config["expected_counts"]["inference_cases"]):
            raise Task020CFailure("prediction_total_case_count_mismatch", integrity["rows"])
        if integrity["unique_patients"] != int(config["expected_counts"]["inference_patients"]):
            raise Task020CFailure("prediction_total_patient_count_mismatch", integrity["unique_patients"])
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_manifest.to_parquet(manifest_path, index=False, engine="pyarrow")
        report.update(
            {
                "status": "PASS", "ended_at": now(),
                "inference_runtime_seconds": perf_counter() - started,
                "prediction_completeness": integrity,
                "prediction_manifest": manifest_path.relative_to(REPOSITORY).as_posix(),
                "prediction_manifest_sha256": sha256_file(manifest_path),
                "prediction_artifact_hashes": {
                    "hard_prediction_set_sha256": __import__("hashlib").sha256(
                        "\n".join(prediction_manifest["hard_prediction_sha256"]).encode("utf-8")
                    ).hexdigest(),
                    "probability_map_set_sha256": (
                        __import__("hashlib").sha256(
                            "\n".join(prediction_manifest["probability_sha256"]).encode("utf-8")
                        ).hexdigest()
                        if save_probabilities
                        else None
                    ),
                },
                "disk_usage_after": storage_preflight(REPOSITORY, config),
                "leakage_checks": {
                    "target_evaluation_used_in_fitting": 0,
                    "ground_truth_used_during_inference": 0,
                    "excluded_vendor_c_training_used": 0,
                    "SEG_TRAIN_inferred": 0,
                    "SEG_VAL_inferred": 0,
                },
                "checkpoint_best_selected_for_inference": False,
                "scientific_checkpoint": "checkpoint_final.pth",
            }
        )
    except BaseException as error:
        report.update(
            {
                "status": "FAIL_INFERENCE", "ended_at": now(),
                "inference_runtime_seconds": perf_counter() - started,
                "failure": {"type": type(error).__name__, "message": str(error)},
            }
        )
    finally:
        atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
