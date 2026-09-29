#!/usr/bin/env python3
"""Run frozen 2D-only nnU-Net preprocessing for TASK-020B."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from shiftqc.segmentation.task020b import (
    Task020BFailure,
    filesystem_preflight,
    frozen_state,
    load_config,
    verify_preprocessed_dataset,
    verify_selected_plan,
    verify_upstream,
)
from shiftqc.data.nnunet_export import sha256_file


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs/experiments/exp_0005_task020b_2d_smoke.json"


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-existing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started_at = now()
    try:
        config = load_config(args.config)
        upstream = verify_upstream(REPOSITORY, config)
        plan = verify_selected_plan(REPOSITORY, config)
        disk_before = filesystem_preflight(REPOSITORY, config)
        report_path = REPOSITORY / config["outputs"]["preprocessing_report"]
        preprocessed_base = (
            REPOSITORY / config["nnunet_roots"]["preprocessed"] / config["dataset"]
        )
        selected_directory = preprocessed_base / f"{config['plans_identifier']}_{config['configuration']}"
        if args.verify_existing:
            if not report_path.is_file():
                raise Task020BFailure("preprocessing_report_missing", str(report_path))
            verification = verify_preprocessed_dataset(REPOSITORY, config)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["preprocessed"]["configuration_tree"] != verification["configuration_tree"]:
                raise Task020BFailure("preprocessing_tree_hash_changed", None)
            print(json.dumps({"status": "PASS", "mode": "verify-existing"}, indent=2))
            return 0
        if report_path.exists() or selected_directory.exists():
            raise Task020BFailure(
                "task020b_preprocessing_already_exists",
                {"report": report_path.exists(), "directory": selected_directory.exists()},
            )
        forbidden_3d = preprocessed_base / f"{config['plans_identifier']}_3d_fullres"
        if forbidden_3d.exists():
            raise Task020BFailure("forbidden_3d_preprocessing_preexists", str(forbidden_3d))
        frozen_before = frozen_state(REPOSITORY, config)
        environment = dict(__import__("os").environ)
        environment.update(
            {
                "nnUNet_raw": str(REPOSITORY / config["nnunet_roots"]["raw"]),
                "nnUNet_preprocessed": str(REPOSITORY / config["nnunet_roots"]["preprocessed"]),
                "nnUNet_results": str(REPOSITORY / config["nnunet_roots"]["results"]),
            }
        )
        command = [
            str(REPOSITORY / ".venv/bin/nnUNetv2_preprocess"),
            "-d",
            str(config["dataset_id"]),
            "-c",
            config["configuration"],
            "-plans_name",
            config["plans_identifier"],
            "-np",
            str(config["preprocessing"]["processes"]),
        ]
        completed = subprocess.run(command, cwd=REPOSITORY, env=environment, check=False)
        if completed.returncode != 0:
            raise Task020BFailure(
                "nnunet_2d_preprocessing_failed", {"returncode": completed.returncode}
            )
        verification = verify_preprocessed_dataset(REPOSITORY, config)
        frozen_after = frozen_state(REPOSITORY, config)
        if frozen_before != frozen_after:
            raise Task020BFailure(
                "frozen_upstream_changed_during_preprocessing",
                {"before": frozen_before, "after": frozen_after},
            )
        disk_after = filesystem_preflight(REPOSITORY, config)
        report = {
            "task": config["task"],
            "experiment_id": config["experiment_id"],
            "status": "PASS",
            "started_at": started_at,
            "ended_at": now(),
            "dataset": config["dataset"],
            "dataset_id": int(config["dataset_id"]),
            "selected_configuration": config["configuration"],
            "selection_provenance": config["selection_provenance"],
            "frozen_plan": plan,
            "preprocessing_command": (
                f"nnUNetv2_preprocess -d {config['dataset_id']} -c 2d "
                f"-plans_name {config['plans_identifier']} -np {config['preprocessing']['processes']}"
            ),
            "preprocessed": verification,
            "disk_preflight": {
                "status": "PASS",
                "before": disk_before,
                "after": disk_after,
                "minimum_free_space_rationale": config["preprocessing"][
                    "minimum_free_space_rationale"
                ],
            },
            "upstream_hashes_verified": upstream,
            "frozen_state_before": frozen_before,
            "frozen_state_after": frozen_after,
            "environment": {
                "python": platform.python_version(),
                "nnunetv2": importlib.metadata.version("nnunetv2"),
                "numpy": importlib.metadata.version("numpy"),
                "blosc2": importlib.metadata.version("blosc2"),
            },
            "git": git_state(),
            "integrity": {
                "only_2d_preprocessed": True,
                "3d_fullres_preprocessed": False,
                "new_plans_created": False,
                "full_training_started": False,
                "inference_started": False,
                "outcome_evaluation_performed": False,
            },
        }
        write_json(report_path, report)
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "preprocessed_cases": verification["preprocessed_cases"],
                    "output_size_bytes": verification["configuration_output_size_bytes"],
                    "tree_sha256": verification["configuration_tree"]["sha256"],
                    "report_sha256": sha256_file(report_path),
                },
                indent=2,
            )
        )
        return 0
    except Task020BFailure as error:
        print(
            json.dumps(
                {"status": "FAIL", "failure": {"code": error.code, "details": error.details}},
                indent=2,
                default=str,
            ),
            file=sys.stderr,
        )
        return 2 if error.code == "insufficient_storage" else 1


if __name__ == "__main__":
    raise SystemExit(main())
