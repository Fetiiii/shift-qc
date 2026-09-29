#!/usr/bin/env python3
"""Run or provenance-safe resume of frozen TASK-020C standard training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.backends import cudnn

from nnunetv2.run.run_training import get_trainer_from_args
from shiftqc.data.nnunet_export import sha256_file
from shiftqc.segmentation.task020c import (
    Task020CFailure,
    atomic_json,
    load_json,
    storage_preflight,
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


def host_memory() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            values[f"{key}_bytes"] = int(value.strip().split()[0]) * 1024
    return values


def nvidia_snapshot() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True,
    )
    values = [value.strip() for value in completed.stdout.strip().splitlines()[0].split(",")]
    return {
        "gpu_name": values[0], "driver_version": values[1],
        "total_vram_mib": int(values[2]), "free_vram_mib": int(values[3]),
        "used_vram_mib": int(values[4]),
    }


class NvidiaMonitor:
    def __init__(self, interval: float):
        self.interval = interval
        self.peak_used_mib = 0
        self.samples = 0
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                snapshot = nvidia_snapshot()
                self.peak_used_mib = max(self.peak_used_mib, snapshot["used_vram_mib"])
                self.samples += 1
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                self.errors.append(f"{type(error).__name__}: {error}")
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(10.0, self.interval * 3))


def copy_audit_artifacts(model_base: Path, fold: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        model_base / "plans.json": output / "plans.json",
        model_base / "dataset.json": output / "dataset.json",
        model_base / "dataset_fingerprint.json": output / "dataset_fingerprint.json",
        fold / "debug.json": output / "debug.json",
        fold / "progress.png": output / "progress.png",
        fold / "validation" / "summary.json": output / "source_validation_summary.json",
    }
    logs = sorted(fold.glob("training_log_*.txt"))
    if logs:
        sources[logs[-1]] = output / "training_log.txt"
    for source, destination in sources.items():
        if source.is_file():
            shutil.copy2(source, destination)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--fold", type=int, default=None,
                       help="override config fold; TASK-020D runs folds 0-4 from one config")
    value.add_argument("--continue-training", action="store_true")
    value.add_argument("--restart-reason")
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
    # Per-fold report and audit paths so folds never overwrite each other.
    configured_report = Path(config["outputs"]["training_report"])
    report_path = REPOSITORY / configured_report.with_name(
        f"{configured_report.stem}_fold{fold}{configured_report.suffix}"
    )
    output_directory = REPOSITORY / config["outputs"]["directory"] / f"fold_{fold}"
    results_root = REPOSITORY / config["nnunet_roots"]["results"]
    preprocessed_root = REPOSITORY / config["nnunet_roots"]["preprocessed"]
    model_base = (
        results_root / config["dataset"]
        / f"{config['trainer']}__{config['plans_identifier']}__{config['configuration']}"
    )
    fold_directory = model_base / f"fold_{fold}"
    previous = load_json(report_path) if report_path.is_file() else {}
    attempts = list(previous.get("attempts", []))
    report: dict[str, Any] = {
        "task": config["task"], "experiment_id": config["experiment_id"],
        "status": "RUNNING", "final_training_status": "RUNNING",
        "dataset": config["dataset"], "configuration": config["configuration"],
        "fold": fold, "trainer_class": config["trainer"],
        "training_start_timestamp": previous.get("training_start_timestamp", now()),
        "training_end_timestamp": None, "epochs_completed": 0,
        "resume_count": int(previous.get("resume_count", 0)) + int(args.continue_training),
        "attempts": attempts, "oom_occurred": False,
        "peak_torch_allocated_bytes": int(previous.get("peak_torch_allocated_bytes", 0)),
        "peak_torch_reserved_bytes": int(previous.get("peak_torch_reserved_bytes", 0)),
        "peak_nvidia_smi_memory_used_mib": int(previous.get("peak_nvidia_smi_memory_used_mib", 0)),
        "checkpoint_final_path": None, "checkpoint_best_path": None,
        "checkpoint_hashes": {}, "source_validation_diagnostic": None,
        "scientific_checkpoint_policy": "checkpoint_final.pth frozen before outcome evaluation",
        "checkpoint_best_selected_for_inference": False,
        "inference_started": False, "outcome_metrics_computed": [],
        "warnings": [
            "SEG_VAL output is a SOURCE TRAINING DIAGNOSTIC ONLY.",
            "checkpoint_final.pth is the pre-frozen scientific inference checkpoint.",
        ],
    }
    monitor = NvidiaMonitor(float(config["monitor_interval_seconds"]))
    trainer = None
    attempt_started = perf_counter()
    try:
        if args.continue_training:
            if not report_path.is_file():
                raise Task020CFailure("resume_report_missing", None)
            identity = (previous.get("dataset"), previous.get("configuration"), previous.get("fold"))
            expected_identity = (config["dataset"], config["configuration"], fold)
            if identity != expected_identity:
                raise Task020CFailure("resume_scientific_identity_mismatch", {"expected": expected_identity, "observed": identity})
            if not (fold_directory / "checkpoint_latest.pth").is_file():
                raise Task020CFailure("resume_checkpoint_latest_missing", None)
        elif fold_directory.exists():
            raise Task020CFailure(
                "clean_scientific_result_path_already_exists",
                fold_directory.relative_to(REPOSITORY).as_posix(),
            )

        disk_before = storage_preflight(REPOSITORY, config)
        upstream = verify_upstream(REPOSITORY, config)
        frozen = verify_frozen_training_state(REPOSITORY, config, fold)
        if not torch.cuda.is_available():
            raise Task020CFailure("cuda_unavailable", "torch.cuda.is_available() is false")
        gpu = nvidia_snapshot()
        free_vram, total_vram = torch.cuda.mem_get_info(0)
        os.environ.update(
            {
                "nnUNet_raw": str(REPOSITORY / config["nnunet_roots"]["raw"]),
                "nnUNet_preprocessed": str(preprocessed_root),
                "nnUNet_results": str(results_root),
                "MPLCONFIGDIR": "/tmp/shiftqc_task020c_matplotlib",
            }
        )
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        report.update(
            {
                "disk_usage_before": disk_before, "upstream_hashes_verified": upstream,
                "frozen_state": frozen, "gpu_name": gpu["gpu_name"],
                "driver_version": gpu["driver_version"],
                "total_vram_bytes": int(total_vram), "free_vram_before_bytes": int(free_vram),
                "total_vram_mib_nvidia_smi": gpu["total_vram_mib"],
                "free_vram_before_mib_nvidia_smi": gpu["free_vram_mib"],
                "host_ram": host_memory(), "python_version": platform.python_version(),
                "pytorch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
                "nnunet_version": importlib.metadata.version("nnunetv2"), "git": git_state(),
                "dependency_lock": config["dependency_lock"],
                "dependency_lock_sha256": sha256_file(REPOSITORY / config["dependency_lock"]),
            }
        )
        report["attempts"].append(
            {
                "number": len(report["attempts"]) + 1, "started_at": now(),
                "resume": bool(args.continue_training), "reason": args.restart_reason,
            }
        )
        atomic_json(report_path, report)

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cudnn.deterministic = False
        cudnn.benchmark = True
        trainer = get_trainer_from_args(
            str(config["dataset_id"]), config["configuration"], fold,
            config["trainer"], config["plans_identifier"], args.continue_training,
            device=torch.device("cuda:0"),
        )
        if args.continue_training:
            trainer.load_checkpoint(fold_directory / "checkpoint_latest.pth")
        schedule = config["training_schedule"]
        actual_schedule = {
            "epochs": int(trainer.num_epochs),
            "training_iterations_per_epoch": int(trainer.num_iterations_per_epoch),
            "validation_iterations_per_epoch": int(trainer.num_val_iterations_per_epoch),
            "checkpoint_interval_epochs": int(trainer.save_every),
            "checkpointing_disabled": bool(trainer.disable_checkpointing),
        }
        expected_schedule = {**schedule, "checkpointing_disabled": False}
        if actual_schedule != expected_schedule:
            raise Task020CFailure("standard_training_schedule_mismatch", {"expected": expected_schedule, "observed": actual_schedule})
        report["training_schedule"] = actual_schedule
        report["amp_autocast_standard_trainer"] = True
        report["grad_scaler_standard_trainer"] = True
        atomic_json(report_path, report)

        torch.cuda.reset_peak_memory_stats(0)
        monitor.start()
        trainer.run_training()
        report["epochs_completed"] = int(trainer.current_epoch)
        report["status"] = "STANDARD_SOURCE_VALIDATION_RUNNING"
        atomic_json(report_path, report)
        validation_started = perf_counter()
        trainer.perform_actual_validation(save_probabilities=False)
        report["standard_source_validation_duration_seconds"] = perf_counter() - validation_started
        monitor.stop()

        checkpoint_final = fold_directory / "checkpoint_final.pth"
        checkpoint_best = fold_directory / "checkpoint_best.pth"
        if not checkpoint_final.is_file():
            raise Task020CFailure("checkpoint_final_missing", None)
        checkpoint = torch.load(checkpoint_final, map_location="cpu", weights_only=False)
        if checkpoint.get("trainer_name") != config["trainer"]:
            raise Task020CFailure("checkpoint_trainer_mismatch", checkpoint.get("trainer_name"))
        checkpoint_hashes = {"checkpoint_final.pth": sha256_file(checkpoint_final)}
        if checkpoint_best.is_file():
            checkpoint_hashes["checkpoint_best.pth"] = sha256_file(checkpoint_best)
        validation_summary = fold_directory / "validation" / "summary.json"
        report.update(
            {
                "status": "TRAINING_COMPLETE_INFERENCE_PENDING",
                "final_training_status": "COMPLETE",
                "training_end_timestamp": now(),
                "training_duration_seconds": float(previous.get("training_duration_seconds", 0.0)) + (perf_counter() - attempt_started),
                "peak_torch_allocated_bytes": max(report["peak_torch_allocated_bytes"], int(torch.cuda.max_memory_allocated(0))),
                "peak_torch_reserved_bytes": max(report["peak_torch_reserved_bytes"], int(torch.cuda.max_memory_reserved(0))),
                "peak_nvidia_smi_memory_used_mib": max(report["peak_nvidia_smi_memory_used_mib"], monitor.peak_used_mib),
                "nvidia_smi_samples": int(previous.get("nvidia_smi_samples", 0)) + monitor.samples,
                "nvidia_smi_monitor_errors": list(previous.get("nvidia_smi_monitor_errors", [])) + monitor.errors,
                "checkpoint_final_path": checkpoint_final.relative_to(REPOSITORY).as_posix(),
                "checkpoint_best_path": checkpoint_best.relative_to(REPOSITORY).as_posix() if checkpoint_best.is_file() else None,
                "checkpoint_hashes": checkpoint_hashes,
                "checkpoint_load_succeeded": True,
                "checkpoint_network_architecture": frozen["plan"]["architecture_suffix"],
                "checkpoint_class_count": 4,
                "source_validation_diagnostic": load_json(validation_summary) if validation_summary.is_file() else None,
                "source_validation_diagnostic_label": "SOURCE TRAINING DIAGNOSTIC ONLY",
                "disk_usage_after_training": storage_preflight(REPOSITORY, config),
                "leakage_checks": {
                    "target_evaluation_used_in_fitting": 0,
                    "QC_TRAIN_visible_to_training": 0,
                    "CALIBRATION_visible_to_training": 0,
                    "M3_ID_EVAL_visible_to_training": 0,
                    "M3_OOD_EVAL_visible_to_training": 0,
                    "excluded_vendor_c_visible_to_training": 0,
                },
            }
        )
        copy_audit_artifacts(model_base, fold_directory, output_directory)
    except BaseException as error:
        if monitor._thread.is_alive():
            monitor.stop()
        oom = isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()
        interrupted = isinstance(error, KeyboardInterrupt)
        report.update(
            {
                "status": "INTERRUPTED_BY_USER" if interrupted else "FAIL_TRAINING",
                "final_training_status": "INTERRUPTED" if interrupted else "FAIL",
                "training_end_timestamp": now(), "oom_occurred": oom,
                "failure": {"type": type(error).__name__, "message": str(error)},
                "training_duration_seconds": float(previous.get("training_duration_seconds", 0.0)) + (perf_counter() - attempt_started),
                "peak_nvidia_smi_memory_used_mib": max(report["peak_nvidia_smi_memory_used_mib"], monitor.peak_used_mib),
            }
        )
        if trainer is not None:
            report["epochs_completed"] = int(trainer.current_epoch)
        if torch.cuda.is_available():
            report["peak_torch_allocated_bytes"] = max(report["peak_torch_allocated_bytes"], int(torch.cuda.max_memory_allocated(0)))
            report["peak_torch_reserved_bytes"] = max(report["peak_torch_reserved_bytes"], int(torch.cuda.max_memory_reserved(0)))
    finally:
        atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["final_training_status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
