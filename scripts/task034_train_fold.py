#!/usr/bin/env python3
"""Train one frozen TASK-034 nnU-Net fold without computing full-val metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from nnunetv2.run.run_training import get_trainer_from_args


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/VALIDATION2_ABDOMENCT/SEGMENTATION"
DATASET = "Dataset503_SHIFTQC_Validation2_AbdomenCT"
TRAINER = "nnUNetTrainer_250epochs"
MODEL = ROOT / "data/nnunet_results" / DATASET / f"{TRAINER}__nnUNetPlans__2d"
PREPROCESSED = ROOT / "data/nnunet_preprocessed" / DATASET
FROZEN_SPLIT = ROOT / "outputs/VALIDATION2_ABDOMENCT/FREEZE/01_validation2_split_manifest.csv"
SEED = 2026


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    args = parser.parse_args()
    fold = args.fold
    if os.environ.get("PYTHONHASHSEED") != str(SEED):
        raise SystemExit("PYTHONHASHSEED=2026 must be set before interpreter start")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise SystemExit("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; CPU fallback is forbidden")
    fold_dir = MODEL / f"fold_{fold}"
    report_path = OUTPUT / f"training/fold_{fold}/training_report.json"
    if fold_dir.exists() or report_path.exists():
        raise SystemExit("fold result already exists; unauthorized overwrite/rerun refused")

    with FROZEN_SPLIT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seg = [row for row in rows if row["role"] == "SEG_DEV"]
    val = [row for row in seg if int(row["segmentation_fold_if_applicable"]) == fold]
    train = [row for row in seg if int(row["segmentation_fold_if_applicable"]) != fold]
    if len(train) != 144 or len(val) != 36:
        raise SystemExit("frozen fold size mismatch")
    if Counter(row["source_dataset"] for row in val) != Counter({"MSD Pancreas": 28, "NIH Pancreas-CT": 8}):
        raise SystemExit("frozen validation provenance mismatch")

    os.environ.update(
        {
            "nnUNet_raw": str(ROOT / "data/nnunet_raw"),
            "nnUNet_preprocessed": str(ROOT / "data/nnunet_preprocessed"),
            "nnUNet_results": str(ROOT / "data/nnunet_results"),
            "MPLCONFIGDIR": "/tmp/shiftqc_task034_matplotlib",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # Strict global deterministic-algorithm enforcement was removed by the
    # result-blind determinism compatibility amendment. The frozen standard 2D
    # nnU-Net cross-entropy path dispatches to the CUDA nll_loss2d kernel, which
    # the installed PyTorch build does not expose a deterministic implementation
    # for, so the enforcement blocked the stock trainer at its first training
    # step. nnU-Net does not itself require it. Seeds, cuDNN determinism, the
    # disabled cuDNN autotuner and the fixed cuBLAS workspace are retained,
    # because none of them alters or blocks the stock training algorithm.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    report = {
        "task_id": "TASK-034", "fold": fold, "status": "RUNNING",
        "start_timestamp": now(), "end_timestamp": None,
        "configuration": "2d", "trainer": TRAINER, "plans": "nnUNetPlans",
        "seed": SEED, "deterministic_algorithms": False,
        "strict_deterministic_enforcement": False,
        "bitwise_reproducibility_claimed": False,
        "reproducibility_note": (
            "GPU training is seeded and environment-pinned but is not claimed to be "
            "bitwise reproducible across independent executions."
        ),
        "cudnn_deterministic": True, "cudnn_benchmark": False,
        "train_patient_count": len(train), "validation_patient_count": len(val),
        "train_source_counts": dict(Counter(row["source_dataset"] for row in train)),
        "validation_source_counts": dict(Counter(row["source_dataset"] for row in val)),
        "validation_patient_ids": sorted(row["source_patient_id"] for row in val),
        "scientific_metrics_computed": [], "full_validation_executed": False,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    atomic_json(report_path, report)
    started = perf_counter()
    try:
        trainer = get_trainer_from_args(
            "503", "2d", fold, TRAINER, "nnUNetPlans", False,
            device=torch.device("cuda:0"),
        )
        schedule = {
            "epochs": int(trainer.num_epochs),
            "training_iterations_per_epoch": int(trainer.num_iterations_per_epoch),
            "validation_iterations_per_epoch": int(trainer.num_val_iterations_per_epoch),
            "checkpoint_interval_epochs": int(trainer.save_every),
        }
        expected = {
            "epochs": 250, "training_iterations_per_epoch": 250,
            "validation_iterations_per_epoch": 50, "checkpoint_interval_epochs": 50,
        }
        if schedule != expected:
            raise RuntimeError(f"standard nnU-Net schedule mismatch: {schedule} != {expected}")
        report["schedule"] = schedule
        atomic_json(report_path, report)
        trainer.run_training()
        checkpoint = fold_dir / "checkpoint_final.pth"
        logs = sorted(fold_dir.glob("training_log_*.txt"))
        if not checkpoint.is_file() or not logs:
            raise RuntimeError("training completed without checkpoint_final or training log")
        report.update(
            {
                "status": "PASS", "exit_status": 0, "end_timestamp": now(),
                "duration_seconds": perf_counter() - started,
                "checkpoint_path": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256(checkpoint),
                "training_log_path": str(logs[-1].relative_to(ROOT)),
                "training_log_sha256": sha256(logs[-1]),
                "epochs_completed": int(trainer.current_epoch),
            }
        )
        atomic_json(report_path, report)
    except BaseException as error:
        report.update(
            {
                "status": "FAIL", "exit_status": 1, "end_timestamp": now(),
                "duration_seconds": perf_counter() - started,
                "error_type": type(error).__name__, "error": str(error),
            }
        )
        atomic_json(report_path, report)
        raise


if __name__ == "__main__":
    main()
