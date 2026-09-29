#!/usr/bin/env python3
"""Run checkpoint-final inference only on each frozen SEG_DEV validation fold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/VALIDATION2_ABDOMENCT/SEGMENTATION"
RAW = ROOT / "data/nnunet_raw/Dataset503_SHIFTQC_Validation2_AbdomenCT"
TRAINER = "nnUNetTrainer_250epochs"
MODEL = ROOT / f"data/nnunet_results/Dataset503_SHIFTQC_Validation2_AbdomenCT/{TRAINER}__nnUNetPlans__2d"
SPLIT = ROOT / "outputs/VALIDATION2_ABDOMENCT/FREEZE/01_validation2_split_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    args = parser.parse_args()
    fold = args.fold
    with SPLIT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validation = [
        row for row in rows
        if row["role"] == "SEG_DEV" and int(row["segmentation_fold_if_applicable"]) == fold
    ]
    if len(validation) != 36 or Counter(row["source_dataset"] for row in validation) != Counter({"MSD Pancreas": 28, "NIH Pancreas-CT": 8}):
        raise SystemExit("frozen validation fold mismatch")

    report_path = OUTPUT / f"inference/fold_{fold}/inference_report.json"
    prediction_dir = OUTPUT / f"oof_predictions/fold_{fold}"
    input_dir = OUTPUT / f"inference/fold_{fold}/input"
    if report_path.exists() or prediction_dir.exists() or input_dir.exists():
        raise SystemExit("fold inference output exists; overwrite refused")
    checkpoint = MODEL / f"fold_{fold}/checkpoint_final.pth"
    training_report = OUTPUT / f"training/fold_{fold}/training_report.json"
    if not checkpoint.is_file() or not training_report.is_file():
        raise SystemExit("successful frozen fold training artefacts are missing")
    report = json.loads(training_report.read_text())
    if report["status"] != "PASS" or report["checkpoint_sha256"] != sha256(checkpoint):
        raise SystemExit("training report/checkpoint integrity failure")

    input_dir.mkdir(parents=True)
    prediction_dir.mkdir(parents=True)
    for row in validation:
        case = row["abdomen_case_id"]
        os.symlink(RAW / "imagesTr" / f"{case}_0000.nii.gz", input_dir / f"{case}_0000.nii.gz")
    command = [
        str(ROOT / ".venv/bin/nnUNetv2_predict"), "-i", str(input_dir), "-o", str(prediction_dir),
        "-d", "503", "-c", "2d", "-f", str(fold), "-tr", TRAINER,
        "-p", "nnUNetPlans", "-chk", "checkpoint_final.pth",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "nnUNet_raw": str(ROOT / "data/nnunet_raw"),
            "nnUNet_preprocessed": str(ROOT / "data/nnunet_preprocessed"),
            "nnUNet_results": str(ROOT / "data/nnunet_results"),
            "MPLCONFIGDIR": "/tmp/shiftqc_task034_matplotlib",
        }
    )
    started = datetime.now().astimezone().isoformat()
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"frozen fold OOF inference failed: returncode={completed.returncode}")
    expected = {row["abdomen_case_id"] for row in validation}
    observed = {path.name.removesuffix(".nii.gz") for path in prediction_dir.glob("*.nii.gz")}
    if observed != expected:
        raise SystemExit("OOF prediction identity mismatch")
    payload = {
        "task_id": "TASK-034", "fold": fold, "status": "PASS",
        "started_at": started, "ended_at": datetime.now().astimezone().isoformat(),
        "command": command, "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint), "prediction_count": len(observed),
        "validation_patient_count": 36, "tta": "standard nnU-Net mirroring enabled",
        "step_size": 0.5, "scientific_metrics_computed": [],
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
