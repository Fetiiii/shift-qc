#!/usr/bin/env python3
"""Verify frozen authority and materialize the SEG_DEV-only nnU-Net dataset."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "outputs/VALIDATION2_ABDOMENCT/FREEZE"
OUTPUT = ROOT / "outputs/VALIDATION2_ABDOMENCT/SEGMENTATION"
RAW = ROOT / "data/nnunet_raw/Dataset503_SHIFTQC_Validation2_AbdomenCT"
PREPROCESSED = ROOT / "data/nnunet_preprocessed/Dataset503_SHIFTQC_Validation2_AbdomenCT"
RESULTS = ROOT / "data/nnunet_results/Dataset503_SHIFTQC_Validation2_AbdomenCT"
TAG = "pre-abdomenct-validation2-training"
FREEZE_COMMIT = "986c04fd7695089d6c986913ce8ce1e3f90de8e9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def environment_record() -> dict[str, object]:
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    gpus = []
    for line in query:
        index, name, driver, total, free = [part.strip() for part in line.split(",")]
        gpus.append(
            {
                "index": int(index), "name": name, "driver": driver,
                "memory_total_mib": int(total), "memory_free_mib": int(free),
            }
        )
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            memory[f"{key}_bytes"] = int(value.strip().split()[0]) * 1024
    disk = shutil.disk_usage(ROOT)
    import torch

    return {
        "task_id": "TASK-034",
        "captured_at": datetime.now().astimezone().isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "working_tree_porcelain": git("status", "--porcelain").splitlines(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": subprocess.check_output(["lscpu"], text=True),
        "memory": memory,
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "gpu": gpus,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "torch_cuda_available": torch.cuda.is_available(),
        "nnunetv2": importlib.metadata.version("nnunetv2"),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "commands": {
            "planning": "nnUNetv2_plan_and_preprocess -d 503 --verify_dataset_integrity -c 2d",
            "training": "PYTHONHASHSEED=2026 CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python scripts/task034_train_fold.py --fold <0..4> [resolves nnUNetTrainer_250epochs]",
            "oof_inference": ".venv/bin/python scripts/task034_oof_inference.py --fold <0..4>",
        },
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protected_outputs = [
        OUTPUT / "00_freeze_integrity.json",
        OUTPUT / "01_training_environment.json",
        OUTPUT / "01_dataset_leakage_audit.json",
    ]
    if any(path.exists() for path in protected_outputs):
        raise SystemExit("refusing to overwrite existing TASK-034 pre-training artefact")
    resolved = git("rev-parse", f"{TAG}^{{}}")
    if resolved != FREEZE_COMMIT:
        raise SystemExit(f"freeze tag mismatch: {resolved} != {FREEZE_COMMIT}")
    authority_path = FREEZE / "02_validation2_freeze_authority_manifest.json"
    authority = json.loads(authority_path.read_text())
    mismatches = []
    for relative, expected in authority["authority_files"].items():
        path = ROOT / relative
        observed = sha256(path) if path.is_file() else None
        if observed != expected["sha256"]:
            mismatches.append({"path": relative, "expected": expected["sha256"], "observed": observed})
    if mismatches:
        raise SystemExit(f"frozen authority mismatch: {mismatches}")
    if any(path.exists() for path in (RAW, PREPROCESSED, RESULTS)):
        raise SystemExit("Dataset503 raw/preprocessed/results path already exists")

    split_rows = read_csv(FREEZE / "01_validation2_split_manifest.csv")
    inventory_rows = read_csv(ROOT / "outputs/VALIDATION2_ABDOMENCT/PREFLIGHT/02_case_inventory.csv")
    inventory = {row["case_id"]: row for row in inventory_rows}
    seg_rows = [row for row in split_rows if row["role"] == "SEG_DEV"]
    if len(seg_rows) != 180 or len({row["source_patient_id"] for row in seg_rows}) != 180:
        raise SystemExit("SEG_DEV identity count mismatch")
    role_counts = Counter(row["role"] for row in split_rows)
    expected_roles = Counter({"SEG_DEV": 180, "QC_TRAIN": 55, "CALIBRATION": 55, "ID_EVAL": 66, "OOD_EVAL": 245})
    if role_counts != expected_roles:
        raise SystemExit("frozen role counts mismatch")
    fold_counts = Counter((row["source_dataset"], row["segmentation_fold_if_applicable"]) for row in seg_rows)
    if any(fold_counts[(source, str(fold))] != count for source, count in (("MSD Pancreas", 28), ("NIH Pancreas-CT", 8)) for fold in range(5)):
        raise SystemExit("frozen fold composition mismatch")

    checked = []
    for row in sorted(seg_rows, key=lambda value: value["abdomen_case_id"]):
        case = row["abdomen_case_id"]
        source = inventory[case]
        image, mask = Path(source["image_path"]), Path(source["mask_path"])
        observed_image, observed_mask = sha256(image), sha256(mask)
        if observed_image != source["image_sha256"] or observed_mask != source["mask_sha256"]:
            raise SystemExit(f"source file hash mismatch: {case}")
        if source["label_values"] != "[0, 1, 2, 3, 4]" or source["required_organs_nonempty"] != '{"kidney": true, "liver": true, "pancreas": true, "spleen": true}':
            raise SystemExit(f"label contract mismatch: {case}")
        if source["geometry_valid"] != "True" or source["mask_values_finite"] != "True" or source["mask_values_near_integer_1e_5"] != "True":
            raise SystemExit(f"geometry/label validity mismatch: {case}")
        checked.append(
            {
                "patient_id": row["source_patient_id"], "abdomen_case_id": case,
                "source_dataset": row["source_dataset"], "fold": int(row["segmentation_fold_if_applicable"]),
                "image_path": str(image), "mask_path": str(mask),
                "image_sha256": observed_image, "mask_sha256": observed_mask,
            }
        )

    staging = RAW.with_name(RAW.name + ".staging")
    if staging.exists():
        raise SystemExit(f"staging path already exists: {staging}")
    (staging / "imagesTr").mkdir(parents=True)
    (staging / "labelsTr").mkdir()
    for row in checked:
        case = row["abdomen_case_id"]
        os.symlink(row["image_path"], staging / "imagesTr" / f"{case}_0000.nii.gz")
        os.symlink(row["mask_path"], staging / "labelsTr" / f"{case}.nii.gz")
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "liver": 1, "kidney": 2, "spleen": 3, "pancreas": 4},
        "numTraining": 180,
        "file_ending": ".nii.gz",
        "name": "SHIFTQC_Validation2_AbdomenCT",
        "description": "TASK-034 frozen SEG_DEV-only AbdomenCT Validation-2 export",
    }
    atomic_json(staging / "dataset.json", dataset_json)
    shutil.copy2(FREEZE / "01_validation2_nnunet_splits_final.json", staging / "splits_final.json")
    staging.replace(RAW)

    raw_ids = {path.name.removesuffix("_0000.nii.gz") for path in (RAW / "imagesTr").iterdir()}
    label_ids = {path.name.removesuffix(".nii.gz") for path in (RAW / "labelsTr").iterdir()}
    expected_ids = {row["abdomen_case_id"] for row in seg_rows}
    forbidden_ids = {row["abdomen_case_id"] for row in split_rows if row["role"] != "SEG_DEV"}
    if raw_ids != expected_ids or label_ids != expected_ids or raw_ids & forbidden_ids:
        raise SystemExit("raw Dataset503 role leakage")

    freeze_record = {
        "task_id": "TASK-034", "status": "PASS", "freeze_tag": TAG,
        "expected_commit": FREEZE_COMMIT, "resolved_commit": resolved,
        "authority_manifest_sha256": sha256(authority_path),
        "authority_file_count": len(authority["authority_files"]),
        "authority_hash_mismatches": mismatches,
    }
    leakage = {
        "task_id": "TASK-034", "status": "PASS", "checked_at": datetime.now().astimezone().isoformat(),
        "role_counts": dict(role_counts), "seg_dev_n": len(seg_rows),
        "seg_dev_source_counts": dict(Counter(row["source_dataset"] for row in seg_rows)),
        "fold_source_counts": {f"{source}|{fold}": count for (source, fold), count in sorted(fold_counts.items())},
        "raw_image_n": len(raw_ids), "raw_label_n": len(label_ids),
        "forbidden_role_ids_in_raw": sorted(raw_ids & forbidden_ids),
        "duplicate_source_patient_ids": 0, "source_file_hash_checks": checked,
        "dataset_json_sha256": sha256(RAW / "dataset.json"),
        "raw_splits_sha256": sha256(RAW / "splits_final.json"),
    }
    atomic_json(OUTPUT / "00_freeze_integrity.json", freeze_record)
    atomic_json(OUTPUT / "01_training_environment.json", environment_record())
    atomic_json(OUTPUT / "01_dataset_leakage_audit.json", leakage)
    print(json.dumps({"status": "PASS", "seg_dev_n": 180, "raw": str(RAW)}, indent=2))


if __name__ == "__main__":
    main()
