#!/usr/bin/env python3
"""Materialize TASK-033 identity-only split/fold artefacts.

This command reads only TASK-032 provenance and eligibility metadata. It has no
prediction, mask, quality, QC, or endpoint input.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from shiftqc.validation2.freeze import make_fixed_quota_split, make_segmentation_fold_assignments


SOURCE_DATASETS = {"MSD Pancreas", "NIH Pancreas-CT"}
OOD_DATASET = "KiTS19"
FIELDS = (
    "abdomen_case_id",
    "source_dataset",
    "source_case_id",
    "source_patient_id",
    "role",
    "segmentation_fold_if_applicable",
    "eligibility_status",
)


def _truth(value: str) -> bool:
    return value.strip().lower() == "true"


def build(input_path: Path, output_dir: Path) -> None:
    with input_path.open(newline="", encoding="utf-8") as handle:
        input_rows = list(csv.DictReader(handle))
    eligible = [row for row in input_rows if _truth(row["eligible_after_provenance"])]
    source_rows = [
        row
        for row in eligible
        if row["source_dataset"] == "NIH Pancreas-CT"
        or (row["source_dataset"] == "MSD Pancreas" and row["source_partition"] == "imagesTr")
    ]
    ood_rows = [row for row in eligible if row["source_dataset"] == OOD_DATASET]
    if Counter(row["source_dataset"] for row in source_rows) != Counter(
        {"MSD Pancreas": 276, "NIH Pancreas-CT": 80}
    ):
        raise RuntimeError("eligible source counts do not match frozen authority")
    if len(ood_rows) != 245:
        raise RuntimeError("eligible KiTS19 count does not match frozen authority")

    split = make_fixed_quota_split(
        [row["source_patient_id"] for row in source_rows],
        [row["source_case_id"] for row in source_rows],
        [row["source_dataset"] for row in source_rows],
    )
    seg_rows = [row for row in source_rows if split[row["source_patient_id"]] == "SEG_DEV"]
    folds = make_segmentation_fold_assignments(
        [row["source_patient_id"] for row in seg_rows],
        [row["source_case_id"] for row in seg_rows],
        [row["source_dataset"] for row in seg_rows],
    )

    manifest: list[dict[str, object]] = []
    for row in source_rows:
        patient = row["source_patient_id"]
        role = split[patient]
        manifest.append(
            {
                "abdomen_case_id": row["abdomen_case_id"],
                "source_dataset": row["source_dataset"],
                "source_case_id": row["source_case_id"],
                "source_patient_id": patient,
                "role": role,
                "segmentation_fold_if_applicable": folds[patient] if role == "SEG_DEV" else "",
                "eligibility_status": "ELIGIBLE",
            }
        )
    for row in ood_rows:
        manifest.append(
            {
                "abdomen_case_id": row["abdomen_case_id"],
                "source_dataset": row["source_dataset"],
                "source_case_id": row["source_case_id"],
                "source_patient_id": row["source_patient_id"],
                "role": "OOD_EVAL",
                "segmentation_fold_if_applicable": "",
                "eligibility_status": "ELIGIBLE",
            }
        )
    manifest.sort(key=lambda row: (str(row["role"]), str(row["source_dataset"]), str(row["source_case_id"])))
    if len({(row["source_dataset"], row["source_patient_id"]) for row in manifest}) != 601:
        raise RuntimeError("manifest patient identities are not unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "01_validation2_split_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest)
    json_path = output_dir / "01_validation2_split_manifest.json"
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # nnU-Net's splits_final.json uses one object per validation fold.
    seg_by_fold = {fold: [] for fold in range(5)}
    for row in manifest:
        if row["role"] == "SEG_DEV":
            seg_by_fold[int(row["segmentation_fold_if_applicable"])].append(row["abdomen_case_id"])
    all_seg = sorted(case for values in seg_by_fold.values() for case in values)
    nnunet = []
    for fold in range(5):
        validation = sorted(seg_by_fold[fold])
        nnunet.append({"train": sorted(set(all_seg) - set(validation)), "val": validation})
    (output_dir / "01_validation2_nnunet_splits_final.json").write_text(
        json.dumps(nnunet, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/VALIDATION2_ABDOMENCT/PROVENANCE/07_provenance_resolved_eligibility.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/VALIDATION2_ABDOMENCT/FREEZE")
    )
    args = parser.parse_args()
    build(args.input, args.output_dir)


if __name__ == "__main__":
    main()
