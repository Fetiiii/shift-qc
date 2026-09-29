#!/usr/bin/env python3
"""Build result-blind AbdomenCT-1K metadata/integrity inventories.

The script never reads a prediction, computes overlap, or derives q_true/q_hat.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


EXPECTED_ARCHIVES = {
    "AbdomenCT-1K-ImagePart1.zip": {
        "md5": "946325d0209179d767bea914c93e9a54",
        "source": "https://zenodo.org/records/5903099",
        "format": "zip",
        "entries": 400,
    },
    "AbdomenCT-1K-ImagePart2.zip": {
        "md5": "eb24ba869589d15fe1d89ccf7f13aaa4",
        "source": "https://zenodo.org/records/5903846",
        "format": "zip",
        "entries": 400,
    },
    "AbdomenCT-1K-ImagePart3.zip": {
        "md5": "7f5a131e83f3bb38932e8fb62d040a3f",
        "source": "https://zenodo.org/records/5903769",
        "format": "zip",
        "entries": 262,
    },
    "Mask.7z": {
        "md5": "dfca75d15912d5d7323585c53e1d75d9",
        "source": "https://zenodo.org/records/5903769",
        "format": "7z",
        "entries": 1000,
    },
}
LABELS = {0: "background", 1: "liver", 2: "kidney", 3: "spleen", 4: "pancreas"}
CASE_RE = re.compile(r"Case_(\d{5})")


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def case_number(path: Path) -> int:
    match = CASE_RE.search(path.name)
    if match is None:
        raise ValueError(f"unrecognized case filename: {path}")
    return int(match.group(1))


def header_metadata(path: Path) -> dict[str, Any]:
    image = nib.load(path)
    affine = np.asarray(image.affine, dtype=np.float64)
    zooms = tuple(float(x) for x in image.header.get_zooms()[:3])
    return {
        "shape": tuple(int(x) for x in image.shape),
        "spacing": zooms,
        "orientation": "".join(nib.aff2axcodes(affine)),
        "affine": affine,
        "geometry_finite": bool(np.isfinite(affine).all() and np.isfinite(zooms).all()),
        "spacing_positive": bool(all(x > 0 for x in zooms)),
        "dtype": str(image.get_data_dtype()),
    }


def mask_metadata(path: Path) -> dict[str, Any]:
    base = header_metadata(path)
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)
    finite = bool(np.isfinite(data).all())
    if finite:
        rounded = np.rint(data)
        integer = bool(np.all(np.abs(data - rounded) <= 1e-5))
        normalized = rounded.astype(np.int64) if integer else None
    else:
        integer = False
        normalized = None
    if normalized is None:
        values: list[int] = []
        counts: dict[str, int] = {}
    else:
        unique, raw_counts = np.unique(normalized, return_counts=True)
        values = [int(x) for x in unique.tolist()]
        counts = {str(int(k)): int(v) for k, v in zip(unique, raw_counts, strict=True)}
    base.update(
        {
            "values_finite": finite,
            "values_near_integer_1e_5": integer,
            "label_values": values,
            "voxel_counts": counts,
            "unexpected_labels": sorted(set(values) - set(LABELS)),
            "required_organs_nonempty": {
                LABELS[label]: int(counts.get(str(label), 0)) > 0 for label in range(1, 5)
            },
        }
    )
    return base


def serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now().astimezone().isoformat()

    archive_rows = []
    for filename, expected in EXPECTED_ARCHIVES.items():
        path = root / filename
        observed = digest(path, "md5")
        archive_rows.append(
            {
                "filename": filename,
                "byte_size": path.stat().st_size,
                "local_path": str(path),
                "format": expected["format"],
                "candidate_source_cohort": "AbdomenCT-1K aggregate; case mapping unavailable",
                "extracted": True,
                "expected_entries": expected["entries"],
                "observed_md5": observed,
                "expected_md5": expected["md5"],
                "match": observed == expected["md5"],
                "source_of_expected_md5": expected["source"],
            }
        )
    json_dump(
        output / "01_archive_integrity.json",
        {
            "task": "TASK-031",
            "generated_at": started,
            "extraction": {
                "tool": "bsdtar",
                "tool_version": subprocess.check_output(["bsdtar", "--version"], text=True).splitlines()[0],
                "commands": [
                    "bsdtar -xf AbdomentCT/AbdomenCT-1K-ImagePart1.zip -C AbdomentCT/extracted",
                    "bsdtar -xf AbdomentCT/AbdomenCT-1K-ImagePart2.zip -C AbdomentCT/extracted",
                    "bsdtar -xf AbdomentCT/AbdomenCT-1K-ImagePart3.zip -C AbdomentCT/extracted",
                    "bsdtar -xf AbdomentCT/Mask.7z -C AbdomentCT/extracted/Mask",
                ],
                "original_archives_preserved": True,
            },
            "archives": archive_rows,
            "all_checksums_match": all(row["match"] for row in archive_rows),
        },
    )

    image_paths = sorted(
        (path for directory in (root / "extracted").glob("AbdomenCT-1K-ImagePart*") for path in directory.glob("*.nii.gz")),
        key=case_number,
    )
    mask_paths = {case_number(path): path for path in (root / "extracted" / "Mask").glob("*.nii.gz")}
    rows: list[dict[str, Any]] = []
    image_hash_groups: dict[str, list[str]] = defaultdict(list)
    mask_hash_groups: dict[str, list[str]] = defaultdict(list)
    geometry_groups: dict[str, list[str]] = defaultdict(list)

    for image_path in image_paths:
        number = case_number(image_path)
        case_id = f"Case_{number:05d}"
        mask_path = mask_paths.get(number)
        try:
            image_meta = header_metadata(image_path)
            image_readable = True
            image_error = None
        except Exception as error:  # hard evidence is retained in the inventory
            image_meta = {}
            image_readable = False
            image_error = f"{type(error).__name__}: {error}"
        image_sha = digest(image_path)
        image_hash_groups[image_sha].append(case_id)

        if mask_path is not None:
            try:
                current_mask_meta = mask_metadata(mask_path)
                mask_readable = True
                mask_error = None
            except Exception as error:
                current_mask_meta = {}
                mask_readable = False
                mask_error = f"{type(error).__name__}: {error}"
            mask_sha = digest(mask_path)
            mask_hash_groups[mask_sha].append(case_id)
        else:
            current_mask_meta = {}
            mask_readable = False
            mask_error = "mask missing from official 1000-mask release"
            mask_sha = None

        if image_readable:
            fingerprint_payload = {
                "shape": image_meta["shape"],
                "spacing": image_meta["spacing"],
                "orientation": image_meta["orientation"],
                "affine": np.asarray(image_meta["affine"]).round(10).tolist(),
            }
            geometry_fingerprint = hashlib.sha256(
                json.dumps(fingerprint_payload, sort_keys=True).encode()
            ).hexdigest()
            geometry_groups[geometry_fingerprint].append(case_id)
        else:
            geometry_fingerprint = None

        geometry_comparable = image_readable and mask_readable
        if geometry_comparable:
            image_affine = np.asarray(image_meta["affine"])
            mask_affine = np.asarray(current_mask_meta["affine"])
            shape_match = image_meta["shape"] == current_mask_meta["shape"]
            affine_exact = bool(np.array_equal(image_affine, mask_affine))
            affine_max_abs_diff = float(np.max(np.abs(image_affine - mask_affine)))
            orientation_match = image_meta["orientation"] == current_mask_meta["orientation"]
        else:
            shape_match = False
            affine_exact = False
            affine_max_abs_diff = None
            orientation_match = False

        required_nonempty = current_mask_meta.get("required_organs_nonempty", {})
        label_valid = bool(
            mask_readable
            and current_mask_meta.get("values_finite")
            and current_mask_meta.get("values_near_integer_1e_5")
            and not current_mask_meta.get("unexpected_labels")
        )
        mask_complete = label_valid and all(required_nonempty.values())
        geometry_valid = bool(
            geometry_comparable
            and shape_match
            and affine_exact
            and orientation_match
            and image_meta.get("geometry_finite")
            and image_meta.get("spacing_positive")
            and current_mask_meta.get("geometry_finite")
            and current_mask_meta.get("spacing_positive")
        )
        preliminary = (
            "PROVENANCE_UNRESOLVED"
            if image_readable and mask_readable and geometry_valid and mask_complete
            else "INELIGIBLE_METADATA_OR_ANNOTATION"
        )
        rows.append(
            {
                "case_id": case_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path else None,
                "candidate_source_dataset": "PROVENANCE_UNRESOLVED",
                "image_exists": image_path.exists(),
                "mask_exists": mask_path is not None and mask_path.exists(),
                "image_readable": image_readable,
                "mask_readable": mask_readable,
                "image_error": image_error,
                "mask_error": mask_error,
                "shape_image": serializable(image_meta.get("shape")),
                "shape_mask": serializable(current_mask_meta.get("shape")),
                "spacing": serializable(image_meta.get("spacing")),
                "spacing_mask": serializable(current_mask_meta.get("spacing")),
                "orientation": image_meta.get("orientation"),
                "orientation_mask": current_mask_meta.get("orientation"),
                "affine_match": affine_exact,
                "affine_max_abs_diff": affine_max_abs_diff,
                "shape_match": shape_match,
                "orientation_match": orientation_match,
                "geometry_valid": geometry_valid,
                "label_values": current_mask_meta.get("label_values", []),
                "nonzero_voxels_by_label": {
                    key: value
                    for key, value in current_mask_meta.get("voxel_counts", {}).items()
                    if key != "0"
                },
                "required_organs_nonempty": required_nonempty,
                "unexpected_labels": current_mask_meta.get("unexpected_labels", []),
                "mask_values_finite": current_mask_meta.get("values_finite"),
                "mask_values_near_integer_1e_5": current_mask_meta.get("values_near_integer_1e_5"),
                "image_sha256": image_sha,
                "mask_sha256": mask_sha,
                "geometry_fingerprint_sha256": geometry_fingerprint,
                "eligibility_preliminary": preliminary,
            }
        )

    json_dump(output / "02_case_inventory.json", rows)
    csv_fields = list(rows[0])
    with (output / "02_case_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()}
            )

    provenance_rows = [
        {
            "case_id": row["case_id"],
            "source_dataset": "PROVENANCE_UNRESOLVED",
            "original_case_id": None,
            "center_site": None,
            "scanner_vendor": None,
            "contrast_phase": None,
            "disease_cohort": None,
            "abdomenct1k_harmonized_identifier": row["case_id"],
            "authority": "No official case-level source mapping found in local release or official metadata",
        }
        for row in rows
    ]
    json_dump(output / "03_case_provenance_manifest.json", provenance_rows)
    with (output / "03_case_provenance_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(provenance_rows[0]))
        writer.writeheader()
        writer.writerows(provenance_rows)

    duplicate_report = {
        "task": "TASK-031",
        "identifier_duplicates": [],
        "exact_image_sha256_duplicate_groups": sorted(
            (group for group in image_hash_groups.values() if len(group) > 1), key=lambda x: x[0]
        ),
        "exact_mask_sha256_duplicate_groups": sorted(
            (group for group in mask_hash_groups.values() if len(group) > 1), key=lambda x: x[0]
        ),
        "geometry_fingerprint_collision_group_count": sum(len(group) > 1 for group in geometry_groups.values()),
        "geometry_note": "Geometry collisions alone are not treated as duplicates.",
        "learned_or_quality_fingerprint_used": False,
        "source_ood_overlap_status": "UNRESOLVED_BECAUSE_CASE_LEVEL_PROVENANCE_IS_UNRESOLVED",
    }
    json_dump(output / "04_duplicate_overlap_audit.json", duplicate_report)

    json_dump(
        output / "05_patient_identity_audit.json",
        {
            "task": "TASK-031",
            "classification": "UNRESOLVED",
            "harmonized_case_ids_unique": len({row["case_id"] for row in rows}) == len(rows),
            "official_release_describes_ct_scans_or_cases": True,
            "case_to_original_patient_mapping_available": False,
            "cross_component_patient_overlap_excludable": False,
            "bootstrap_unit_frozen": False,
            "reason": "Unique harmonized volumes do not prove one unique patient per volume or exclude cross-component patient overlap without original identifiers.",
        },
    )

    missing_masks = [row["case_id"] for row in rows if not row["mask_exists"]]
    incomplete = [
        {"case_id": row["case_id"], "required_organs_nonempty": row["required_organs_nonempty"]}
        for row in rows
        if row["mask_exists"] and not all(row["required_organs_nonempty"].values())
    ]
    unexpected = [
        {"case_id": row["case_id"], "unexpected_labels": row["unexpected_labels"]}
        for row in rows if row["unexpected_labels"]
    ]
    geometry_failures = [row["case_id"] for row in rows if row["mask_exists"] and not row["geometry_valid"]]
    json_dump(
        output / "06_mask_label_contract.json",
        {
            "task": "TASK-031",
            "authority": [
                "https://flare.grand-challenge.org/Data/",
                "https://github.com/JunMa11/AbdomenCT-1K/blob/main/2-Semi-supervisedLearning/nnUNet/nnunet/dataset_conversion/Task233_SplPanLiTSKiTS.py",
            ],
            "labels": {str(key): value for key, value in LABELS.items()},
            "kidney_semantics": "combined kidney class; left and right are not separate labels",
            "required_organs": [LABELS[index] for index in range(1, 5)],
            "missing_mask_cases": missing_masks,
            "unexpected_label_cases": unexpected,
            "empty_required_organ_cases": incomplete,
            "geometry_failure_cases": geometry_failures,
            "gt_empty_required_organ_interpretation": "annotation incomplete/invalid for Validation-2; not biological absence",
        },
    )
    completeness_fields = [
        "case_id", "mask_exists", "mask_readable", "label_values", "required_organs_nonempty",
        "unexpected_labels", "geometry_valid", "eligibility_preliminary",
    ]
    with (output / "06_mask_completeness.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=completeness_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: json.dumps(row[key], sort_keys=True) if isinstance(row[key], (dict, list)) else row[key] for key in completeness_fields}
            )

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "started_at": started,
        "python": sys.version,
        "platform": platform.platform(),
        "nibabel": nib.__version__,
        "numpy": np.__version__,
        "image_count": len(rows),
        "mask_count": sum(row["mask_exists"] for row in rows),
        "image_readable_count": sum(row["image_readable"] for row in rows),
        "mask_readable_count": sum(row["mask_readable"] for row in rows),
        "geometry_valid_count_among_masks": sum(row["geometry_valid"] for row in rows),
        "complete_four_organ_mask_count": sum(
            row["mask_exists"] and all(row["required_organs_nonempty"].values()) for row in rows
        ),
        "provenance_resolved_count": 0,
        "scientific_values_computed": False,
        "predictions_accessed": False,
    }
    json_dump(output / "00_preflight_inventory_summary.json", summary)


if __name__ == "__main__":
    main()
