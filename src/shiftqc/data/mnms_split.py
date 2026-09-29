"""Deterministic patient-level M3 split freeze for TASK-019."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from shiftqc.data.mnms_audit import canonical_hash


SOURCE_SPLITS = ("SEG_TRAIN", "SEG_VAL", "QC_TRAIN", "CALIBRATION")
SCIENTIFIC_ROLES = ("SOURCE", "M3_ID_EVAL", "M3_OOD_EVAL")
AUDIT_METADATA_COLUMNS = (
    "patient_id",
    "vendor",
    "official_partition",
    "image_path_relative",
    "gt_path_relative",
    "has_image",
    "has_gt",
    "ed_frame",
    "es_frame",
)
MANIFEST_COLUMNS = (
    "patient_id",
    "vendor",
    "official_partition",
    "scientific_role",
    "source_split",
    "image_path_relative",
    "gt_path_relative",
    "ed_frame",
    "es_frame",
    "in_scientific_population",
    "exclusion_reason",
)
FORBIDDEN_TARGET_CONTENT_COLUMNS = (
    "label_values",
    "unexpected_label_values",
    "missing_classes",
    "nonzero_gt_frames",
    "image_shape",
    "gt_shape",
    "spacing",
    "gt_spacing",
)


class MnmsSplitFailure(RuntimeError):
    """Raised when a frozen TASK-019 invariant is violated."""

    def __init__(self, code: str, details: Any = None):
        self.code = code
        self.details = details
        super().__init__(f"{code}: {details}")


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_audit_metadata(path: Path) -> pd.DataFrame:
    """Read only split-authorized metadata, never target GT contents/statistics."""
    return pd.read_parquet(path, columns=list(AUDIT_METADATA_COLUMNS))


def patient_id_hash(patient_ids: Any) -> str:
    """Hash a sorted patient-ID set using canonical JSON encoding."""
    return canonical_hash(sorted(str(value) for value in patient_ids))


def _validate_relative_path(value: Any, *, column: str, patient_id: str) -> None:
    if value is None or pd.isna(value):
        raise MnmsSplitFailure("missing_relative_path", {"patient_id": patient_id, "column": column})
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise MnmsSplitFailure(
            "machine_local_or_escaping_path",
            {"patient_id": patient_id, "column": column, "path": str(value)},
        )


def _validated_audit(audit: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    missing = sorted(set(AUDIT_METADATA_COLUMNS) - set(audit.columns))
    if missing:
        raise MnmsSplitFailure("audit_metadata_columns_missing", missing)
    frame = audit.loc[:, list(AUDIT_METADATA_COLUMNS)].copy()
    expected_total = (
        int(config["source"]["expected_total"])
        + sum(sum(role.values()) for role in config["evaluation"]["roles"].values())
        + int(config["excluded"]["count"])
    )
    if len(frame) != expected_total:
        raise MnmsSplitFailure(
            "dataset_inventory_count_mismatch",
            {"expected": expected_total, "observed": len(frame)},
        )
    if frame["patient_id"].isna().any() or frame["patient_id"].duplicated().any():
        duplicates = sorted(
            frame.loc[frame["patient_id"].duplicated(False), "patient_id"].astype(str).tolist()
        )
        raise MnmsSplitFailure("patient_id_integrity_failure", duplicates)
    if not frame["has_image"].eq(True).all():
        raise MnmsSplitFailure(
            "required_image_missing",
            frame.loc[~frame["has_image"], "patient_id"].astype(str).tolist(),
        )
    for row in frame.itertuples(index=False):
        _validate_relative_path(
            row.image_path_relative, column="image_path_relative", patient_id=str(row.patient_id)
        )
        _validate_relative_path(
            row.gt_path_relative, column="gt_path_relative", patient_id=str(row.patient_id)
        )

    source = frame.loc[
        frame["official_partition"].eq(config["source"]["official_partition"])
    ]
    observed_source = source["vendor"].value_counts().sort_index().to_dict()
    expected_source = {
        vendor: int(config["source"]["expected_per_vendor"])
        for vendor in config["source"]["vendors"]
    }
    if observed_source != expected_source or len(source) != int(config["source"]["expected_total"]):
        raise MnmsSplitFailure(
            "source_inventory_mismatch",
            {"expected": expected_source, "observed": observed_source},
        )
    if not source["has_gt"].eq(True).all():
        raise MnmsSplitFailure("source_gt_presence_mismatch", None)

    evaluation = frame.loc[
        frame["official_partition"].isin(config["evaluation"]["official_partitions"])
    ]
    expected_eval: dict[str, int] = {}
    for role_counts in config["evaluation"]["roles"].values():
        for vendor, count in role_counts.items():
            expected_eval[vendor] = expected_eval.get(vendor, 0) + int(count)
    observed_eval = evaluation["vendor"].value_counts().sort_index().to_dict()
    if observed_eval != expected_eval:
        raise MnmsSplitFailure(
            "evaluation_inventory_mismatch",
            {"expected": expected_eval, "observed": observed_eval},
        )
    if not evaluation["has_gt"].eq(True).all():
        raise MnmsSplitFailure("evaluation_gt_presence_mismatch", None)

    excluded_cfg = config["excluded"]
    excluded = frame.loc[
        frame["official_partition"].eq(excluded_cfg["official_partition"])
        & frame["vendor"].eq(excluded_cfg["vendor"])
    ]
    if len(excluded) != int(excluded_cfg["count"]):
        raise MnmsSplitFailure(
            "excluded_inventory_mismatch",
            {"expected": int(excluded_cfg["count"]), "observed": len(excluded)},
        )
    recognized = set(source["patient_id"]) | set(evaluation["patient_id"]) | set(excluded["patient_id"])
    if recognized != set(frame["patient_id"]):
        raise MnmsSplitFailure("unrecognized_inventory_rows", sorted(set(frame["patient_id"]) - recognized))
    return frame


def _vendor_source_assignments(
    source: pd.DataFrame, vendor: str, config: dict[str, Any]
) -> dict[str, str]:
    patient_ids = np.asarray(
        sorted(source.loc[source["vendor"].eq(vendor), "patient_id"].astype(str).tolist()),
        dtype=object,
    )
    expected = int(config["source"]["expected_per_vendor"])
    if len(patient_ids) != expected:
        raise MnmsSplitFailure(
            "vendor_source_count_mismatch",
            {"vendor": vendor, "expected": expected, "observed": len(patient_ids)},
        )
    # The RNG is reset independently for each vendor, as frozen in config.
    permutation = np.random.default_rng(int(config["seed"])).permutation(patient_ids)
    counts = config["source"]["counts_per_vendor"]
    assignments: dict[str, str] = {}
    start = 0
    for split_name in config["source"]["assignment_order"]:
        stop = start + int(counts[split_name])
        assignments.update({str(patient_id): split_name for patient_id in permutation[start:stop]})
        start = stop
    if start != len(permutation):
        raise MnmsSplitFailure(
            "source_split_arithmetic_mismatch",
            {"vendor": vendor, "assigned": start, "available": len(permutation)},
        )
    return assignments


def generate_split_manifest(audit: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Generate the complete 345-row inventory with frozen scientific roles."""
    frame = _validated_audit(audit, config)
    source_partition = config["source"]["official_partition"]
    source = frame.loc[frame["official_partition"].eq(source_partition)]
    assignment: dict[str, str] = {}
    for vendor in config["source"]["vendors"]:
        assignment.update(_vendor_source_assignments(source, vendor, config))

    result = frame[
        [
            "patient_id",
            "vendor",
            "official_partition",
            "image_path_relative",
            "gt_path_relative",
            "ed_frame",
            "es_frame",
        ]
    ].copy()
    result["scientific_role"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["source_split"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["exclusion_reason"] = pd.Series(pd.NA, index=result.index, dtype="string")

    is_source = result["official_partition"].eq(source_partition)
    result.loc[is_source, "scientific_role"] = "SOURCE"
    result.loc[is_source, "source_split"] = result.loc[is_source, "patient_id"].map(assignment)

    eval_partitions = config["evaluation"]["official_partitions"]
    is_evaluation = result["official_partition"].isin(eval_partitions)
    id_vendors = set(config["evaluation"]["roles"]["M3_ID_EVAL"])
    ood_vendors = set(config["evaluation"]["roles"]["M3_OOD_EVAL"])
    result.loc[is_evaluation & result["vendor"].isin(id_vendors), "scientific_role"] = "M3_ID_EVAL"
    result.loc[is_evaluation & result["vendor"].isin(ood_vendors), "scientific_role"] = "M3_OOD_EVAL"

    excluded_cfg = config["excluded"]
    is_excluded = result["official_partition"].eq(excluded_cfg["official_partition"])
    result.loc[is_excluded, "exclusion_reason"] = excluded_cfg["reason"]
    result["in_scientific_population"] = result["scientific_role"].notna()

    partition_order = {
        "training_labeled": 0,
        "training_unlabeled": 1,
        "validation": 2,
        "test": 3,
    }
    result["_partition_order"] = result["official_partition"].map(partition_order)
    result = result.sort_values(
        ["_partition_order", "vendor", "patient_id"], kind="stable", ignore_index=True
    ).drop(columns="_partition_order")
    result = result.loc[:, list(MANIFEST_COLUMNS)]
    validate_split_manifest(result, frame, config)
    return result


def overlap_checks(manifest: pd.DataFrame, config: dict[str, Any]) -> dict[str, int]:
    split_sets = {
        name: set(manifest.loc[manifest["source_split"].eq(name), "patient_id"])
        for name in SOURCE_SPLITS
    }
    pairwise = sum(
        len(split_sets[left] & split_sets[right])
        for index, left in enumerate(SOURCE_SPLITS)
        for right in SOURCE_SPLITS[index + 1 :]
    )
    source = set(manifest.loc[manifest["scientific_role"].eq("SOURCE"), "patient_id"])
    id_eval = set(manifest.loc[manifest["scientific_role"].eq("M3_ID_EVAL"), "patient_id"])
    ood_eval = set(manifest.loc[manifest["scientific_role"].eq("M3_OOD_EVAL"), "patient_id"])
    excluded = manifest.loc[manifest["official_partition"].eq(config["excluded"]["official_partition"])]
    return {
        "source_split_pairwise_intersections": int(pairwise),
        "source_evaluation_intersection": len(source & (id_eval | ood_eval)),
        "id_ood_evaluation_intersection": len(id_eval & ood_eval),
        "excluded_with_scientific_role": int(excluded["scientific_role"].notna().sum()),
        "excluded_with_source_split": int(excluded["source_split"].notna().sum()),
        "vendor_c_or_d_in_source_split": int(
            (manifest["source_split"].notna() & manifest["vendor"].isin(("C", "D"))).sum()
        ),
        "validation_or_test_in_source_split": int(
            (
                manifest["source_split"].notna()
                & manifest["official_partition"].isin(config["evaluation"]["official_partitions"])
            ).sum()
        ),
    }


def manifest_counts(manifest: pd.DataFrame) -> dict[str, Any]:
    source_counts = (
        manifest.loc[manifest["scientific_role"].eq("SOURCE")]
        .groupby(["source_split", "vendor"], observed=True)
        .size()
    )
    evaluation_counts = (
        manifest.loc[manifest["scientific_role"].isin(("M3_ID_EVAL", "M3_OOD_EVAL"))]
        .groupby(["scientific_role", "vendor"], observed=True)
        .size()
    )
    official_counts = manifest.groupby(["official_partition", "vendor"], observed=True).size()
    return {
        "dataset_inventory": int(len(manifest)),
        "scientific_included": int(manifest["in_scientific_population"].sum()),
        "source_total": int(manifest["scientific_role"].eq("SOURCE").sum()),
        "source_splits": {
            name: {
                vendor: int(source_counts.get((name, vendor), 0))
                for vendor in ("A", "B")
            }
            | {"total": int(sum(source_counts.get((name, vendor), 0) for vendor in ("A", "B")))}
            for name in SOURCE_SPLITS
        },
        "evaluation": {
            role: {
                vendor: int(evaluation_counts.get((role, vendor), 0))
                for vendor in (("A", "B") if role == "M3_ID_EVAL" else ("C", "D"))
            }
            for role in ("M3_ID_EVAL", "M3_OOD_EVAL")
        },
        "evaluation_totals": {
            role: int(manifest["scientific_role"].eq(role).sum())
            for role in ("M3_ID_EVAL", "M3_OOD_EVAL")
        },
        "excluded_vendor_c_training_unlabeled": int(
            manifest["official_partition"].eq("training_unlabeled").sum()
        ),
        "official_partition_by_vendor": {
            partition: {
                vendor: int(official_counts.get((partition, vendor), 0))
                for vendor in ("A", "B", "C", "D")
                if int(official_counts.get((partition, vendor), 0)) > 0
            }
            for partition in ("training_labeled", "training_unlabeled", "validation", "test")
        },
    }


def validate_split_manifest(
    manifest: pd.DataFrame, audit: pd.DataFrame, config: dict[str, Any]
) -> None:
    if tuple(manifest.columns) != MANIFEST_COLUMNS:
        raise MnmsSplitFailure(
            "manifest_schema_mismatch",
            {"expected": list(MANIFEST_COLUMNS), "observed": list(manifest.columns)},
        )
    if len(manifest) != len(audit) or not manifest["patient_id"].is_unique:
        raise MnmsSplitFailure("manifest_patient_integrity_failure", len(manifest))
    if set(manifest["patient_id"]) != set(audit["patient_id"]):
        raise MnmsSplitFailure("manifest_inventory_set_mismatch", None)
    counts = manifest_counts(manifest)
    expected_per_vendor = config["source"]["counts_per_vendor"]
    for split_name in SOURCE_SPLITS:
        observed = counts["source_splits"][split_name]
        expected = int(expected_per_vendor[split_name])
        if observed != {"A": expected, "B": expected, "total": 2 * expected}:
            raise MnmsSplitFailure(
                "source_split_count_mismatch",
                {"split": split_name, "expected_per_vendor": expected, "observed": observed},
            )
    for role, expected in config["evaluation"]["roles"].items():
        observed = counts["evaluation"][role]
        if observed != {vendor: int(count) for vendor, count in expected.items()}:
            raise MnmsSplitFailure(
                "evaluation_role_count_mismatch",
                {"role": role, "expected": expected, "observed": observed},
            )
    if counts["excluded_vendor_c_training_unlabeled"] != int(config["excluded"]["count"]):
        raise MnmsSplitFailure("excluded_count_mismatch", counts)
    checks = overlap_checks(manifest, config)
    if any(checks.values()):
        raise MnmsSplitFailure("patient_overlap_or_firewall_failure", checks)
    for column in ("image_path_relative", "gt_path_relative"):
        for patient_id, value in manifest[["patient_id", column]].itertuples(index=False):
            _validate_relative_path(value, column=column, patient_id=str(patient_id))


def provenance_hashes(manifest: pd.DataFrame) -> dict[str, Any]:
    return {
        "source_patient_ids": patient_id_hash(
            manifest.loc[manifest["scientific_role"].eq("SOURCE"), "patient_id"]
        ),
        "id_evaluation_patient_ids": patient_id_hash(
            manifest.loc[manifest["scientific_role"].eq("M3_ID_EVAL"), "patient_id"]
        ),
        "ood_evaluation_patient_ids": patient_id_hash(
            manifest.loc[manifest["scientific_role"].eq("M3_OOD_EVAL"), "patient_id"]
        ),
        "per_source_split_patient_ids": {
            split_name: patient_id_hash(
                manifest.loc[manifest["source_split"].eq(split_name), "patient_id"]
            )
            for split_name in SOURCE_SPLITS
        },
        "per_scientific_role_patient_ids": {
            role: patient_id_hash(
                manifest.loc[manifest["scientific_role"].eq(role), "patient_id"]
            )
            for role in SCIENTIFIC_ROLES
        },
    }


def _json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def build_json_manifest(manifest: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Build a timestamp-free deterministic JSON twin of the Parquet manifest."""
    rows = [
        {column: _json_value(value) for column, value in zip(MANIFEST_COLUMNS, row, strict=True)}
        for row in manifest.itertuples(index=False, name=None)
    ]
    return {
        "schema_version": 1,
        "task": config["task"],
        "experiment_id": config["experiment_id"],
        "status": "FROZEN",
        "seed": int(config["seed"]),
        "upstream": {
            "task018_audit_manifest": config["task018_audit"]["manifest_path"],
            "task018_audit_sha256": config["task018_audit"]["manifest_sha256"],
            "m3_spec": config["m3_spec"],
        },
        "randomization": config["randomization"],
        "primary_analysis_policy": config["analysis_policy"],
        "counts": manifest_counts(manifest),
        "overlap_checks": overlap_checks(manifest, config),
        "patient_id_hashes": provenance_hashes(manifest),
        "rows": rows,
    }


def scientifically_identical(existing: pd.DataFrame, expected: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(existing, expected, check_dtype=True, check_like=False)
    except AssertionError:
        return False
    return True


def manifest_content_hash(manifest: pd.DataFrame) -> str:
    """Hash the deterministic scientific row contents independent of Parquet encoding."""
    return hashlib.sha256(
        manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
