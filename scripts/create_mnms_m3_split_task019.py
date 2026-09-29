#!/usr/bin/env python3
"""Create or verify the frozen TASK-019 M&Ms patient-level split."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from shiftqc.data.mnms_audit import frozen_experiment_tree_hash
from shiftqc.data.mnms_split import (
    MnmsSplitFailure,
    build_json_manifest,
    generate_split_manifest,
    load_audit_metadata,
    load_config,
    manifest_content_hash,
    manifest_counts,
    overlap_checks,
    provenance_hashes,
    scientifically_identical,
)
from shiftqc.data.nnunet_export import sha256_file


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs/splits/mnms_m3_split.json"


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


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise MnmsSplitFailure("frozen_input_missing", {"label": label, "path": str(path)})
    observed = sha256_file(path)
    if observed != expected:
        raise MnmsSplitFailure(
            "frozen_input_hash_mismatch",
            {"label": label, "expected": expected, "observed": observed},
        )


def verify_inputs(config: dict[str, Any]) -> dict[str, Any]:
    spec = REPOSITORY / config["m3_spec"]["path"]
    audit = REPOSITORY / config["task018_audit"]["manifest_path"]
    audit_report_path = REPOSITORY / config["task018_audit"]["report_path"]
    _verify_hash(spec, config["m3_spec"]["sha256"], "M3_SPEC")
    _verify_hash(audit, config["task018_audit"]["manifest_sha256"], "TASK-018 audit manifest")
    _verify_hash(audit_report_path, config["task018_audit"]["report_sha256"], "TASK-018 audit report")
    audit_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    if audit_report.get("m3_compatibility_decision") != "M3_DATA_REVIEW_REQUIRED":
        raise MnmsSplitFailure(
            "unexpected_task018_decision", audit_report.get("m3_compatibility_decision")
        )
    expected_frozen_tree = audit_report["immutability"]["frozen_experiment_tree_after"]
    observed_frozen_tree = frozen_experiment_tree_hash(REPOSITORY)
    if observed_frozen_tree != expected_frozen_tree:
        raise MnmsSplitFailure(
            "frozen_experiment_tree_changed",
            {"expected": expected_frozen_tree, "observed": observed_frozen_tree},
        )
    return {"audit_report": audit_report, "frozen_experiment_tree": observed_frozen_tree}


def render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    source = counts["source_splits"]
    evaluation = counts["evaluation"]
    hashes = report["patient_id_hashes"]
    return f"""# TASK-019 — M3 Patient-Level Split Freeze

Status: **{report['status']}**

## Frozen protocol

- Seed: `{report['seed']}`
- Amendment: `{report['m3_spec']['amendment']}`
- Decision: `{report['m3_spec']['decision']}`
- Primary analysis: **vendor-balanced domain aggregation**
- ID weights: `0.5 × Vendor A + 0.5 × Vendor B`
- OOD weights: `0.5 × Vendor C + 0.5 × Vendor D`

## Counts

| Role | Vendor A | Vendor B | Vendor C | Vendor D | Total |
|---|---:|---:|---:|---:|---:|
| SEG_TRAIN | {source['SEG_TRAIN']['A']} | {source['SEG_TRAIN']['B']} | 0 | 0 | {source['SEG_TRAIN']['total']} |
| SEG_VAL | {source['SEG_VAL']['A']} | {source['SEG_VAL']['B']} | 0 | 0 | {source['SEG_VAL']['total']} |
| QC_TRAIN | {source['QC_TRAIN']['A']} | {source['QC_TRAIN']['B']} | 0 | 0 | {source['QC_TRAIN']['total']} |
| CALIBRATION | {source['CALIBRATION']['A']} | {source['CALIBRATION']['B']} | 0 | 0 | {source['CALIBRATION']['total']} |
| M3_ID_EVAL | {evaluation['M3_ID_EVAL']['A']} | {evaluation['M3_ID_EVAL']['B']} | 0 | 0 | {counts['evaluation_totals']['M3_ID_EVAL']} |
| M3_OOD_EVAL | 0 | 0 | {evaluation['M3_OOD_EVAL']['C']} | {evaluation['M3_OOD_EVAL']['D']} | {counts['evaluation_totals']['M3_OOD_EVAL']} |

- Dataset inventory: {counts['dataset_inventory']}
- Scientific included: {counts['scientific_included']}
- Source: {counts['source_total']}
- Excluded Vendor-C training-unlabeled: {counts['excluded_vendor_c_training_unlabeled']}

## Integrity

All overlap/firewall counters are zero: `{json.dumps(report['overlap_checks'], sort_keys=True)}`

Only patient/vendor/official-partition and path/phase provenance metadata from
the frozen TASK-018 audit were read. Target GT content was not read or used.

## Patient-ID hashes

- Source: `{hashes['source_patient_ids']}`
- ID evaluation: `{hashes['id_evaluation_patient_ids']}`
- OOD evaluation: `{hashes['ood_evaluation_patient_ids']}`

No preprocessing, training, inference, QC fitting, conformal calibration, or
outcome analysis was performed.
"""


def build_report(
    manifest: pd.DataFrame,
    config: dict[str, Any],
    *,
    started_at: str,
    input_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": config["task"],
        "experiment_id": config["experiment_id"],
        "status": "COMPLETE",
        "generation_timestamp": now(),
        "started_at": started_at,
        "seed": int(config["seed"]),
        "m3_spec": config["m3_spec"],
        "task018_audit": {
            **config["task018_audit"],
            "verified": True,
            "prior_decision": input_state["audit_report"]["m3_compatibility_decision"],
        },
        "randomization": config["randomization"],
        "counts": manifest_counts(manifest),
        "overlap_checks": overlap_checks(manifest, config),
        "patient_id_hashes": provenance_hashes(manifest),
        "manifest_scientific_content_sha256": manifest_content_hash(manifest),
        "analysis_policy": config["analysis_policy"],
        "integrity": {
            "patient_unit": True,
            "deterministic_regeneration_verified": True,
            "target_gt_content_access_count": 0,
            "target_gt_content_used_for_split": False,
            "outcome_analysis_performed": False,
            "preprocessing_performed": False,
            "training_performed": False,
            "inference_performed": False,
            "qc_fitting_performed": False,
            "conformal_calibration_performed": False,
            "frozen_experiment_tree_before": input_state["frozen_experiment_tree"],
            "frozen_experiment_tree_after": frozen_experiment_tree_hash(REPOSITORY),
            "m3_spec_sha256_after": sha256_file(REPOSITORY / config["m3_spec"]["path"]),
            "task018_audit_sha256_after": sha256_file(
                REPOSITORY / config["task018_audit"]["manifest_path"]
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np_version(),
            "pandas": pd.__version__,
        },
        "git": _git_state(),
    }


def np_version() -> str:
    import numpy as np

    return np.__version__


def _verify_existing(
    paths: dict[str, Path], expected_frame: pd.DataFrame, expected_json: dict[str, Any]
) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if not existing:
        return
    if len(existing) != len(paths):
        raise MnmsSplitFailure(
            "partial_frozen_artifact_set",
            {"existing": [str(path) for path in existing], "expected": [str(path) for path in paths.values()]},
        )
    observed_frame = pd.read_parquet(paths["parquet_manifest"])
    if not scientifically_identical(observed_frame, expected_frame):
        raise MnmsSplitFailure("frozen_parquet_regeneration_mismatch", None)
    observed_json = json.loads(paths["json_manifest"].read_text(encoding="utf-8"))
    if observed_json != expected_json:
        raise MnmsSplitFailure("frozen_json_regeneration_mismatch", None)
    report = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    if report.get("artifacts", {}).get("parquet_manifest", {}).get("sha256") != sha256_file(
        paths["parquet_manifest"]
    ):
        raise MnmsSplitFailure("frozen_report_manifest_hash_mismatch", None)
    raise MnmsSplitFailure("frozen_artifacts_already_exist_and_verify", "no files were overwritten")


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
        input_state = verify_inputs(config)
        audit = load_audit_metadata(REPOSITORY / config["task018_audit"]["manifest_path"])
        manifest = generate_split_manifest(audit, config)
        regenerated = generate_split_manifest(audit, config)
        if not scientifically_identical(manifest, regenerated):
            raise MnmsSplitFailure("nondeterministic_regeneration", None)
        json_manifest = build_json_manifest(manifest, config)
        paths = {name: REPOSITORY / relative for name, relative in config["outputs"].items()}
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            _verify_existing(paths, manifest, json_manifest)

        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        parquet_temporary = paths["parquet_manifest"].with_suffix(".parquet.tmp")
        manifest.to_parquet(parquet_temporary, index=False, engine="pyarrow")
        parquet_temporary.replace(paths["parquet_manifest"])
        _write_json(paths["json_manifest"], json_manifest)

        report = build_report(
            manifest, config, started_at=started_at, input_state=input_state
        )
        report["artifacts"] = {
            "parquet_manifest": {
                "path": config["outputs"]["parquet_manifest"],
                "sha256": sha256_file(paths["parquet_manifest"]),
            },
            "json_manifest": {
                "path": config["outputs"]["json_manifest"],
                "sha256": sha256_file(paths["json_manifest"]),
            },
        }
        markdown = render_markdown(report)
        _write_text(paths["report_markdown"], markdown)
        report["artifacts"]["report_markdown"] = {
            "path": config["outputs"]["report_markdown"],
            "sha256": sha256_file(paths["report_markdown"]),
        }
        _write_json(paths["report_json"], report)
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "parquet_manifest_sha256": sha256_file(paths["parquet_manifest"]),
                    "json_manifest_sha256": sha256_file(paths["json_manifest"]),
                    "report_json_sha256": sha256_file(paths["report_json"]),
                    "report_markdown_sha256": sha256_file(paths["report_markdown"]),
                },
                indent=2,
            )
        )
        return 0
    except MnmsSplitFailure as error:
        if error.code == "frozen_artifacts_already_exist_and_verify" and args.verify_existing:
            print(json.dumps({"status": "PASS", "mode": "verify-existing"}, indent=2))
            return 0
        print(
            json.dumps(
                {"status": "FAIL", "failure": {"code": error.code, "details": error.details}},
                indent=2,
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
