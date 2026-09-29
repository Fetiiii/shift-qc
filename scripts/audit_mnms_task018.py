#!/usr/bin/env python3
"""Run the frozen read-only TASK-018 M&Ms dataset audit."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from shiftqc.data.mnms_audit import (
    MnmsAuditFailure,
    audit_dataset,
    build_audit_report,
    dataset_tree_inventory,
    frozen_experiment_tree_hash,
    load_config,
    render_markdown_report,
)
from shiftqc.data.nnunet_export import sha256_file


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY / "configs/datasets/mnms_task018.json"


def now() -> str:
    return datetime.now().astimezone().isoformat()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dataset-root", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started_at = now()
    config = load_config(args.config)
    dataset_root = args.dataset_root
    if dataset_root is None:
        configured = os.environ.get(config["dataset_root_env"])
        dataset_root = Path(configured) if configured else None
    if dataset_root is None or not dataset_root.is_dir():
        print(
            json.dumps(
                {
                    "task": config["task"],
                    "status": "BLOCKED — DATASET NOT AVAILABLE LOCALLY",
                    "required_environment_variable": config["dataset_root_env"],
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    manifest_path = REPOSITORY / config["manifest_output"]
    report_json_path = REPOSITORY / config["report_json_output"]
    report_markdown_path = REPOSITORY / config["report_markdown_output"]
    targets = (manifest_path, report_json_path, report_markdown_path)
    existing = [path for path in targets if path.exists()]
    if existing:
        raise MnmsAuditFailure(f"refusing to overwrite TASK-018 artifacts: {existing}")
    if sha256_file(REPOSITORY / config["m3_spec"]["path"]) != config["m3_spec"]["sha256"]:
        raise MnmsAuditFailure("frozen docs/M3_SPEC.md hash mismatch")

    dataset_before = dataset_tree_inventory(dataset_root)
    frozen_before = frozen_experiment_tree_hash(REPOSITORY)

    def progress(index: int, total: int, patient_id: str) -> None:
        if index == 1 or index % 25 == 0 or index == total:
            print(
                f"TASK-018 progress {index}/{total} patient={patient_id}",
                flush=True,
            )

    frame = audit_dataset(dataset_root, config, progress=progress)
    dataset_after = dataset_tree_inventory(dataset_root)
    frozen_after = frozen_experiment_tree_hash(REPOSITORY)
    report = build_audit_report(
        frame,
        config,
        dataset_root=dataset_root,
        dataset_inventory_before=dataset_before,
        dataset_inventory_after=dataset_after,
        frozen_tree_before=frozen_before,
        frozen_tree_after=frozen_after,
    )
    report.update(
        {
            "started_at": started_at,
            "ended_at": now(),
            "m3_spec": {
                "path": config["m3_spec"]["path"],
                "sha256": config["m3_spec"]["sha256"],
                "verified": True,
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": _package_version("numpy"),
                "pandas": _package_version("pandas"),
                "nibabel": _package_version("nibabel"),
                "pyarrow": _package_version("pyarrow"),
            },
            "git": _git_state(),
        }
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    frame.to_parquet(manifest_temporary, index=False, engine="pyarrow")
    manifest_temporary.replace(manifest_path)
    markdown = render_markdown_report(report)
    _atomic_text(report_markdown_path, markdown)
    report["artifacts"] = {
        "manifest": {
            "path": config["manifest_output"],
            "sha256": sha256_file(manifest_path),
        },
        "report_markdown": {
            "path": config["report_markdown_output"],
            "sha256": sha256_file(report_markdown_path),
        },
    }
    _atomic_json(report_json_path, report)
    print(
        json.dumps(
            {
                "decision": report["m3_compatibility_decision"],
                "manifest_sha256": sha256_file(manifest_path),
                "report_json_sha256": sha256_file(report_json_path),
                "report_markdown_sha256": sha256_file(report_markdown_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
