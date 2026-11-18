#!/usr/bin/env python3
"""TASK-052 / EXP-0007: synthetic finite-sample study of the frozen R5 / R8 estimators.

Binding text: docs/EXP0007_R8_SYNTHETIC_PROTOCOL.md (tag pre-exp0007-synthetic-freeze).
Synthetic data only: this script reads the config and nothing else. Outputs are
written once and never overwritten.

    python scripts/task052_simulate.py reference | tier1 | tier2 | all
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"  # one BLAS thread per worker; parallelism is across datasets

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shiftqc.confirmatory.r8 import compute_r8  # noqa: E402
from shiftqc.exploratory.synthetic_r8 import (  # noqa: E402
    frozen_folds,
    patient_ids,
    point_estimates,
    simulate_domain,
)

CONFIG_PATH = ROOT / "configs/experiments/exp_0007_r8_synthetic.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUT = ROOT / CONFIG["output_dir"]
TIER_KEYS = {"reference": 0, "tier1": 1, "tier2": 2}


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / name
    if target.exists():
        raise SystemExit(f"{name} already exists; refusing to overwrite")
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def rng_for(tier: str, *key: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(CONFIG["root_seed"], spawn_key=(TIER_KEYS[tier], *key)))


def draw(cell: dict, n_id: int, n_ood: int, rng: np.random.Generator):
    id_q_hat, id_q = simulate_domain(n_id, cell["rho"], rng, scenario=cell["scenario"],
                                     out_of_domain=False)
    ood_q_hat, ood_q = simulate_domain(n_ood, cell["rho"], rng, scenario=cell["scenario"],
                                       out_of_domain=True)
    return id_q_hat, id_q, ood_q_hat, ood_q


def _point_block(task: tuple) -> list[dict[str, float]]:
    tier, cell_index, pair_index, block_index, n_id, n_ood, count = task
    cell = CONFIG["cells"][cell_index] if tier != "tier2" else CONFIG["tier2"]["cells"][cell_index]
    rng = rng_for(tier, cell_index, pair_index, block_index)
    id_folds, ood_folds = frozen_folds(n_id, "i"), frozen_folds(n_ood, "o")
    return [point_estimates(*draw(cell, n_id, n_ood, rng), id_folds, ood_folds)
            for _ in range(count)]


def _tier2_block(task: tuple) -> list[dict[str, float]]:
    _tier, cell_index, pair_index, block_index, n_id, n_ood, count = task
    settings = CONFIG["tier2"]
    cell = settings["cells"][cell_index]
    rng = rng_for("tier2", cell_index, pair_index, block_index)
    id_ids, ood_ids = patient_ids(n_id, "i"), patient_ids(n_ood, "o")
    rows = []
    for _ in range(count):
        id_q_hat, id_q, ood_q_hat, ood_q = draw(cell, n_id, n_ood, rng)
        result = compute_r8(id_ids, id_q_hat, id_ids, id_q, ood_ids, ood_q_hat, ood_ids, ood_q,
                            recalibrator=settings["recalibrator"],
                            bootstrap_B=settings["bootstrap_B"])
        rows.append({"Delta8": float(result["Delta8"]), "CI_low": float(result["CI_low"]),
                     "CI_high": float(result["CI_high"])})
    return rows


def _summary(values: np.ndarray) -> dict[str, float]:
    low, median, high = np.percentile(values, [2.5, 50.0, 97.5])
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
            "p2.5": float(low), "p50": float(median), "p97.5": float(high),
            "n": int(values.size)}


def _label(cell: dict) -> str:
    return f"{cell['scenario']}_rho{cell['rho']}"


def run_reference(pool: Pool) -> None:
    settings = CONFIG["reference"]
    n = settings["n_per_domain"]
    tasks = [("reference", c, 0, d, n, n, 1)
             for c in range(len(CONFIG["cells"])) for d in range(settings["draws"])]
    results = pool.map(_point_block, tasks, chunksize=1)
    cells = {}
    for index, cell in enumerate(CONFIG["cells"]):
        rows = [results[i][0] for i, task in enumerate(tasks) if task[1] == index]
        cells[_label(cell)] = {key: _summary(np.array([row[key] for row in rows]))
                               for key in rows[0]}
    write_new("01_reference.json", {"task_id": "TASK-052", "experiment_id": "EXP-0007",
                                    "created_at": now(), "n_per_domain": n,
                                    "draws": settings["draws"], "cells": cells})
    print("reference done", flush=True)


def run_tier1(pool: Pool) -> None:
    reference = json.loads((OUT / "01_reference.json").read_text(encoding="utf-8"))["cells"]
    settings = CONFIG["tier1"]
    blocks = settings["datasets_per_cell"] // settings["block"]
    tasks = [("tier1", c, p, b, n_id, n_ood, settings["block"])
             for c in range(len(CONFIG["cells"]))
             for p, (n_id, n_ood) in enumerate(CONFIG["sample_size_pairs"])
             for b in range(blocks)]
    results = pool.map(_point_block, tasks, chunksize=1)
    cells: dict[str, dict] = {}
    for c, cell in enumerate(CONFIG["cells"]):
        for p, (n_id, n_ood) in enumerate(CONFIG["sample_size_pairs"]):
            rows = [row for task, block in zip(tasks, results, strict=True)
                    if task[1] == c and task[2] == p for row in block]
            entry: dict[str, object] = {"scenario": cell["scenario"], "rho": cell["rho"],
                                        "n_id": n_id, "n_ood": n_ood}
            for key in rows[0]:
                stats = _summary(np.array([row[key] for row in rows]))
                ref = reference[_label(cell)][key]
                stats["reference"] = ref["mean"]
                stats["mean_minus_reference"] = stats["mean"] - ref["mean"]
                stats["monte_carlo_se"] = float(np.sqrt(
                    stats["sd"] ** 2 / stats["n"] + ref["sd"] ** 2 / ref["n"]))
                entry[key] = stats
            cells[f"{_label(cell)}_n{n_id}_{n_ood}"] = entry
    write_new("02_tier1.json", {"task_id": "TASK-052", "experiment_id": "EXP-0007",
                                "created_at": now(), "datasets_per_cell":
                                settings["datasets_per_cell"], "cells": cells})
    print("tier1 done", flush=True)


def run_tier2(pool: Pool) -> None:
    reference = json.loads((OUT / "01_reference.json").read_text(encoding="utf-8"))["cells"]
    settings = CONFIG["tier2"]
    block = 5
    blocks = settings["datasets_per_cell"] // block
    tasks = [("tier2", c, p, b, n_id, n_ood, block)
             for c in range(len(settings["cells"]))
             for p, (n_id, n_ood) in enumerate(settings["sample_size_pairs"])
             for b in range(blocks)]
    results = pool.map(_tier2_block, tasks, chunksize=1)

    def rate(flags: np.ndarray) -> dict[str, float]:
        p = float(np.mean(flags))
        return {"frequency": p, "binomial_se": float(np.sqrt(p * (1.0 - p) / flags.size))}

    cells: dict[str, dict] = {}
    for c, cell in enumerate(settings["cells"]):
        ref = reference[_label(cell)][f"R8.{settings['recalibrator']}.Delta"]["mean"]
        for p, (n_id, n_ood) in enumerate(settings["sample_size_pairs"]):
            rows = [row for task, group in zip(tasks, results, strict=True)
                    if task[1] == c and task[2] == p for row in group]
            low = np.array([row["CI_low"] for row in rows])
            high = np.array([row["CI_high"] for row in rows])
            cells[f"{_label(cell)}_n{n_id}_{n_ood}"] = {
                "scenario": cell["scenario"], "rho": cell["rho"], "n_id": n_id, "n_ood": n_ood,
                "datasets": len(rows), "reference_Delta8": ref,
                "Delta8": _summary(np.array([row["Delta8"] for row in rows])),
                "CI_high_below_zero": rate(high < 0.0),
                "CI_low_above_zero": rate(low > 0.0),
                "interval_covers_reference": rate((low <= ref) & (ref <= high)),
                "interval_covers_zero": rate((low <= 0.0) & (0.0 <= high)),
                "mean_interval_width": float(np.mean(high - low)),
            }
    write_new("03_tier2.json", {
        "task_id": "TASK-052", "experiment_id": "EXP-0007", "created_at": now(),
        "bootstrap_B": settings["bootstrap_B"],
        "registered_bootstrap_B": settings["registered_bootstrap_B"],
        "recalibrator": settings["recalibrator"], "cells": cells})
    print("tier2 done", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("reference", "tier1", "tier2", "all"))
    args = parser.parse_args()
    started = now()
    steps = ("reference", "tier1", "tier2") if args.command == "all" else (args.command,)
    with Pool(CONFIG["workers"]) as pool:
        for step in steps:
            {"reference": run_reference, "tier1": run_tier1, "tier2": run_tier2}[step](pool)
    if args.command == "all":
        import scipy
        import sklearn
        git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()  # noqa: E731
        write_new("04_run_record.json", {
            "task_id": "TASK-052", "experiment_id": "EXP-0007", "started_at": started,
            "ended_at": now(), "git_commit": git("rev-parse", "HEAD"),
            "tracked_files_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "config_sha256": sha256_file(CONFIG_PATH), "root_seed": CONFIG["root_seed"],
            "workers": CONFIG["workers"], "patient_data_read": False, "gpu_used": False,
            "code_sha256": {p: sha256_file(ROOT / p) for p in (
                "scripts/task052_simulate.py", "src/shiftqc/exploratory/synthetic_r8.py",
                "src/shiftqc/confirmatory/r5.py", "src/shiftqc/confirmatory/r8.py",
                "src/shiftqc/confirmatory/recalibration.py", "src/shiftqc/confirmatory/skill.py",
                "src/shiftqc/confirmatory/bootstrap.py")},
            "software": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "sklearn": sklearn.__version__},
            "output_sha256": {n: sha256_file(OUT / n) for n in (
                "01_reference.json", "02_tier1.json", "03_tier2.json")},
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
