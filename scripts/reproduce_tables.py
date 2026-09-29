from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
LEDGER = {}
MISSING = "Required source-data-derived input not included in this repository. See docs/data_access.md and docs/full_reproduction.md."

def d(result_id):
    if result_id not in LEDGER: raise ValueError(f"Missing result row {result_id}. {MISSING}")
    value = LEDGER[result_id].get("display")
    if not isinstance(value, str) or not value: raise ValueError(f"Invalid display for {result_id}")
    return value

def ci(low, high):
    return f"{d(low)} to {d(high)}"

def status(result_id):
    d(result_id)
    value = LEDGER[result_id].get("status")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing local endpoint status for {result_id}; regenerate from the frozen endpoint output.")
    return value

def source_total():
    keys = ("V2-ELIG-MSD-OK", "V2-ELIG-NIH-OK")
    total = 0
    for key in keys:
        d(key)
        value = float(LEDGER[key]["full_precision"])
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError(f"Invalid local cohort count for {key}")
        total += int(value)
    return str(total)

def table1() -> str:
    return f"""
**Table 1.** Datasets, populations and shift structure. Abdominal images were
obtained through AbdomenCT-1K and matched back to their original releases; the
eligibility contract is outcome-independent and is detailed in Supplementary
Section S5. The reserved AbdomenCT calibration population was consumed by no
registered endpoint.

| | M&Ms | AbdomenCT |
|---|---|---|
| Modality | Cardiac MRI | Abdominal CT |
| Foreground classes | LV, myocardium, RV | liver, kidney, spleen, pancreas |
| Target quality | macro Dice, ED and ES averaged per patient | macro Dice over four organs |
| Acquisition path | original release | via AbdomenCT-1K, source identity recovered |
| Source cohort | Vendors A and B | MSD Pancreas {d('V2-ELIG-MSD-OK')}/{d('V2-ELIG-MSD-CAND')} + NIH Pancreas-CT {d('V2-ELIG-NIH-OK')}/{d('V2-ELIG-NIH-CAND')} = {source_total()} |
| Shift | vendor and acquisition | cross-cohort (KiTS19 {d('V2-ELIG-KITS-OK')}/{d('V2-ELIG-KITS-CAND')}), with kidney pathology |
| Segmentation | nnU-Net v2, `3d_fullres`, 5-fold | nnU-Net v2.8.1, `nnUNetTrainer_250epochs`, `2d`, 5-fold |
| Segmentation development n | — | {d('V2-POP-SEG_DEV')} |
| QC training n | {d('MM-QC-NTRAIN')} | {d('V2-POP-QC_TRAIN')} |
| Reserved calibration n | — | {d('V2-POP-CALIBRATION')} (unused) |
| In-domain evaluation n | {d('MM-POP-ID')} (A {d('MM-VENDOR-A')}, B {d('MM-VENDOR-B')}) | {d('V2-POP-ID_EVAL')} |
| Out-of-domain evaluation n | {d('MM-POP-OOD')} (C {d('MM-VENDOR-C')}, D {d('MM-VENDOR-D')}) | {d('V2-POP-OOD_EVAL')} |
| Label-free features | {d('MM-QC-NFEAT')} | {d('V2-QC-NFEAT')} |
| Selected ridge alpha | {d('MM-QC-ALPHA')} | {d('V2-QC-ALPHA')} |
| Prospective completeness | pre-specified; vendor aggregation partially closed | frozen pre-training; hashed authority manifest |
"""

def table2() -> str:
    return f"""
**Table 2.** Registered endpoints in both validations, with their place in the
frozen hierarchy. Every interval is a two-sided 95% patient-level bootstrap
percentile interval. The decision system defines POSITIVE and UNRESOLVED only.

| Endpoint | Role | Dataset | S_ID | S_OOD | Estimate | 95% CI | Verdict |
|---|---|---|---|---|---|---|---|
| Baseline competence (rho) | prerequisite | M&Ms | — | — | {d('MM-B-RHO')} | {ci('MM-B-LO', 'MM-B-HI')} | {status('MM-B-RHO')} (n = {d('MM-B-N')}) |
| Baseline competence (rho) | prerequisite | AbdomenCT | — | — | {d('V2-B-RHO')} | {ci('V2-B-LO', 'V2-B-HI')} | {status('V2-B-RHO')} (n = {d('V2-B-N')}) |
| R5 deployment-state (Delta5) | key secondary | M&Ms | {d('MM-R5-SID')} | {d('MM-R5-SOOD')} | {d('MM-R5-D')} | {ci('MM-R5-LO', 'MM-R5-HI')} | {status('MM-R5-D')} |
| R5 deployment-state (Delta5) | key secondary | AbdomenCT | {d('V2-R5-SID')} | {d('V2-R5-SOOD')} | {d('V2-R5-D')} | {ci('V2-R5-LO', 'V2-R5-HI')} | {status('V2-R5-D')} |
| R8 recoverable, affine-L1 (Delta8) | **primary** | M&Ms | {d('MM-R8-S_ID')} | {d('MM-R8-S_OOD')} | {d('MM-R8-Delta8')} | {ci('MM-R8-CI_low', 'MM-R8-CI_high')} | {status('MM-R8-Delta8')} |
| R8 recoverable, affine-L1 (Delta8) | **primary** | AbdomenCT | {d('V2-R8-S_ID')} | {d('V2-R8-S_OOD')} | {d('V2-R8-Delta8')} | {ci('V2-R8-CI_low', 'V2-R8-CI_high')} | {status('V2-R8-Delta8')} |
| R8 affine-OLS (Delta8) | sensitivity | M&Ms | — | — | {d('MM-OLS-Delta8')} | {ci('MM-OLS-CI_low', 'MM-OLS-CI_high')} | — |
| R8 affine-OLS (Delta8) | sensitivity | AbdomenCT | — | — | {d('V2-OLS-Delta8')} | {ci('V2-OLS-CI_low', 'V2-OLS-CI_high')} | — |
| R8 isotonic (Delta8) | sensitivity | M&Ms | — | — | {d('MM-ISO-Delta8')} | {ci('MM-ISO-CI_low', 'MM-ISO-CI_high')} | — |
| R8 isotonic (Delta8) | sensitivity | AbdomenCT | — | — | {d('V2-ISO-Delta8')} | {ci('V2-ISO-CI_low', 'V2-ISO-CI_high')} | — |

Sensitivities carry no verdict: they were registered to probe the primary
analysis and cannot alter it.
"""

def ledger_table():
    # Generic machine-ledger view; not the final manuscript narrative Table S1.
    lines = ["# Locally regenerated numerical ledger", "", "| Result ID | Dataset | Endpoint | Quantity | Display |", "|---|---|---|---|---|"]
    for key, row in LEDGER.items():
        cells = [key, row.get("dataset", ""), row.get("endpoint", ""), row.get("quantity", ""), d(key)]
        lines.append("| " + " | ".join(str(v).replace("|", chr(92)+"|").replace(chr(10), " ") for v in cells) + " |")
    return "\n".join(lines) + "\n"

def main():
    parser = argparse.ArgumentParser(description="Render source-data-dependent tables from a locally regenerated frozen analysis ledger.")
    parser.add_argument("--ledger", type=Path, default=Path("local_inputs/frozen_result_ledger.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("local_outputs/tables"))
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    path = args.ledger if args.ledger.is_absolute() else ROOT / args.ledger
    try:
        if not path.is_file(): raise ValueError(MISSING)
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows: raise ValueError("Expected nonempty local result-row list.")
        global LEDGER
        LEDGER = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("result_id"), str): raise ValueError("Invalid local result row.")
            key = row["result_id"]
            if key in LEDGER: raise ValueError(f"Duplicate result ID {key}")
            number = float(row["full_precision"])
            if not math.isfinite(number): raise ValueError(f"Nonfinite result value {key}")
            LEDGER[key] = row
        outputs = {"table1.md": table1(), "table2.md": table2(), "ledger_table.md": ledger_table()}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"SOURCE_DATA_REQUIRED: {exc}", file=sys.stderr)
        return 2
    if args.status:
        print("Locally regenerated input contract verified; manuscript authority equality is a separate audit.")
        return 0
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, body in outputs.items(): (out / name).write_text(body, encoding="utf-8")
    print("Rendered Table 1/2 layout and numerical ledger view; no endpoint was recomputed.")
    return 0
if __name__ == "__main__": raise SystemExit(main())
