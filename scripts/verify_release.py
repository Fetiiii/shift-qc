from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/"RELEASE_MANIFEST.json"
IGNORED_DIRS={".git",".pytest_cache","__pycache__",".venv","local_inputs","local_outputs"}
def verify_inventory():
    document=json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows=document.get("files")
    if not isinstance(rows,list) or not rows: raise SystemExit("Release manifest has no file inventory.")
    expected=set()
    for row in rows:
        rel=row.get("path")
        if not isinstance(rel,str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise SystemExit(f"Unsafe destination: {rel!r}")
        if rel=="RELEASE_MANIFEST.json" or rel in expected: raise SystemExit(f"Duplicate or self-referencing destination: {rel}")
        expected.add(rel);path=ROOT/rel
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()): raise SystemExit(f"Missing, symlink or unsafe file: {rel}")
        if path.stat().st_size!=row.get("size"): raise SystemExit(f"Size mismatch: {rel}")
        if not re.fullmatch(r"[0-9a-f]{64}",str(row.get("sha256",""))): raise SystemExit(f"Invalid SHA-256: {rel}")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row["sha256"]: raise SystemExit(f"Hash mismatch: {rel}")
        if row.get("source_commit")!=document.get("source_commit"): raise SystemExit(f"Source commit mismatch: {rel}")
        if row.get("contains_patient_data") is not False: raise SystemExit(f"Unexpected patient data: {rel}")
        if row.get("redistribution_status") != "PUBLIC_OK": raise SystemExit(f"Unexpected distribution classification: {rel}")
    actual=set()
    for path in ROOT.rglob("*"):
        parts=path.relative_to(ROOT).parts
        if any(p in IGNORED_DIRS or p.endswith(".egg-info") for p in parts): continue
        if path.is_file() and path.relative_to(ROOT).as_posix()!="RELEASE_MANIFEST.json": actual.add(path.relative_to(ROOT).as_posix())
    if actual!=expected: raise SystemExit(f"Inventory mismatch: missing={sorted(expected-actual)}; unexpected={sorted(actual-expected)}")
    return len(expected)
def main():
    parser=argparse.ArgumentParser(description="Verify software/public-safe artifacts; no patient-derived number verification.")
    parser.add_argument("--code-only",action="store_true")
    args=parser.parse_args();count=verify_inventory()
    print(f"CODE_REPRODUCIBILITY: integrity PASS ({count} embedded files).")
    print("PUBLIC_SAFE_ARTIFACT_VERIFICATION: PASS.")
    print("FULL_SOURCE_DATA_REPRODUCTION: DOCUMENTED / REQUIRES ORIGINAL DATA (not tested PASS).")
    print("Publication still requires separate author approval.")
    return 0
if __name__=="__main__": raise SystemExit(main())
