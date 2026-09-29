import importlib.util
import json
import hashlib
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
def verifier():
    spec=importlib.util.spec_from_file_location("candidate_verifier",ROOT/"scripts/verify_release.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def mock_inventory(tmp_path,monkeypatch):
    module=verifier();payload=b"review fixture"
    (tmp_path/"fixture.txt").write_bytes(payload)
    doc={"source_commit":"a"*40,"files":[{"path":"fixture.txt","size":len(payload),
         "sha256":hashlib.new("sha256",payload).hexdigest(),"source_commit":"a"*40,
         "contains_patient_data":False,"redistribution_status":"PUBLIC_OK"}]}
    mp=tmp_path/"RELEASE_MANIFEST.json";mp.write_text(json.dumps(doc))
    monkeypatch.setattr(module,"ROOT",tmp_path);monkeypatch.setattr(module,"MANIFEST",mp)
    return module,doc,mp
def test_inventory_rejects_hash_tampering(tmp_path,monkeypatch):
    module,doc,mp=mock_inventory(tmp_path,monkeypatch)
    assert module.verify_inventory()==1
    (tmp_path/"fixture.txt").write_bytes(b"changed bytes!")
    with pytest.raises(SystemExit,match="Hash mismatch"):module.verify_inventory()
def test_inventory_rejects_unlisted_file(tmp_path,monkeypatch):
    module,doc,mp=mock_inventory(tmp_path,monkeypatch)
    (tmp_path/"unexpected.txt").write_text("not exported")
    with pytest.raises(SystemExit,match="Inventory mismatch"):module.verify_inventory()
def test_inventory_rejects_traversal(tmp_path,monkeypatch):
    module,doc,mp=mock_inventory(tmp_path,monkeypatch)
    doc["files"][0]["path"]="../fixture.txt";mp.write_text(json.dumps(doc))
    with pytest.raises(SystemExit,match="Unsafe destination"):module.verify_inventory()
def test_explicit_author_orcid_and_pending_release_metadata():
    text=(ROOT/"CITATION.cff").read_text()
    assert "https://orcid.org/0009-0001-9358-0563" in text
    assert 'version: "1.0.0"' in text
    assert "doi:" not in text and "date-released:" not in text and "repository-code:" not in text
    digits="0009000193580563";total=0
    for digit in digits[:-1]:total=(total+int(digit))*2
    assert str((12-total%11)%11)==digits[-1]
