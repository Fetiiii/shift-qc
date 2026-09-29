"""Exercise release input gates with synthetic ledger rows only."""
import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]

def table_module():
    path=ROOT/"scripts/reproduce_tables.py"
    spec=importlib.util.spec_from_file_location("synthetic_table_renderer",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,path

def test_tables_render_synthetic_rows_without_result_literal_fallback():
    module,path=table_module()
    keys=set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in {"d","ci","status"}:
            for arg in node.args:
                if isinstance(arg,ast.Constant) and isinstance(arg.value,str):keys.add(arg.value)
    keys.update(("V2-ELIG-MSD-OK","V2-ELIG-NIH-OK"))
    module.LEDGER={key:{"result_id":key,"display":"1","full_precision":1,"status":"synthetic-status"} for key in keys}
    first=module.table1();second=module.table2()
    assert " = 2 |" in first
    assert "synthetic-status" in second
    del module.LEDGER["MM-R5-D"]
    with pytest.raises(ValueError,match="Missing result row"): module.table2()

@pytest.mark.parametrize("script,args",[("reproduce_tables.py",[]),("reproduce_figures.py",["--only","figure2"]),("reproduce_figures.py",["--only","figure3","figure4"])])
def test_absent_data_is_controlled_error_and_no_output(tmp_path,script,args):
    result=subprocess.run([sys.executable,str(ROOT/"scripts"/script),*args,"--ledger",str(tmp_path/"missing.json"),"--output-dir",str(tmp_path/"outputs")],capture_output=True,text=True)
    assert result.returncode==2
    assert "Required source-data-derived input not included" in result.stderr
    assert "docs/data_access.md" in result.stderr and "docs/full_reproduction.md" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path/"outputs").exists()

def test_software_license_and_data_boundary_are_explicit():
    text=(ROOT/"LICENSE").read_text()
    assert "MIT License" in text and "excludes source" in text
    rights=(ROOT/"DATA_AND_ARTIFACT_RIGHTS.md").read_text()
    assert "grants" in rights and "no rights" in rights

@pytest.mark.parametrize("script,args",[("reproduce_tables.py",[]),("reproduce_figures.py",["--only","figure2"])])
def test_malformed_local_input_is_controlled_error(tmp_path,script,args):
    ledger=tmp_path/"invalid.json";ledger.write_text("not json")
    result=subprocess.run([sys.executable,str(ROOT/"scripts"/script),*args,"--ledger",str(ledger),"--output-dir",str(tmp_path/"outputs")],capture_output=True,text=True)
    assert result.returncode==2
    assert "Traceback" not in result.stderr and not (tmp_path/"outputs").exists()

def test_figure1_does_not_read_source_data_input(tmp_path):
    ledger=tmp_path/"invalid.json";ledger.write_text("not json")
    result=subprocess.run([sys.executable,str(ROOT/"scripts/reproduce_figures.py"),"--only","figure1","--ledger",str(ledger),"--output-dir",str(tmp_path/"concept")],capture_output=True,text=True)
    assert result.returncode==0
    assert (tmp_path/"concept/figure1_shiftqc_concept.pdf").is_file()
