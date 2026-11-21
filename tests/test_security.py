import tempfile
from pathlib import Path

from cortex.security.analyzer import analyze_source, analyze_tree, summarize


VULN = (
    "import os, pickle, yaml, subprocess\n"
    "def boom(user_input):\n"
    "    return eval(user_input)\n\n"
    "def sh(cmd):\n"
    "    os.system(cmd)\n\n"
    "def unpick(data):\n"
    "    return pickle.loads(data)\n\n"
    "def yml(doc):\n"
    "    return yaml.load(doc)\n\n"
    "def sp(args):\n"
    "    subprocess.run(args, shell=True)\n"
)


def test_vulnerability_detectors_fire():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "vuln.py"
        p.write_text(VULN)
        findings = analyze_source(str(p))
        rules = {f.rule_id for f in findings}
        assert "CORTEX-SEC-001" in rules
        assert "CORTEX-SEC-004" in rules
        assert "CORTEX-SEC-005" in rules
        assert "CORTEX-SEC-007" in rules
        assert "CORTEX-SEC-010" in rules
        s = summarize(findings)
        assert s["critical"] >= 1 and s["high"] >= 3


def test_clean_file_has_no_findings():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "clean.py"
        p.write_text("def add(a, b):\n    return a + b\n")
        assert analyze_source(str(p)) == []


def test_secret_scanning():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "secrets.py"
        p.write_text("API_KEY = 'sk-abcdefghijklmnopqrstuvwx'\n")
        findings = analyze_source(str(p))
        assert any(f.rule_id == "CORTEX-SEC-020" for f in findings)


def test_tree_scan(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "bad.py").write_text("import os\nos.system('ls')\n")
    (tmp_path / "pkg" / "good.py").write_text("x = 1\n")
    findings = analyze_tree(str(tmp_path))
    assert any("bad.py" in f.file for f in findings)
