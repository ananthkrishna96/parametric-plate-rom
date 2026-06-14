from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_rom_suite_cli_compiles_and_lists_methods():
    script = Path("scripts/project2/run_rom_suite.py")
    subprocess.run([sys.executable, "-m", "py_compile", str(script)], check=True)
    out = subprocess.check_output([sys.executable, str(script), "--list", "--suite", "all"], text=True)
    assert "podi-rbf" in out
    assert "pod-gpr" in out
    assert "pod-nn" in out
    assert "pod-ae" in out


def test_run_rom_suite_cli_accepts_max_iter_option():
    script = Path("scripts/project2/run_rom_suite.py")
    out = subprocess.check_output([sys.executable, str(script), "--help"], text=True)
    assert "--max-iter" in out
