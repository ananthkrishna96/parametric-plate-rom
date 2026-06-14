from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _script_path() -> Path:
    script = Path("scripts/project2/report_rom_suite.py")
    if script.exists():
        return script
    return Path(__file__).resolve().parents[1] / "scripts" / "project2" / "report_rom_suite.py"


def _write_flat_summaries(base: Path, case: str, production_layout: bool) -> tuple[Path, Path]:
    suite_root = base / "suites"
    if production_layout:
        intrusive = suite_root / "intrusive" / case / "pod_projected"
        nonintrusive = suite_root / "nonintrusive" / case / "pod_gpr"
    else:
        case_root = suite_root / case
        intrusive = case_root / "intrusive" / "pod_projected"
        nonintrusive = case_root / "nonintrusive" / "pod_gpr"
    intrusive.mkdir(parents=True)
    nonintrusive.mkdir(parents=True)

    (intrusive / "intrusive_projection_summary.json").write_text(json.dumps({
        "rows": [
            {"suite": "intrusive", "method": "pod-projected", "field": "w", "rank": 3, "test_error": 0.002, "train_error": 0.003},
        ]
    }))
    (nonintrusive / "nonintrusive_summary.json").write_text(json.dumps({
        "rows": [
            {"suite": "nonintrusive", "method": "pod-gpr", "field": "w", "rank": 3, "coeff_test_error": 0.001, "field_test_error": 0.0025},
        ]
    }))

    pod_dir = base / "pod" / case
    pod_dir.mkdir(parents=True)
    return suite_root, pod_dir


def _write_nested_project2_summaries(base: Path, case: str) -> tuple[Path, Path]:
    suite_root = base / "suites"
    intrusive = suite_root / "intrusive" / case / "pod_projected"
    nonintrusive = suite_root / "nonintrusive" / case / "pod_gpr"
    intrusive.mkdir(parents=True)
    nonintrusive.mkdir(parents=True)

    (intrusive / "intrusive_projection_summary.json").write_text(json.dumps({
        "suite": "intrusive",
        "method": "pod-projected",
        "case_name": case,
        "w": {
            "field_name": "w",
            "rank": 3,
            "n_train": 120,
            "n_test": 30,
            "relative_error_train": 0.0026,
            "relative_error_test": 0.0023,
        },
        "theta": {
            "field_name": "theta",
            "rank": 1,
            "n_train": 120,
            "n_test": 30,
            "relative_error_train": 0.0012,
            "relative_error_test": 0.00095,
        },
    }))
    (nonintrusive / "nonintrusive_summary.json").write_text(json.dumps({
        "suite": "nonintrusive",
        "method": "pod-gpr",
        "case_name": case,
        "w": {
            "field_name": "w",
            "method": "pod-gpr",
            "rank": 3,
            "coefficient_relative_error_train": 0.001,
            "coefficient_relative_error_test": 0.002,
            "field_relative_error_train": 0.0024,
            "field_relative_error_test": 0.0025,
        },
        "theta": {
            "field_name": "theta",
            "method": "pod-gpr",
            "rank": 1,
            "coefficient_relative_error_train": 0.00001,
            "coefficient_relative_error_test": 0.00002,
            "field_relative_error_train": 0.00094,
            "field_relative_error_test": 0.000953,
        },
    }))

    pod_dir = base / "pod" / case
    pod_dir.mkdir(parents=True)
    return suite_root, pod_dir


def _run_report(tmp_path: Path, suite_root: Path, pod_dir: Path, case: str):
    out_dir = tmp_path / "report"
    result = subprocess.run([
        sys.executable,
        str(_script_path()),
        "--case", case,
        "--suite-root", str(suite_root),
        "--pod-dir", str(pod_dir),
        "--output-dir", str(out_dir),
        "--no-plots",
    ], check=True, text=True, capture_output=True)
    return result, out_dir


def test_report_rom_suite_collects_production_layout_flat_rows(tmp_path: Path):
    case = "CASE_TEST"
    suite_root, pod_dir = _write_flat_summaries(tmp_path, case, production_layout=True)
    result, out_dir = _run_report(tmp_path, suite_root, pod_dir, case)
    assert "collected rows: 2" in result.stdout
    assert (out_dir / "rom_suite_comparison.csv").exists()
    assert (out_dir / "rom_suite_comparison.md").exists()
    assert (out_dir / "rom_suite_comparison_table.tex").exists()
    assert (out_dir / "rom_suite_report.json").exists()
    text = (out_dir / "rom_suite_comparison.csv").read_text()
    assert "pod-projected" in text
    assert "pod-gpr" in text


def test_report_rom_suite_collects_legacy_flat_layout_summaries(tmp_path: Path):
    case = "CASE_TEST"
    suite_root, pod_dir = _write_flat_summaries(tmp_path, case, production_layout=False)
    result, out_dir = _run_report(tmp_path, suite_root, pod_dir, case)
    assert "collected rows: 2" in result.stdout
    text = (out_dir / "rom_suite_comparison.csv").read_text()
    assert "pod-projected" in text
    assert "pod-gpr" in text


def test_report_rom_suite_collects_actual_nested_project2_method_summaries(tmp_path: Path):
    case = "CASE_TEST"
    suite_root, pod_dir = _write_nested_project2_summaries(tmp_path, case)
    result, out_dir = _run_report(tmp_path, suite_root, pod_dir, case)
    assert "collected rows: 4" in result.stdout
    text = (out_dir / "rom_suite_comparison.csv").read_text()
    assert "pod-projected" in text
    assert "pod-gpr" in text
    assert "theta" in text
    assert "0.0023" in text
    assert "0.000953" in text
