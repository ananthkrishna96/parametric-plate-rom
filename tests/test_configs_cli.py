from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from paramplate.io.configuration import ConfigurationError, load_config, resolve_repo_path
from paramplate.workflows import mechanical_config, thermomechanical_config, transient_config


ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=ENV,
        text=True,
        capture_output=True,
        check=True,
    )


def test_all_yaml_configs_load_and_case_files_compose() -> None:
    for path in sorted((ROOT / "configs").rglob("*.yml")):
        if path.name.startswith("data_locations"):
            continue
        payload = load_config(path)
        assert payload["mode"] in {"smoke", "thesis", "full"}
    thermo = load_config(ROOT / "configs/thermomechanical/thesis_case2.yml", expected_study="thermomechanical")
    assert thermomechanical_config(thermo).heat_nx == 64
    panel = load_config(ROOT / "configs/dynamics/thesis_case2_one_interface.yml", expected_study="dynamics")
    assert transient_config(panel).mechanical.geometry.n_panels == 2
    mechanical = load_config(ROOT / "configs/mechanical/smoke.yml", expected_study="mechanical")
    assert mechanical_config(mechanical).geometry.resolution == 8


def test_repo_path_rejects_home_shorthand() -> None:
    with pytest.raises(ConfigurationError):
        resolve_repo_path("~/private/data", root=ROOT)


def test_scripts_expose_help_and_dry_run_without_fenics() -> None:
    scripts = [
        "scripts/mechanical/run_workflow.py",
        "scripts/thermomechanical/run_workflow.py",
        "scripts/dynamics/run_workflow.py",
        "scripts/common/run_rom.py",
        "scripts/common/show_capabilities.py",
    ]
    for script in scripts:
        completed = run(script, "--help")
        assert "usage:" in completed.stdout.lower()

    mech = run(
        "scripts/mechanical/run_workflow.py",
        "--config",
        "configs/mechanical/smoke.yml",
        "--dry-run",
    )
    assert json.loads(mech.stdout)["study"] == "mechanical"
    thermo = run(
        "scripts/thermomechanical/run_workflow.py",
        "--config",
        "configs/thermomechanical/thesis_case3.yml",
        "--dry-run",
    )
    thermo_payload = json.loads(thermo.stdout)
    assert thermo_payload["config"]["heat_nx"] == 64
    assert thermo_payload["case"] == 3
    assert thermo_payload["requested_samples"] == 379
    dynamics = run(
        "scripts/dynamics/run_workflow.py",
        "--config",
        "configs/dynamics/thesis_case2_one_interface.yml",
        "--dry-run",
    )
    dynamics_payload = json.loads(dynamics.stdout)
    assert dynamics_payload["config"]["mechanical"]["geometry"]["n_vertical_interfaces"] == 1
    assert dynamics_payload["campaign"] == "case2_one_interface"
    assert dynamics_payload["requested_trajectories"] == 250


def test_dependency_light_examples_and_archive_cli() -> None:
    assert "maximum |w|" in run("examples/static_navier_smoke.py").stdout
    assert "theta =" in run("examples/thermal_driver_smoke.py").stdout
    assert "energy drift" in run("examples/newmark_smoke.py").stdout
    rom = run("examples/rom_smoke.py", "--study", "mechanical", "--method", "pod-proj", "--rank", "4")
    assert "POD--Proj" in rom.stdout
    validation = run("-m", "paramplate.io.archives", "data/sample/dynamics_smoke.npz")
    assert json.loads(validation.stdout)["n_trajectories"] == 20
