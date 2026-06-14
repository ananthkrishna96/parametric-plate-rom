from pathlib import Path

import numpy as np

from paramplate.io.data_locations import load_data_locations
from paramplate.io.snapshot_archives import detect_archive_kind, summarize_snapshot_archive
from paramplate.io.snapshot_manifest import project2_final_archives, paper1_archives
from paramplate.rom.datasets import load_rom_dataset


def test_detect_and_load_project2_archive(tmp_path):
    p = tmp_path / "SNAPSHOTS__CASE_01_SUMMER_0MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH.npz"
    np.savez(
        p,
        parameters=np.ones((3, 2)),
        snapshots=np.ones((3, 5)),
        theta_snapshots=np.ones((3, 4)),
        mech_parameters=np.ones((3, 6)),
        thermal_parameters=np.ones((3, 8)),
        parameter_names=np.array(["f", "q_s"], dtype=object),
    )

    info = summarize_snapshot_archive(p)
    assert info.project == "paper2"
    assert info.parameter_shape == (3, 2)
    assert info.snapshot_shape == (3, 5)
    assert info.theta_shape == (3, 4)

    ds = load_rom_dataset(p)
    assert ds.project == "paper2"
    assert ds.n_samples == 3
    assert ds.snapshots.shape == (3, 5)
    assert ds.theta_snapshots is not None


def test_detect_and_load_paper1_archive(tmp_path):
    p = tmp_path / "nonlinear_snapshots_1000.npz"
    np.savez(
        p,
        mus=np.arange(12).reshape(4, 3),
        fom_snapshots=np.ones((4, 6)),
    )

    info = summarize_snapshot_archive(p)
    assert info.project == "paper1"
    assert info.parameter_shape == (4, 3)
    assert info.snapshot_shape == (4, 6)

    ds = load_rom_dataset(p)
    assert ds.project == "paper1"
    assert ds.n_samples == 4
    assert ds.theta_snapshots is None


def test_data_location_resolution_with_temp_config(tmp_path):
    data_root = tmp_path / "parametric-plate-rom-data"
    cfg = tmp_path / "data_locations.local.yml"
    cfg.write_text(
        f"""
data_root: "{data_root}"
paper1:
  snapshots_root: "paper1_mechanical/snapshots"
paper2:
  baseline_root: "paper2_thermomechanical/campaigns/baseline"
""".strip()
    )

    loc = load_data_locations(config_path=cfg)
    assert loc.data_root == data_root.resolve()
    assert loc.paper1_snapshots_root == data_root.resolve() / "paper1_mechanical/snapshots"
    assert loc.paper2_baseline_root == data_root.resolve() / "paper2_thermomechanical/campaigns/baseline"


def test_archive_discovery(tmp_path):
    data_root = tmp_path / "data"
    p2_dir = data_root / "paper2_thermomechanical/campaigns/baseline/CASE_01"
    p2_dir.mkdir(parents=True)
    p2_archive = p2_dir / "SNAPSHOTS__CASE_01.npz"
    np.savez(p2_archive, parameters=np.ones((2, 1)), snapshots=np.ones((2, 3)), theta_snapshots=np.ones((2, 2)))

    p1_dir = data_root / "paper1_mechanical/snapshots/archives/CASE_1"
    p1_dir.mkdir(parents=True)
    p1_archive = p1_dir / "nonlinear_snapshots_1000.npz"
    np.savez(p1_archive, mus=np.ones((2, 1)), fom_snapshots=np.ones((2, 3)))

    cfg = tmp_path / "cfg.yml"
    cfg.write_text(
        f"""
data_root: "{data_root}"
paper1:
  snapshots_root: "paper1_mechanical/snapshots"
paper2:
  baseline_root: "paper2_thermomechanical/campaigns/baseline"
""".strip()
    )
    loc = load_data_locations(config_path=cfg)
    assert project2_final_archives(loc) == [p2_archive.resolve()]
    assert p1_archive.resolve() in paper1_archives(loc)
