from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paramplate.io.archives import validate_archive
from paramplate.rom.datasets import load_snapshot_dataset
from paramplate.rom.pipeline import prepare_rom_data
from paramplate.rom.suite import run_rom
from paramplate.thermomechanical.snapshots import pack_coupling_histories


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"


def test_bundled_archives_validate() -> None:
    mechanical = validate_archive(SAMPLE / "mechanical_smoke.npz")
    thermo = validate_archive(SAMPLE / "thermomechanical_smoke.npz")
    dynamics = validate_archive(SAMPLE / "dynamics_smoke.npz")
    assert mechanical["kind"] == "mechanical"
    assert thermo["kind"] == "thermomechanical"
    assert dynamics["kind"] == "dynamics"
    assert dynamics["n_trajectories"] == 20
    assert dynamics["n_times"] == 11


def test_thermomechanical_outputs_remain_separate() -> None:
    archive = SAMPLE / "thermomechanical_smoke.npz"
    displacement = load_snapshot_dataset(archive, study="thermomechanical", output="displacement")
    theta = load_snapshot_dataset(archive, study="thermomechanical", output="thermal_driver")
    assert displacement.output_unit == "m"
    assert theta.output_unit == "K/m"
    assert displacement.n_rows == theta.n_rows
    assert not np.shares_memory(displacement.snapshots, theta.snapshots)


def test_single_output_studies_reject_thermal_driver_label() -> None:
    with pytest.raises(ValueError, match="displacement as the only reduced output"):
        load_snapshot_dataset(
            SAMPLE / "mechanical_smoke.npz",
            study="mechanical",
            output="thermal_driver",
        )
    with pytest.raises(ValueError, match="displacement as the only reduced output"):
        load_snapshot_dataset(
            SAMPLE / "dynamics_smoke.npz",
            study="dynamics",
            output="thermal_driver",
        )


def test_coupling_histories_use_numeric_padded_storage() -> None:
    first = np.asarray([[1.0, 0.2, np.inf], [2.0, 0.01, 0.02]])
    second = np.asarray([[1.0, 0.3, np.inf]])
    packed, lengths = pack_coupling_histories([first, second])
    assert packed.dtype != object
    assert packed.shape == (2, 2, 3)
    assert lengths.tolist() == [2, 1]
    np.testing.assert_allclose(packed[0], first)
    assert np.isnan(packed[1, 1]).all()


def test_training_only_pod_and_trajectory_rows() -> None:
    dataset = load_snapshot_dataset(SAMPLE / "dynamics_smoke.npz", study="dynamics")
    prepared = prepare_rom_data(dataset, metric=None, rank=3, metric_name="Euclidean")
    assert prepared.split.level == "trajectory"
    assert (len(prepared.train_rows), len(prepared.validation_rows), len(prepared.test_rows)) == (176, 22, 22)
    train_ids = set(dataset.trajectory_ids[prepared.train_rows])
    test_ids = set(dataset.trajectory_ids[prepared.test_rows])
    assert train_ids.isdisjoint(test_ids)


def test_projection_and_predictive_rom_smoke_paths() -> None:
    mechanical = load_snapshot_dataset(SAMPLE / "mechanical_smoke.npz", study="mechanical")
    projection = run_rom(mechanical, method="pod-proj", rank=4)
    assert projection.summary.maximum < 1.0e-10
    rbf = run_rom(
        mechanical,
        method="podi-rbf",
        rank=4,
        options={"neighbors": 24, "smoothing": 1.0e-10},
    )
    assert rbf.predicted_fields.shape == (rbf.n_test, mechanical.n_dofs)
    assert np.all(np.isfinite(rbf.predicted_fields))


def test_transient_predictor_enforces_zero_initial_state() -> None:
    dataset = load_snapshot_dataset(SAMPLE / "dynamics_smoke.npz", study="dynamics")
    result = run_rom(dataset, method="podi-linear", rank=3, options={"linear_fit_limit": 200})
    prepared = prepare_rom_data(dataset, metric=None, rank=3, metric_name="Euclidean")
    test_inputs = dataset.parameters[prepared.test_rows]
    zero = np.isclose(test_inputs[:, -1], 0.0)
    assert zero.any()
    np.testing.assert_allclose(result.predicted_fields[zero], 0.0, atol=1.0e-14)


def test_method_inventory_rejects_unsupported_combinations() -> None:
    mechanical = load_snapshot_dataset(SAMPLE / "mechanical_smoke.npz", study="mechanical")
    dynamics = load_snapshot_dataset(SAMPLE / "dynamics_smoke.npz", study="dynamics")
    with pytest.raises(ValueError, match="not an active thesis method"):
        run_rom(mechanical, method="pod-dl-rom", rank=3, options={"maximum_epochs": 1})
    with pytest.raises(ValueError, match="not an active thesis method"):
        run_rom(dynamics, method="dl-rom", rank=3, options={"maximum_epochs": 1})
