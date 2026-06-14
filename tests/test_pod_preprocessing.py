from __future__ import annotations

from pathlib import Path

import numpy as np

from paramplate.rom.datasets import ROMDataset
from paramplate.rom.pod import compute_pod, project_snapshots, reconstruct_snapshots, relative_frobenius_error
from paramplate.rom.preprocessing import build_pod_preprocessing, fit_parameter_scaler, save_pod_preprocessing
from paramplate.rom.splits import make_train_test_split


def test_compute_pod_reconstructs_low_rank_data():
    rng = np.random.default_rng(123)
    coeffs = rng.normal(size=(12, 3))
    basis = rng.normal(size=(3, 20))
    snapshots = coeffs @ basis

    pod = compute_pod(snapshots, energy_tol=0.999999, center=False)
    projected = project_snapshots(snapshots, pod)
    reconstructed = reconstruct_snapshots(projected, pod)

    assert pod.rank <= 3
    assert relative_frobenius_error(snapshots, reconstructed) < 1e-12


def test_train_test_split_is_reproducible_and_disjoint():
    a = make_train_test_split(20, train_fraction=0.75, seed=7)
    b = make_train_test_split(20, train_fraction=0.75, seed=7)

    np.testing.assert_array_equal(a.train_indices, b.train_indices)
    np.testing.assert_array_equal(a.test_indices, b.test_indices)
    assert set(a.train_indices).isdisjoint(set(a.test_indices))
    assert a.n_train == 15
    assert a.n_test == 5


def test_parameter_scaler_handles_constant_columns():
    params = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]])
    scaler = fit_parameter_scaler(params)
    scaled = scaler.transform(params)

    assert np.allclose(scaled.mean(axis=0), [0.0, 0.0])
    assert scaler.scale[1] == 1.0
    assert np.allclose(scaler.inverse_transform(scaled), params)


def test_build_and_save_pod_preprocessing(tmp_path: Path):
    rng = np.random.default_rng(9)
    params = rng.normal(size=(16, 2))
    snapshots = rng.normal(size=(16, 30))
    theta = rng.normal(size=(16, 10))

    ds = ROMDataset(
        source_path=tmp_path / "CASE_TEST" / "SNAPSHOTS__CASE_TEST.npz",
        project="paper2",
        parameters=params,
        snapshots=snapshots,
        theta_snapshots=theta,
        metadata={},
        info=None,
    )

    result = build_pod_preprocessing(ds, train_fraction=0.75, seed=1, energy_tol=0.99)
    assert result.split.n_train == 12
    assert result.split.n_test == 4
    assert result.pod_w.basis.shape[0] == 30
    assert result.pod_theta is not None
    assert result.pod_theta.basis.shape[0] == 10

    out = save_pod_preprocessing(result, tmp_path / "pod_out")
    assert (out / "pod_w.npz").exists()
    assert (out / "pod_theta.npz").exists()
    assert (out / "parameter_scaler.json").exists()
    assert (out / "train_test_split.json").exists()
    assert (out / "pod_summary.json").exists()
    assert (out / "energy_w.csv").exists()
