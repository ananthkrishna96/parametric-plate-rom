from __future__ import annotations

import numpy as np
import pytest

from paramplate.rom.metrics import metric_gram, relative_metric_errors
from paramplate.rom.pod import (
    compute_metric_pod,
    compute_transient_pod,
    orthogonality_error,
)
from paramplate.rom.predictors import PODGPR, PODILinear, PODIRBF, maximin_subset


def low_rank_data(n: int = 40, d: int = 18, rank: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(100)
    inputs = rng.uniform(-1.0, 1.0, (n, 2))
    coefficients = np.column_stack(
        [inputs[:, 0], inputs[:, 1], inputs[:, 0] * inputs[:, 1]]
    )[:, :rank]
    basis, _ = np.linalg.qr(rng.normal(size=(d, rank)))
    return inputs, coefficients @ basis.T


def test_metric_pod_is_orthonormal_and_reconstructs_training_span() -> None:
    _, snapshots = low_rank_data()
    metric = np.diag(np.linspace(1.0, 2.0, snapshots.shape[1]))
    pod = compute_metric_pod(snapshots, metric=metric, rank=3, metric_name="test metric")
    assert pod.modes.shape == (snapshots.shape[1], 3)
    assert orthogonality_error(pod, metric) < 1.0e-10
    np.testing.assert_allclose(metric_gram(pod.modes, metric), np.eye(3), atol=1.0e-10)
    assert np.nanmax(pod.projection_errors(snapshots, metric)) < 1.0e-10


def test_transient_pod_reorthonormalizes_in_fe_metric() -> None:
    _, snapshots = low_rank_data(n=50, d=16, rank=3)
    metric = np.diag(np.linspace(0.5, 1.5, snapshots.shape[1]))
    pod = compute_transient_pod(snapshots, finite_element_metric=metric, rank=3)
    assert pod.extraction.startswith("training-only Euclidean SVD")
    assert orthogonality_error(pod, metric) < 1.0e-10


def test_rbf_and_linear_predictors_reproduce_training_points() -> None:
    inputs, snapshots = low_rank_data(n=35, d=12, rank=3)
    targets = snapshots[:, :3]
    rbf = PODIRBF(smoothing=0.0, kernel="thin_plate_spline", degree=1).fit(inputs, targets)
    np.testing.assert_allclose(rbf.predict(inputs), targets, atol=2.0e-9)
    linear = PODILinear().fit(inputs, targets)
    np.testing.assert_allclose(linear.predict(inputs), targets, atol=1.0e-10)
    outside = linear.predict(np.array([[5.0, 5.0]]))
    assert np.all(np.isfinite(outside))


def test_gpr_predictor_and_maximin_subset() -> None:
    inputs, snapshots = low_rank_data(n=24, d=10, rank=2)
    targets = snapshots[:, :2]
    chosen = maximin_subset(inputs, 8, seed=100)
    assert len(chosen) == 8
    assert len(np.unique(chosen)) == 8
    gpr = PODGPR(optimizer_restarts=0, fit_limit=20, white_noise=1.0e-10).fit(inputs, targets)
    mean, std = gpr.predict(inputs[:4], return_std=True)
    assert mean.shape == std.shape == (4, 2)
    assert np.all(np.isfinite(mean))
    assert np.all(std >= 0.0)


def test_relative_metric_error_contract() -> None:
    reference = np.eye(3)
    errors = relative_metric_errors(reference, reference.copy())
    assert np.allclose(errors, 0.0)
    with pytest.raises(ValueError):
        relative_metric_errors(np.zeros((2, 3)), np.zeros((3, 2)))
