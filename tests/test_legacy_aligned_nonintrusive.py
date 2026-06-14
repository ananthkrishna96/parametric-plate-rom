"""Tests for legacy-aligned non-intrusive ROM implementations."""

from __future__ import annotations

import numpy as np

from paramplate.rom.intrusive import LoadedPOD
from paramplate.rom.nonintrusive import (
    LEGACY_PODAE_CONFIG,
    LEGACY_PODGPR_CONFIG,
    LEGACY_PODI_CONFIG,
    LEGACY_PODNN_CONFIG,
    evaluate_coefficient_regressor,
    make_coefficient_regressor,
)


def _toy_problem(n=18, rank=3, dofs=7):
    rng = np.random.default_rng(123)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    C = np.column_stack([
        1.0 + 0.5 * X[:, 0],
        X[:, 1] ** 2,
        np.sin(X[:, 0] + X[:, 1]),
    ])[:, :rank]
    B = rng.normal(size=(dofs, rank))
    Q, _ = np.linalg.qr(B)
    B = Q[:, :rank]
    mean = rng.normal(scale=0.1, size=dofs)
    S = C @ B.T + mean
    train = np.arange(0, 12)
    test = np.arange(12, n)
    pod = LoadedPOD(
        basis=B,
        mean=mean,
        singular_values=np.ones(rank),
        energy=np.ones(rank) / rank,
        cumulative_energy=np.linspace(1.0 / rank, 1.0, rank),
        train_coefficients=C[train],
        test_coefficients=C[test],
        train_indices=train,
        test_indices=test,
        centered=True,
    )
    return X, C, S, pod, train, test


def test_legacy_defaults_are_the_notebook_values():
    assert LEGACY_PODI_CONFIG["seed"] == 100
    assert LEGACY_PODI_CONFIG["method_list"] == ("rbf", "linear")
    assert LEGACY_PODI_CONFIG["rbf_kwargs"]["kernel"] == "thin_plate_spline"
    assert LEGACY_PODGPR_CONFIG["alpha"] == 1e-9
    assert LEGACY_PODGPR_CONFIG["n_restarts_optimizer"] == 10
    assert LEGACY_PODNN_CONFIG["hidden_layers"] == (128, 96, 64)
    assert LEGACY_PODNN_CONFIG["activation"] == "elu"
    assert LEGACY_PODNN_CONFIG["coeff_epochs"] == 2500
    assert LEGACY_PODNN_CONFIG["field_epochs"] == 275
    assert LEGACY_PODAE_CONFIG["ae_hidden"] == (128, 96, 64)
    assert LEGACY_PODAE_CONFIG["latent_hidden"] == (128, 96, 64)
    assert LEGACY_PODAE_CONFIG["end_to_end_finetune"] is True


def test_podi_rbf_and_linear_fit_predict_shapes():
    X, C, _, _, train, test = _toy_problem()
    for method in ("podi-rbf", "podi-linear"):
        model = make_coefficient_regressor(method)
        model.fit(X[train], C[train])
        pred = model.predict(X[test])
        assert pred.shape == C[test].shape
        assert np.all(np.isfinite(pred))


def test_gpr_uses_legacy_scaling_and_predicts_shape():
    X, C, _, _, train, test = _toy_problem()
    model = make_coefficient_regressor(
        "pod-gpr",
        config={"n_restarts_optimizer": 0, "fallback_n_restarts_optimizer": 0},
    )
    model.fit(X[train], C[train])
    pred = model.predict(X[test])
    assert pred.shape == C[test].shape
    assert np.all(np.isfinite(pred))


def test_nn_and_ae_legacy_architectures_with_short_training():
    X, C, S, pod, train, test = _toy_problem()
    fast_nn = dict(
        coeff_epochs=2,
        field_epochs=1,
        coeff_early_stopping_patience=2,
        field_early_stopping_patience=2,
        log_every=1000,
        hidden_layers=(8, 6),
        coeff_batch_size=32,
        field_batch_size=32,
    )
    fast_ae = dict(
        ae_epochs=2,
        latent_epochs=2,
        end_to_end_finetune=False,
        ae_early_stopping_patience=2,
        latent_early_stopping_patience=2,
        log_every=1000,
        ae_hidden=(8, 6),
        latent_hidden=(8, 6),
        ae_batch_size=32,
        latent_batch_size=32,
    )
    for method, cfg in [("pod-nn", fast_nn), ("pod-ae", fast_ae)]:
        model = make_coefficient_regressor(method, n_basis=pod.rank, config=cfg)
        metrics, pred_train, pred_test = evaluate_coefficient_regressor(
            method=method,
            field_name="w",
            regressor=model,
            parameters_train=X[train],
            parameters_test=X[test],
            coefficients_train=pod.train_coefficients,
            coefficients_test=pod.test_coefficients,
            pod=pod,
            snapshots_train=S[train],
            snapshots_test=S[test],
        )
        assert pred_train.shape == pod.train_coefficients.shape
        assert pred_test.shape == pod.test_coefficients.shape
        assert np.isfinite(metrics.field_relative_error_test)
