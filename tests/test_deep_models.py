from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from paramplate.rom.deep import (  # noqa: E402
    DeepTrainingConfig,
    DirectDLROM,
    PODDLROM,
    TransientPODDLROM,
)
from paramplate.rom.neural import PODNNConfig, PODNNRegressor  # noqa: E402


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(100)
    X = rng.normal(size=(24, 2))
    Y = np.column_stack([X[:, 0], X[:, 1], X[:, 0] * X[:, 1]])
    return X[:18], Y[:18], X[18:], Y[18:]


def test_pod_nn_training_and_shape() -> None:
    Xtr, Ytr, Xva, Yva = _data()
    model = PODNNRegressor(
        2,
        3,
        PODNNConfig(
            hidden_widths=(12, 10, 8),
            batch_size=8,
            maximum_epochs=6,
            patience=3,
            scheduler_patience=2,
            seed=100,
        ),
    )
    model.fit(Xtr, Ytr, Xva, Yva)
    assert model.predict(Xva).shape == Yva.shape
    assert model.history.best_epoch >= 0


def test_neural_model_initialization_is_seeded() -> None:
    nn_config = PODNNConfig(hidden_widths=(8, 6, 4), seed=37)
    first_nn = PODNNRegressor(2, 3, nn_config)
    second_nn = PODNNRegressor(2, 3, nn_config)
    for first, second in zip(first_nn.model.parameters(), second_nn.model.parameters(), strict=True):
        assert torch.equal(first, second)

    deep_config = DeepTrainingConfig(maximum_epochs=1, patience=1, seed=37)
    first_deep = PODDLROM(2, 3, coefficient_widths=(8, 6), parameter_widths=(8, 6), config=deep_config)
    second_deep = PODDLROM(2, 3, coefficient_widths=(8, 6), parameter_widths=(8, 6), config=deep_config)
    for key, value in first_deep.autoencoder.state_dict().items():
        assert torch.equal(value, second_deep.autoencoder.state_dict()[key])
    for key, value in first_deep.parameter_map.state_dict().items():
        assert torch.equal(value, second_deep.parameter_map.state_dict()[key])


def test_steady_pod_dl_and_direct_dl_smoke() -> None:
    Xtr, Ytr, Xva, Yva = _data()
    cfg = DeepTrainingConfig(batch_size=8, maximum_epochs=3, patience=2, seed=100)
    pod_dl = PODDLROM(
        2,
        3,
        latent_dimension=2,
        coefficient_widths=(8, 6),
        parameter_widths=(8, 6),
        config=cfg,
    )
    pod_dl.fit(Xtr, Ytr, Xva, Yva, joint_refinement_epochs=2)
    assert pod_dl.predict(Xva).shape == Yva.shape

    states = np.column_stack([Ytr, Ytr[:, :2]])
    states_val = np.column_stack([Yva, Yva[:, :2]])
    direct = DirectDLROM(
        2,
        5,
        field="displacement",
        encoder_widths=(10, 7),
        latent_dimension=2,
        parameter_widths=(8, 6),
        config=cfg,
    )
    direct.fit(Xtr, states, Xva, states_val, joint_refinement_epochs=2)
    assert direct.predict(Xva).shape == states_val.shape


def test_transient_pod_dl_feature_path_and_zero_time() -> None:
    rng = np.random.default_rng(101)
    amplitudes = np.repeat(np.linspace(2000.0, 8000.0, 6), 4)
    times = np.tile(np.linspace(0.0, 0.2, 4), 6)
    X = np.column_stack([amplitudes, times])
    Y = np.column_stack(
        [
            (amplitudes / 8000.0) * np.sin(5.0 * times),
            (amplitudes / 8000.0) * (1.0 - np.cos(5.0 * times)),
        ]
    )
    zero_time = np.isclose(times, 0.0)
    Y[zero_time] = 7.0  # Fitting must impose the prescribed zero initial state.
    model = TransientPODDLROM(
        2,
        2,
        hidden_widths=(12, 8),
        config=DeepTrainingConfig(batch_size=8, maximum_epochs=4, patience=2, seed=100),
    )
    model.fit(X[:16], Y[:16], X[16:], Y[16:])
    expected_train = Y[:16].copy()
    expected_train[np.isclose(X[:16, -1], 0.0)] = 0.0
    np.testing.assert_allclose(model.coordinate_standardizer.mean, expected_train.mean(axis=0))
    predicted = model.predict(X[16:])
    assert predicted.shape == Y[16:].shape
    np.testing.assert_allclose(predicted[np.isclose(X[16:, -1], 0.0)], 0.0, atol=0.0)
