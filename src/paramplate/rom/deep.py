"""POD--DL-ROM, direct DL-ROM, and time-conditioned deep predictors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .features import TransientFeatureMap, impose_zero_time
from .preprocessing import Standardizer

try:  # Optional dependency; importing paramplate remains possible without it.
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None or nn is None:  # pragma: no cover - optional dependency
        raise RuntimeError("Deep ROM workflows require the optional `torch` dependency.")


def _seed_torch(seed: int) -> None:
    """Seed network initialization and data-order generators consistently."""

    _require_torch()
    value = int(seed)
    torch.manual_seed(value)
    if torch.cuda.is_available():  # pragma: no cover - hardware dependent
        torch.cuda.manual_seed_all(value)
    np.random.seed(value)


if nn is not None:
    class MLP(nn.Module):
        def __init__(
            self,
            input_dimension: int,
            output_dimension: int,
            hidden_widths: Sequence[int],
            *,
            activation: str = "elu",
            output_activation: nn.Module | None = None,
            layer_normalization: bool = False,
            dropout: float = 0.0,
        ):
            super().__init__()
            if activation == "elu":
                activation_factory = nn.ELU
            elif activation == "silu":
                activation_factory = nn.SiLU
            else:
                raise ValueError(f"Unsupported activation {activation!r}.")
            layers: list[nn.Module] = []
            previous = int(input_dimension)
            for width in hidden_widths:
                width = int(width)
                layers.append(nn.Linear(previous, width))
                if layer_normalization:
                    layers.append(nn.LayerNorm(width))
                layers.append(activation_factory())
                if dropout > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
                previous = width
            layers.append(nn.Linear(previous, int(output_dimension)))
            if output_activation is not None:
                layers.append(output_activation)
            self.network = nn.Sequential(*layers)

        def forward(self, values):
            return self.network(values)


    class Autoencoder(nn.Module):
        def __init__(
            self,
            state_dimension: int,
            latent_dimension: int,
            encoder_widths: Sequence[int],
            *,
            activation: str = "elu",
        ):
            super().__init__()
            self.encoder = MLP(
                state_dimension,
                latent_dimension,
                encoder_widths,
                activation=activation,
            )
            self.decoder = MLP(
                latent_dimension,
                state_dimension,
                tuple(reversed(tuple(int(x) for x in encoder_widths))),
                activation=activation,
            )

        def forward(self, values):
            return self.decoder(self.encoder(values))

else:  # pragma: no cover - optional dependency
    class MLP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class Autoencoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()


@dataclass(frozen=True)
class DeepTrainingConfig:
    batch_size: int = 256
    learning_rate: float = 5.0e-4
    weight_decay: float = 2.0e-6
    maximum_epochs: int = 1500
    patience: int = 120
    seed: int = 100
    device: str = "cpu"


@dataclass
class DeepHistory:
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_validation_loss: float = float("inf")
    stopped_early: bool = False


def _fit_network(
    model,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_validation: np.ndarray,
    Y_validation: np.ndarray,
    config: DeepTrainingConfig,
    *,
    loss_callback=None,
) -> DeepHistory:
    _require_torch()
    Xtr = torch.as_tensor(np.asarray(X_train, dtype=np.float32))
    Ytr = torch.as_tensor(np.asarray(Y_train, dtype=np.float32))
    Xva = torch.as_tensor(np.asarray(X_validation, dtype=np.float32))
    Yva = torch.as_tensor(np.asarray(Y_validation, dtype=np.float32))
    if Xtr.ndim != 2 or Ytr.ndim != 2 or Xva.ndim != 2 or Yva.ndim != 2:
        raise ValueError("Deep-ROM training arrays must be two-dimensional.")
    if len(Xtr) != len(Ytr) or len(Xva) != len(Yva):
        raise ValueError("Deep-ROM input and target row counts differ.")

    _seed_torch(config.seed)
    device = torch.device(config.device)
    model.to(device)
    dataset = torch.utils.data.TensorDataset(Xtr, Ytr)
    generator = torch.Generator().manual_seed(int(config.seed))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(int(config.batch_size), len(dataset)),
        shuffle=True,
        generator=generator,
    )
    Xva, Yva = Xva.to(device), Yva.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=max(5, int(config.patience // 4)),
        min_lr=1.0e-7,
    )
    mse = nn.MSELoss()
    history = DeepHistory()
    best_state = None
    stale = 0

    def loss_fn(predicted, target):
        return mse(predicted, target) if loss_callback is None else loss_callback(predicted, target, mse)

    for epoch in range(int(config.maximum_epochs)):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
            count += len(xb)
        train_loss = total / max(count, 1)
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(Xva), Yva).detach().cpu())
        scheduler.step(validation_loss)
        history.train_loss.append(train_loss)
        history.validation_loss.append(validation_loss)
        if validation_loss < history.best_validation_loss - 1.0e-12:
            history.best_validation_loss = validation_loss
            history.best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(config.patience):
            history.stopped_early = True
            break
    if best_state is None:
        raise RuntimeError("Deep-ROM training did not produce a finite validation checkpoint.")
    model.load_state_dict(best_state)
    return history


class PODDLROM:
    """Steady coefficient-space POD--DL-ROM.

    The executed default encoder is ``N -> 128 -> 96 -> 64 -> n_z`` and the
    decoder is mirrored. The parameter-to-latent map uses ``128/96/64`` ELU
    hidden layers. By default ``n_z=N`` as in the thermomechanical campaigns.
    """

    def __init__(
        self,
        input_dimension: int,
        coefficient_dimension: int,
        *,
        latent_dimension: int | None = None,
        coefficient_widths: Sequence[int] = (128, 96, 64),
        parameter_widths: Sequence[int] = (128, 96, 64),
        config: DeepTrainingConfig | None = None,
    ):
        _require_torch()
        self.input_dimension = int(input_dimension)
        self.coefficient_dimension = int(coefficient_dimension)
        self.latent_dimension = int(latent_dimension or coefficient_dimension)
        self.coefficient_widths = tuple(int(x) for x in coefficient_widths)
        self.parameter_widths = tuple(int(x) for x in parameter_widths)
        self.config = config or DeepTrainingConfig()
        _seed_torch(self.config.seed)
        self.autoencoder = Autoencoder(
            self.coefficient_dimension,
            self.latent_dimension,
            self.coefficient_widths,
            activation="elu",
        )
        self.parameter_map = MLP(
            self.input_dimension,
            self.latent_dimension,
            self.parameter_widths,
            activation="elu",
        )
        self.history: dict[str, DeepHistory] = {}

    def fit(
        self,
        inputs_train: np.ndarray,
        coefficients_train: np.ndarray,
        inputs_validation: np.ndarray,
        coefficients_validation: np.ndarray,
        *,
        joint_refinement_epochs: int | None = None,
    ) -> "PODDLROM":
        _require_torch()
        coeff_train = np.asarray(coefficients_train, dtype=np.float32)
        coeff_val = np.asarray(coefficients_validation, dtype=np.float32)
        self.history["autoencoder"] = _fit_network(
            self.autoencoder,
            coeff_train,
            coeff_train,
            coeff_val,
            coeff_val,
            self.config,
        )
        device = torch.device(self.config.device)
        self.autoencoder.to(device).eval()
        with torch.no_grad():
            latent_train = self.autoencoder.encoder(torch.as_tensor(coeff_train, device=device)).cpu().numpy()
            latent_val = self.autoencoder.encoder(torch.as_tensor(coeff_val, device=device)).cpu().numpy()
        self.history["latent_map"] = _fit_network(
            self.parameter_map,
            inputs_train,
            latent_train,
            inputs_validation,
            latent_val,
            self.config,
        )

        class DeployedModel(nn.Module):
            def __init__(self, parameter_map, decoder):
                super().__init__()
                self.parameter_map = parameter_map
                self.decoder = decoder

            def forward(self, x):
                return self.decoder(self.parameter_map(x))

        deployed = DeployedModel(self.parameter_map, self.autoencoder.decoder)
        joint_cfg = DeepTrainingConfig(
            **{
                **asdict(self.config),
                "maximum_epochs": int(joint_refinement_epochs or max(20, self.config.maximum_epochs // 3)),
                "seed": self.config.seed + 1,
            }
        )
        self.history["joint_refinement"] = _fit_network(
            deployed,
            inputs_train,
            coeff_train,
            inputs_validation,
            coeff_val,
            joint_cfg,
        )
        return self

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        _require_torch()
        X = np.asarray(inputs, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        device = torch.device(self.config.device)
        self.parameter_map.to(device).eval()
        self.autoencoder.decoder.to(device).eval()
        with torch.no_grad():
            latent = self.parameter_map(torch.as_tensor(X, device=device))
            result = self.autoencoder.decoder(latent).cpu().numpy()
        return np.asarray(result, dtype=float)


class DirectDLROM:
    """Field-specific direct DL-ROM for steady thermomechanical outputs."""

    DEFAULT_WIDTHS = {
        "displacement": (2048, 1024, 512, 256),
        "thermal_driver": (1024, 512, 256, 128),
    }
    DEFAULT_LATENT = {"displacement": 6, "thermal_driver": 4}

    def __init__(
        self,
        input_dimension: int,
        state_dimension: int,
        *,
        field: str,
        encoder_widths: Sequence[int] | None = None,
        latent_dimension: int | None = None,
        parameter_widths: Sequence[int] = (128, 96, 64),
        config: DeepTrainingConfig | None = None,
    ):
        _require_torch()
        if field not in self.DEFAULT_WIDTHS:
            raise ValueError("field must be displacement or thermal_driver.")
        self.field = field
        self.input_dimension = int(input_dimension)
        self.state_dimension = int(state_dimension)
        self.encoder_widths = tuple(encoder_widths or self.DEFAULT_WIDTHS[field])
        self.latent_dimension = int(latent_dimension or self.DEFAULT_LATENT[field])
        self.parameter_widths = tuple(int(x) for x in parameter_widths)
        self.config = config or DeepTrainingConfig()
        _seed_torch(self.config.seed)
        self.autoencoder = Autoencoder(
            self.state_dimension,
            self.latent_dimension,
            self.encoder_widths,
            activation="elu",
        )
        self.parameter_map = MLP(
            self.input_dimension,
            self.latent_dimension,
            self.parameter_widths,
            activation="elu",
        )
        self.history: dict[str, DeepHistory] = {}

    def fit(
        self,
        inputs_train: np.ndarray,
        states_train: np.ndarray,
        inputs_validation: np.ndarray,
        states_validation: np.ndarray,
        *,
        joint_refinement_epochs: int | None = None,
    ) -> "DirectDLROM":
        _require_torch()
        Ytr = np.asarray(states_train, dtype=np.float32)
        Yva = np.asarray(states_validation, dtype=np.float32)
        self.history["autoencoder"] = _fit_network(self.autoencoder, Ytr, Ytr, Yva, Yva, self.config)
        device = torch.device(self.config.device)
        self.autoencoder.to(device).eval()
        with torch.no_grad():
            Ztr = self.autoencoder.encoder(torch.as_tensor(Ytr, device=device)).cpu().numpy()
            Zva = self.autoencoder.encoder(torch.as_tensor(Yva, device=device)).cpu().numpy()
        self.history["latent_map"] = _fit_network(
            self.parameter_map,
            inputs_train,
            Ztr,
            inputs_validation,
            Zva,
            self.config,
        )

        class DeployedModel(nn.Module):
            def __init__(self, mapper, decoder):
                super().__init__()
                self.mapper = mapper
                self.decoder = decoder

            def forward(self, x):
                return self.decoder(self.mapper(x))

        deployed = DeployedModel(self.parameter_map, self.autoencoder.decoder)
        joint_cfg = DeepTrainingConfig(
            **{
                **asdict(self.config),
                "maximum_epochs": int(joint_refinement_epochs or max(20, self.config.maximum_epochs // 3)),
                "seed": self.config.seed + 1,
            }
        )
        self.history["joint_refinement"] = _fit_network(
            deployed,
            inputs_train,
            Ytr,
            inputs_validation,
            Yva,
            joint_cfg,
        )
        return self

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        _require_torch()
        X = np.asarray(inputs, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        device = torch.device(self.config.device)
        self.parameter_map.to(device).eval()
        self.autoencoder.decoder.to(device).eval()
        with torch.no_grad():
            result = self.autoencoder.decoder(
                self.parameter_map(torch.as_tensor(X, device=device))
            ).cpu().numpy()
        return np.asarray(result, dtype=float)


class TransientPODDLROM:
    """Deployed time-conditioned POD--DL-ROM coordinate predictor."""

    def __init__(
        self,
        raw_input_dimension: int,
        coordinate_dimension: int,
        *,
        config: DeepTrainingConfig | None = None,
        feature_map: TransientFeatureMap | None = None,
        hidden_widths: Sequence[int] = (192, 160, 128),
        dropout: float = 0.02,
    ):
        _require_torch()
        self.raw_input_dimension = int(raw_input_dimension)
        self.coordinate_dimension = int(coordinate_dimension)
        self.config = config or DeepTrainingConfig()
        self.feature_map = feature_map or TransientFeatureMap(time_column=-1)
        self.hidden_widths = tuple(int(x) for x in hidden_widths)
        self.dropout = float(dropout)
        self.feature_standardizer: Standardizer | None = None
        self.coordinate_standardizer: Standardizer | None = None
        self.network = None
        self.history = DeepHistory()

    def fit(
        self,
        inputs_train: np.ndarray,
        coordinates_train: np.ndarray,
        inputs_validation: np.ndarray,
        coordinates_validation: np.ndarray,
    ) -> "TransientPODDLROM":
        raw_train = np.asarray(inputs_train, dtype=float)
        raw_validation = np.asarray(inputs_validation, dtype=float)
        coordinate_train = impose_zero_time(
            np.asarray(coordinates_train, dtype=float),
            raw_train,
            time_column=-1,
        )
        coordinate_validation = impose_zero_time(
            np.asarray(coordinates_validation, dtype=float),
            raw_validation,
            time_column=-1,
        )
        Ftr = self.feature_map.fit_transform(raw_train)
        Fva = self.feature_map.transform(raw_validation)
        self.feature_standardizer = Standardizer.fit(Ftr)
        Ftr = self.feature_standardizer.transform(Ftr)
        Fva = self.feature_standardizer.transform(Fva)
        self.coordinate_standardizer = Standardizer.fit(coordinate_train)
        Ctr = self.coordinate_standardizer.transform(coordinate_train)
        Cva = self.coordinate_standardizer.transform(coordinate_validation)
        _seed_torch(self.config.seed)
        self.network = MLP(
            Ftr.shape[1],
            self.coordinate_dimension,
            self.hidden_widths,
            activation="silu",
            layer_normalization=True,
            dropout=self.dropout,
        )
        self.history = _fit_network(
            self.network,
            Ftr,
            Ctr,
            Fva,
            Cva,
            self.config,
        )
        return self

    def predict(self, inputs: np.ndarray, *, impose_initial_state: bool = True) -> np.ndarray:
        _require_torch()
        if (
            self.network is None
            or self.feature_standardizer is None
            or self.coordinate_standardizer is None
        ):
            raise RuntimeError("TransientPODDLROM has not been fitted.")
        raw = np.asarray(inputs, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        features = self.feature_standardizer.transform(self.feature_map.transform(raw)).astype(np.float32)
        device = torch.device(self.config.device)
        self.network.to(device).eval()
        with torch.no_grad():
            output = self.network(torch.as_tensor(features, device=device)).cpu().numpy()
        result = self.coordinate_standardizer.inverse_transform(np.asarray(output, dtype=float))
        return impose_zero_time(result, raw, time_column=-1) if impose_initial_state else result


def save_torch_payload(path: str | Path, payload: dict[str, Any]) -> Path:
    _require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target
